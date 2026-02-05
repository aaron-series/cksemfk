# libs/api/app.py
import time
import os
import sys
import shutil
import logging
import uuid
import subprocess

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from libs.core.pool import WorkerPool
from libs.common.config import load_config

from pdf2image import convert_from_path 

from concurrent.futures import ThreadPoolExecutor


# Python 버전 체크
if sys.version_info < (3, 10):
    raise RuntimeError("Python 3.10+ is required.")


# ---------------------------------------------------------
# 설정 로드
# ---------------------------------------------------------
try:
    CONF = load_config()
except Exception as e:
    print(f"Failed to load config: {e}")
    sys.exit(1)


PROJECT_ROOT = CONF["project_root"]
SERVER_CONF = CONF["server"]
PATHS_CONF = CONF["paths"]
WORKER_CONF = CONF["worker"]

PDF_THREADS = CONF.get("pdf", {}).get("threads", 4)


# 경로 설정
LOG_DIR = os.path.join(PROJECT_ROOT, PATHS_CONF["logs_dir"])
TEMP_DIR = os.path.join(PROJECT_ROOT, PATHS_CONF["temp_dir"])
WORKER_SCRIPT = os.path.join(PROJECT_ROOT, WORKER_CONF["script_path"])

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


# ---------------------------------------------------------
# 로깅 설정
# ---------------------------------------------------------
log_file_path = os.path.join(LOG_DIR, "api_server.log")
logging.basicConfig(
    filename=log_file_path,
    level=getattr(logging, SERVER_CONF["log_level"].upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Python 경로 해석 (1회만)
# ---------------------------------------------------------
def resolve_python_bin() -> str:
    candidates = WORKER_CONF.get("python_path", [])

    logger.info(f"Resolving python from candidates: {candidates}")

    for rel_path in candidates:
        candidate = (
            os.path.join(PROJECT_ROOT, rel_path)
            if not rel_path.startswith("/") else rel_path
        )

        # 1) 존재 + 실행권한
        if not (os.path.isfile(candidate) and os.access(candidate, os.X_OK)):
            logger.info(f"[SKIP] not executable: {candidate}")
            continue

        try:
            # 2) 실제 동작 검증 ★핵심★
            test_code = (
                "import sys, json\n"
                "import chandra\n"           # ← chandra 존재 검증
                "print('OK')"
            )

            result = subprocess.run(
                [candidate, "-c", test_code],
                capture_output=True,
                text=True,
                timeout=5
            )

            logger.info(
                f"Python test: {candidate}\n"
                f"rc={result.returncode}\n"
                f"stdout={result.stdout}\n"
                f"stderr={result.stderr}"
            )

            if result.returncode == 0 and "OK" in result.stdout:
                logger.info(f"✅ Selected python: {candidate}")
                return candidate

        except Exception as e:
            logger.error(f"[ERROR] python test failed: {candidate} → {e}")

    # 다 실패
    msg = f"No valid python interpreter found from: {candidates}"
    logger.error(msg)
    raise RuntimeError(msg)


PYTHON_BIN = resolve_python_bin()


# ============================
# ✅ 핵심: Worker Pool 생성
# ============================
pool = WorkerPool(
    python_bin=PYTHON_BIN,
    worker_script=WORKER_SCRIPT,
    size=WORKER_CONF.get("pool_size", 1)
)


app = FastAPI(title="Chandra API", description="Supports Multi-page PDF Processing")


# ---------------------------------------------------------
# Worker 호출 (Pool 기반)
# ---------------------------------------------------------
def call_worker(image_path: str) -> dict:
    try:
        result = pool.run(image_path)

        if result.get("status") == "error":
            logger.error(
                f"Worker error:\n"
                f"stage={result.get('stage')}\n"
                f"message={result.get('message')}\n"
                f"type={result.get('type')}\n"
                f"traceback={result.get('traceback')}"
            )
        else:
            logger.info(f"Worker success: {result.get('meta')}")

        return result

    except Exception as e:
        logger.error(f"Pool error: {e}", exc_info=True)
        return {
            "status": "error",
            "stage": "pool",
            "message": str(e)
        }
        }

# ---------------------------------------------------------
# 페이지 처리
# ---------------------------------------------------------
def process_page(page_info):
    page_num, page_image_path = page_info
    result = call_worker(page_image_path)
    return page_num, result

# ---------------------------------------------------------
# API
# ---------------------------------------------------------
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    original_filename = file.filename
    temp_files = []

    pdf_path = os.path.join(TEMP_DIR, f"{request_id}_{original_filename}")
    temp_files.append(pdf_path)

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    response = {
        "filename": original_filename,
        "results": []
    }

    try:
        # PDF → 이미지
        images = convert_from_path(pdf_path)

        page_jobs = []

        for i, img in enumerate(images):
            page = i + 1
            img_path = os.path.join(
                TEMP_DIR, f"{request_id}_p{page}.jpg"
            )

            img.save(img_path, "JPEG")
            temp_files.append(img_path)

            page_jobs.append((page, img_path))

        # 🔥 병렬 처리 핵심
        with ThreadPoolExecutor(max_workers=PDF_THREADS) as ex:
            results = ex.map(process_page, page_jobs)

        has_error = False

        for page, result in results:
            if result.get("status") == "error":
                has_error = True

            response["results"].append({
                "page": page,
                **result
            })

        return JSONResponse(
            status_code=500 if has_error else 200,
            content=response
        )

    except Exception as e:
        logger.exception("server error")
        return JSONResponse(
            status_code=500,
            content={"status": "server_error", "message": str(e)}
        )

    finally:
        for p in temp_files:
            try:
                os.remove(p)
            except:
                pass


@app.get("/health")
def health():
    return {"status": "ok", "pdf_support": True}
