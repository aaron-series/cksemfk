import os
import sys

REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr

import json
import traceback
import time
from typing import Any, Dict

# ===== 응답 헬퍼 =====
def ok(data: Any, meta: Dict = None):
    return {
        "status": "success",
        "timestamp": time.time(),
        "data": data,
        "meta": meta or {} 
    }

def error(stage: str, e: Exception):
    return {
        "status": "error",
        "stage": stage,
        "message": str(e),
        "type": e.__class__.__name__,
        "traceback": traceback.format_exc(),
        "timestamp": time.time()
    }

# ===== vLLM 초기화 =====
try:
    # 🔥 실제 구조 기반 import
    from chandra.model import generate_vllm, BatchInputItem
    from chandra.model.util import Image

    print("[WORKER] Loading vLLM engine...", file=sys.stderr)

    # ✅ Chandra 방식 정식 생성
    ENGINE = generate_vllm

    print("[WORKER] vLLM ready", file=sys.stderr)

except Exception as e:
    REAL_STDOUT.write(json.dumps(error("vllm_init", e)) + "\n")
    REAL_STDOUT.flush()
    sys.exit(1)


# ===== 추론 =====
def run_inference(image_path: str):
    start = time.time()
    try:
        # 1. 이미지 로드 (Chandra util 구조)
        img = Image.open(image_path)

        item = BatchInputItem(image=img, prompt="Analyze this document.")

        # 2. 추론
        result = ENGINE([item])

        output = result[0] if isinstance(result, list) else result

        return ok(
            data=output,
            meta={
                "latency": round(time.time() - start, 3),
                "image": os.path.basename(image_path)
            }
        )

    except Exception as e:
        return error("inference", e)


# ===== 스트리밍 인터페이스 =====
def main_loop():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            image_path = req.get("image")

            if not image_path:
                REAL_STDOUT.write(json.dumps(error("input", ValueError("no image field"))) + "\n")
                REAL_STDOUT.flush()
                continue

            result = run_inference(image_path)

        except Exception as e:
            result = error("protocol", e)

        REAL_STDOUT.write(json.dumps(result, ensure_ascii=False) + "\n")
        REAL_STDOUT.flush()


if __name__ == "__main__":

    if not sys.stdin.isatty():
        main_loop()
        sys.exit(0)

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image")
    args = p.parse_args()

    REAL_STDOUT.write(json.dumps(run_inference(args.image), ensure_ascii=False) + "\n")
