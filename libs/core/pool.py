import subprocess
import queue
import threading
import json
import os
import time
import logging

logger = logging.getLogger(__name__)


class VLLMWorker:
    def __init__(self, python_bin, worker_script): 
        self.python_bin = python_bin
        self.worker_script = worker_script
        self.proc = None
        self.lock = threading.Lock()
        self.spawn()

    def spawn(self):
        self.proc = subprocess.Popen(
            [self.python_bin, self.worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=os.environ.copy()
        )

        logger.info(f"[POOL] worker spawned pid={self.proc.pid}")

    def health(self):
        return self.proc and self.proc.poll() is None

    def restart(self):
        try:
            self.proc.kill()
        except:
            pass

        self.spawn()

    def run(self, payload: dict, timeout=300):
        with self.lock:
            if not self.health():
                logger.warning("[POOL] worker dead → restart")
                self.restart()

            try:
                self.proc.stdin.write(json.dumps(payload) + "\n")
                self.proc.stdin.flush()

                start = time.time()

                while True:
                    if time.time() - start > timeout:
                        raise TimeoutError("vllm timeout")

                    line = self.proc.stdout.readline()

                    if not line:
                        raise RuntimeError("broken pipe")

                    return json.loads(line)

            except Exception as e:
                logger.error(f"[POOL] error: {e}")
                self.restart()

                return {
                    "status": "error",
                    "stage": "pool",
                    "message": str(e)
                }


class WorkerPool:
    def __init__(self, python_bin, worker_script, size=4):
        self.workers = queue.Queue()

        for _ in range(size):
            self.workers.put(
                VLLMWorker(python_bin, worker_script)
            )

    def run(self, image_path: str):
        worker = self.workers.get()

        try:
            return worker.run({"image": image_path})
        finally:
            self.workers.put(worker)
