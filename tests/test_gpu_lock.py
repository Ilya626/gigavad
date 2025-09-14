import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference_gigaam import acquire_gpu_lock, release_gpu_lock


def test_acquire_gpu_lock_removes_stale(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    lock_path.write_text("999999", encoding="utf-8")
    start = time.time()
    acquired, pid = acquire_gpu_lock(lock_path, timeout_s=3)
    elapsed = time.time() - start
    assert acquired
    assert pid == os.getpid()
    assert elapsed < 1
    assert lock_path.read_text(encoding="utf-8").strip() == str(pid)
    release_gpu_lock(lock_path, pid)
    assert not lock_path.exists()
