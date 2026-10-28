"""A failing or orphaned rank must end the whole local job quickly."""

import os
import signal
import subprocess
import sys
import time

_JOB = """
import os, sys, threading, time
from dream_trainer.utils.entrypoint import entrypoint

@entrypoint
def main():
    print("rank", os.environ["RANK"], os.getpid(), flush=True)
    if sys.argv[1] == "raise" and os.environ["RANK"] == "0":
        # Non-daemon thread keeps a normal interpreter shutdown from finishing (like wandb).
        threading.Thread(target=time.sleep, args=(3600,)).start()
        raise RuntimeError("boom")
    time.sleep(3600)

main()
"""

_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"}  # world_size 2; ranks never touch CUDA


def _launch(mode: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _JOB, mode],
        env=_ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def test_rank_exception_ends_job_fast():
    job = _launch("raise")
    try:
        start = time.monotonic()
        out, _ = job.communicate(timeout=60)
        assert job.returncode != 0
        assert "boom" in out
        assert time.monotonic() - start < 60
    finally:
        job.kill()


def test_ranks_die_with_launcher():
    job = _launch("sleep")
    try:
        assert job.stdout is not None
        ranks = []
        while len(ranks) < 2:
            line = job.stdout.readline()
            assert line, "job exited before both ranks started"
            if line.startswith("rank "):
                ranks.append(int(line.split()[2]))
        job.send_signal(signal.SIGKILL)
        job.wait()

        deadline = time.monotonic() + 10
        while any(map(_alive, ranks)) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not any(map(_alive, ranks)), f"orphaned ranks: {ranks}"
    finally:
        job.kill()
