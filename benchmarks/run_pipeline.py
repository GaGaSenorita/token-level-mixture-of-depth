"""Persistent local pipeline; training and timing never compete for the GPU."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "latency_20261108"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    commands = [
        [sys.executable, "-u", "-m", "benchmarks.run_experiment", "--phase", "all"],
        [sys.executable, "-u", "-m", "benchmarks.measure_latency", "--config", str(OUT / "config_resolved.json"), "--run-dir", str(OUT)],
    ]
    start = time.time()
    for index, command in enumerate(commands):
        (OUT / "pipeline_status.json").write_text(json.dumps({"status": "running", "phase": index,
            "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
        print(json.dumps({"event": "pipeline_phase", "index": index, "command": command}), flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            record = {"status": "failed", "phase": index, "returncode": completed.returncode, "elapsed_seconds": time.time()-start}
            (OUT / "pipeline_status.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            raise SystemExit(completed.returncode)
    record = {"status": "complete", "elapsed_seconds": time.time()-start, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT / "pipeline_status.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
