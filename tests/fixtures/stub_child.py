"""Stand-in for ``_child.py`` so the runner can be tested without Triton.

It speaks the same protocol: ``stub_child.py <mode> <request.json> <events.jsonl>``.
The behaviour is chosen by the kernel's ``target`` string in the request.
"""

import json
import os
import signal
import subprocess
import sys
import time

mode, request_path, events_path = sys.argv[1:4]
with open(request_path) as handle:
    request = json.load(handle)


def emit(event, **fields):
    with open(events_path, "a") as handle:
        handle.write(json.dumps({"ev": event, **fields}) + "\n")


emit("start", mode=mode)
if mode == "probe":
    emit(
        "result", payload={"triton_version": "9.9.9", "python": "3.12.0", "site_paths": ["/base/site"], "backends": {}}
    )
    sys.exit(0)
if mode == "static":
    kernels = {k["id"]: {"targets": {t: [] for t in request["targets"]}} for k in request["kernels"]}
    for kernel in request["kernels"]:
        if "predict" in kernel["target"]:
            kernels[kernel["id"]]["targets"] = {
                t: [
                    {
                        "rule": "libdevice-missing",
                        "symbol": "tgamma",
                        "detail": "libdevice.tgamma is not defined",
                        "table": "hip",
                        "line": 7,
                    }
                ]
                for t in request["targets"]
                if t.startswith("hip")
            }
    emit("result", payload={"kernels": kernels})
    sys.exit(0)

behaviour = request["kernel"]["target"].rpartition(":")[2]
emit("stage_begin", stage="frontend")
emit("stage_end", stage="frontend", seconds=0.01)
emit("stage_begin", stage="ttgir")
if behaviour == "sleep":
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])  # a grandchild, like ptxas
    time.sleep(600)
elif behaviour == "alloc":
    hog = bytearray(300 * 1024 * 1024)
    time.sleep(600)
elif behaviour == "abort":
    sys.stderr.write("LLVM ERROR: something broke inside the compiler\n")
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGABRT)
elif behaviour == "fail":
    emit(
        "result",
        payload={
            "ok": False,
            "stage": "ttgir",
            "seconds": 0.2,
            "stage_seconds": {"frontend": 0.01},
            "location": {"file": "/repo/k.py", "line": 4},
            "error": {"type": "builtins.RuntimeError", "message": "PassManager::run failed", "line": 9, "column": 3},
        },
    )
    sys.stderr.write("/repo/k.py:9:3: error: op is not supported here\n")
    sys.exit(0)
emit("stage_end", stage="ttgir", seconds=0.02)
emit(
    "result",
    payload={
        "ok": True,
        "seconds": 0.3,
        "stage_seconds": {"frontend": 0.01, "ttgir": 0.02},
        "artifact_sizes": {"ttir": 10, "cubin": 20},
        "metadata": {"shared": 128},
        "ptxas": {"registers": 17},
        "peak_rss_bytes": 1000,
        "location": {"file": "/repo/k.py", "line": 4},
        "notes": [f"fallback={request.get('fallback_sys_path')}"],
    },
)
