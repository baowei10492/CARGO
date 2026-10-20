from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture reproducibility metadata.")
    parser.add_argument("--output", default="environment/system_info.json")
    args = parser.parse_args()
    report = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "pip_freeze": _command_output(
            [sys.executable, "-m", "pip", "freeze"]
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()

