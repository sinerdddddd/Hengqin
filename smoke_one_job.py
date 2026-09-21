"""Run one representative 200EX smoke test without retaining outputs."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test one fixed 200EX entry point")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--script",
        default="seed042_horizon01_stg_informer.py",
        help="one wrapper filename under ROOT (default: representative STG-Informer job)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke-points", type=int, default=16)
    args = parser.parse_args()

    root = args.root.resolve()
    script = root / args.script
    if not script.exists():
        parser.error(f"entry point does not exist: {script}")
    command = [
        sys.executable,
        str(script),
        "--smoke",
        "--device",
        args.device,
        "--torch-threads",
        "1",
        "--smoke-points",
        str(args.smoke_points),
    ]
    completed = subprocess.run(command, cwd=str(root), text=True)
    if completed.returncode:
        return completed.returncode
    print(f"One-job smoke test passed: {args.script}; no output folder was retained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
