"""Generate the 200 independent reviewer experiment entry points.

The generated files are intentionally small wrappers.  All scientific logic
lives in :mod:`single_job_runner`, while each wrapper fixes one seed, one
horizon and one model so a scheduler can submit it as a separate task.
Running this generator is idempotent and also refreshes the CSV/JSON job
manifest used by cluster launchers and the aggregation script.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SEEDS = (42, 123, 2024, 7, 99)
HORIZONS = (1, 3, 6, 9, 12)
MODELS = (
    "STG-Informer",
    "G-Informer",
    "ST-Informer",
    "Informer",
    "GRU",
    "Transformer",
    "ConvLSTM",
    "CNN-LSTM",
)

MODEL_FOLDER_NAMES = {
    "STG-Informer": "STG",
    "G-Informer": "G",
    "ST-Informer": "ST",
    "Informer": "Informer",
    "GRU": "GRU",
    "Transformer": "Transformer",
    "ConvLSTM": "ConvLSTM",
    "CNN-LSTM": "CNN-LSTM",
}


def job_id(model: str, seed: int, horizon: int) -> str:
    safe_model = "".join(ch if ch.isalnum() else "_" for ch in model)
    return f"seed{seed:03d}_horizon{horizon:02d}_{safe_model}"


def script_name(model: str, seed: int, horizon: int) -> str:
    safe_model = "".join(ch.lower() if ch.isalnum() else "_" for ch in model)
    return f"seed{seed:03d}_horizon{horizon:02d}_{safe_model}.py"


def result_folder_name(model: str, seed: int, horizon: int) -> str:
    """Return the flat result directory name for one fixed task."""

    try:
        short_model = MODEL_FOLDER_NAMES[model]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model}") from exc
    return f"{short_model}_seed{int(seed)}_horizon{int(horizon)}"


def wrapper_text(model: str, seed: int, horizon: int) -> str:
    identifier = job_id(model, seed, horizon)
    return f'''"""Run reviewer job {identifier}.

Fixed combination: model={model}, seed={seed}, horizon={horizon}.
"""
from pathlib import Path
import sys

JOB_ROOT = Path(__file__).resolve().parent
if str(JOB_ROOT) not in sys.path:
    sys.path.insert(0, str(JOB_ROOT))

from single_job_runner import run_job_cli

MODEL = {model!r}
SEED = {seed}
HORIZON = {horizon}
JOB_ID = {identifier!r}


if __name__ == "__main__":
    raise SystemExit(run_job_cli(model=MODEL, seed=SEED, horizon=HORIZON))
'''


def build_manifest(root: Path) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for seed in SEEDS:
        for horizon in HORIZONS:
            for model in MODELS:
                identifier = job_id(model, seed, horizon)
                filename = script_name(model, seed, horizon)
                jobs.append(
                    {
                        "job_index": len(jobs) + 1,
                        "job_id": identifier,
                        "seed": seed,
                        "horizon": horizon,
                        "model": model,
                        "script": filename,
                        "output_directory": result_folder_name(model, seed, horizon),
                        "command": f"python {filename}",
                    }
                )
    if len(jobs) != 200:
        raise AssertionError(f"Expected 200 jobs, generated {len(jobs)}")
    return jobs


def write_jobs(root: Path, overwrite: bool) -> list[dict[str, object]]:
    jobs = build_manifest(root)
    for item in jobs:
        path = root / str(item["script"])
        if path.exists() and not overwrite:
            # Existing generated wrappers are deterministic; leave them in
            # place unless --overwrite is requested so local edits survive.
            continue
        path.write_text(
            wrapper_text(str(item["model"]), int(item["seed"]), int(item["horizon"])),
            encoding="utf-8",
        )

    fieldnames = [
        "job_index",
        "job_id",
        "seed",
        "horizon",
        "model",
        "script",
        "output_directory",
        "command",
    ]
    with (root / "job_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(jobs)
    (root / "job_manifest.json").write_text(
        json.dumps(jobs, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the 200EX job wrappers and manifest")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    jobs = write_jobs(args.root.resolve(), args.overwrite)
    print(f"Generated manifest with {len(jobs)} jobs under {args.root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
