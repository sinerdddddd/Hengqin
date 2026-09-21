"""Aggregate completed 200EX job folders into reviewer-level tables.

Only completed folders listed in ``job_manifest.csv`` are considered.  Every
row keeps the originating ``job_id``, model, seed and horizon, so missing or
failed scheduler tasks remain visible in ``job_collection_status.csv``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _deduplicate(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    """Concatenate task files and remove intentional baseline repetitions.

    Each neural task writes persistence and linear-trend rows so it can be
    evaluated in isolation.  Those baseline rows occur eight times for every
    seed/horizon when jobs are aggregated; de-duplicating by the scientific
    identity restores the same population as the monolithic runner.
    """

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    present = [key for key in keys if key in combined.columns]
    if present:
        combined = combined.drop_duplicates(subset=present, keep="first", ignore_index=True)
    return combined


def aggregate(
    root: Path,
    output: Path | None = None,
    outputs_root: Path | None = None,
) -> dict[str, int]:
    root = root.resolve()
    output = (output or (root / "aggregated_outputs")).resolve()
    outputs_root = (outputs_root or root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "job_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    uncertainty_frames: list[pd.DataFrame] = []
    hazard_frames: list[pd.DataFrame] = []
    recursive_frames: list[pd.DataFrame] = []
    wilcoxon_frames: list[pd.DataFrame] = []
    parameter_frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []

    for record in manifest.to_dict("records"):
        job_id = str(record["job_id"])
        result_directory = str(record.get("output_directory", job_id))
        job_dir = outputs_root / result_directory
        status_path = job_dir / f"job_status_{job_id}.json"
        status: dict[str, object] = {}
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                status = {"status": "invalid_status_json"}
        status_rows.append(
            {
                "job_index": record["job_index"],
                "job_id": job_id,
                "seed": record["seed"],
                "horizon": record["horizon"],
                "model": record["model"],
                "status": status.get("status", "missing"),
                "error_type": status.get("error_type", ""),
                "error": status.get("error", ""),
                "output_directory": str(job_dir),
            }
        )
        if status.get("status") != "completed":
            continue
        for filename, destination in (
            (f"metrics_{job_id}.csv", metric_frames),
            (f"predictions_{job_id}.csv", prediction_frames),
            (f"uncertainty_{job_id}.csv", uncertainty_frames),
            (f"relative_hazard_{job_id}.csv", hazard_frames),
            (f"recursive_forecasts_{job_id}.csv", recursive_frames),
            (f"wilcoxon_{job_id}.csv", wilcoxon_frames),
            (f"model_parameter_count_{job_id}.csv", parameter_frames),
        ):
            frame = _read_csv(job_dir / filename)
            if not frame.empty:
                destination.append(frame)

    status_frame = pd.DataFrame(status_rows)
    status_frame.to_csv(output / "job_collection_status.csv", index=False)
    frames = {
        "metrics_by_job.csv": metric_frames,
        "predictions_by_job.csv": prediction_frames,
        "uncertainty_by_job.csv": uncertainty_frames,
        "relative_hazard_by_job.csv": hazard_frames,
        "recursive_forecasts_by_job.csv": recursive_frames,
        "wilcoxon_by_job.csv": wilcoxon_frames,
        "model_parameter_counts_by_job.csv": parameter_frames,
    }
    counts: dict[str, int] = {
        "manifest_jobs": int(len(manifest)),
        "completed_jobs": int((status_frame["status"] == "completed").sum())
        if not status_frame.empty
        else 0,
        "failed_jobs": int((status_frame["status"] == "failed").sum())
        if not status_frame.empty
        else 0,
    }
    dedupe_keys = {
        "metrics_by_job.csv": ["seed", "horizon", "model"],
        "predictions_by_job.csv": [
            "seed",
            "horizon",
            "model",
            "point_index",
            "target_start",
            "step",
        ],
        "uncertainty_by_job.csv": ["seed", "horizon", "model"],
        "relative_hazard_by_job.csv": ["seed", "horizon", "model", "point_index"],
        "recursive_forecasts_by_job.csv": [
            "seed",
            "horizon",
            "model",
            "point_index",
            "origin_target_start",
            "step",
        ],
        "wilcoxon_by_job.csv": ["seed", "horizon", "comparison"],
        "model_parameter_counts_by_job.csv": ["seed", "horizon", "model"],
    }
    combined_frames: dict[str, pd.DataFrame] = {}
    for filename, grouped in frames.items():
        combined = _deduplicate(grouped, dedupe_keys[filename])
        combined_frames[filename] = combined
        combined.to_csv(output / filename, index=False)
        counts[filename] = int(len(combined))

    completed_metrics = combined_frames["metrics_by_job.csv"]
    if not completed_metrics.empty and {"horizon", "model"}.issubset(completed_metrics.columns):
        metric_columns = [
            column
            for column in (
                "MAE_mm",
                "RMSE_mm",
                "R2",
                "MAPE_percent",
                "WAPE_percent",
                "rate_MAE_mm_per_step",
                "rate_MAE_mm_per_day",
            )
            if column in completed_metrics.columns
        ]
        if metric_columns:
            summary = (
                completed_metrics.groupby(["horizon", "model"])[metric_columns]
                .agg(["mean", "std"])
                .reset_index()
            )
            summary.to_csv(output / "metrics_summary_by_horizon_model.csv", index=False)
            counts["metrics_summary_by_horizon_model.csv"] = int(len(summary))

    (output / "aggregation_summary.json").write_text(
        json.dumps(counts, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate completed 200EX outputs")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=None,
        help="directory containing one result subdirectory per job (default: ROOT)",
    )
    args = parser.parse_args()
    counts = aggregate(args.root, args.output, args.outputs_root)
    print(json.dumps(counts, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
