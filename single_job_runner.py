"""Run exactly one seed/horizon/model reviewer experiment.

This module is the shared engine used by the 200 generated entry points in
the 200EX root.  Keeping the training implementation in one place avoids 200
copies of the model code while each entry point remains an independently
submittable Python task.  Every production task writes to a unique directory
and includes ``job_id``, ``seed``, ``horizon`` and ``model`` in all tabular
outputs.

The smoke path deliberately writes to a temporary directory and removes it in
``finally``.  It therefore validates imports, data loading, window creation,
training and output serialization without leaving smoke artifacts in the
project tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

# The copied core makes 200EX self-contained.  The fallback path also permits
# running this module from a checkout where the core is one directory up.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
try:
    from reviewer_experiment_core import (
        ExperimentConfig,
        ForecastGRU,
        ForecastLSTM,
        ForecastTransformer,
        ForecastConvLSTM,
        ForecastCNNLSTM,
        VanillaTransformer,
        WindowStore,
        baseline,
        fit_model,
        load_export,
        mc_predict,
        metrics,
        recursive_mc_predict,
        set_seed,
    )
except ImportError:  # pragma: no cover - useful when called from a copied tree
    PARENT = HERE.parent
    if str(PARENT) not in sys.path:
        sys.path.insert(0, str(PARENT))
    from reviewer_complete_experiment import (
        ExperimentConfig,
        ForecastGRU,
        ForecastLSTM,
        ForecastTransformer,
        ForecastConvLSTM,
        ForecastCNNLSTM,
        VanillaTransformer,
        WindowStore,
        baseline,
        fit_model,
        load_export,
        mc_predict,
        metrics,
        recursive_mc_predict,
        set_seed,
    )


MODEL_SPECS: Mapping[str, tuple[type[torch.nn.Module], int, bool, bool]] = {
    "STG-Informer": (ForecastTransformer, 4, True, True),
    "G-Informer": (ForecastTransformer, 2, False, True),
    "ST-Informer": (ForecastTransformer, 3, True, False),
    "Informer": (ForecastTransformer, 1, False, False),
    "GRU": (ForecastGRU, 4, True, True),
    "Transformer": (VanillaTransformer, 4, True, True),
    "ConvLSTM": (ForecastConvLSTM, 4, True, True),
    "CNN-LSTM": (ForecastCNNLSTM, 4, True, True),
}

DEFAULT_SEEDS = (42, 123, 2024, 7, 99)
DEFAULT_HORIZONS = (1, 3, 6, 9, 12)
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


def make_job_id(model: str, seed: int, horizon: int) -> str:
    """Return a filesystem-safe, human-readable identifier."""

    safe_model = "".join(ch if ch.isalnum() else "_" for ch in model)
    return f"seed{int(seed):03d}_horizon{int(horizon):02d}_{safe_model}"


def result_folder_name(model: str, seed: int, horizon: int) -> str:
    """Return the flat result directory name used beside the task scripts."""

    try:
        short_model = MODEL_FOLDER_NAMES[model]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model!r}") from exc
    return f"{short_model}_seed{int(seed)}_horizon{int(horizon)}"


def _jsonable(value: Any) -> Any:
    """Convert numpy/torch scalar values for JSON metadata."""

    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(v) for v in value]
    return value


def _write_csv(path: Path, frame: pd.DataFrame, columns: Sequence[str] | None = None) -> None:
    """Write a CSV, retaining a useful header even when a result is empty."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty and columns:
        frame = pd.DataFrame(columns=list(columns))
    frame.to_csv(path, index=False)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=True), encoding="utf-8")


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _configure_threads(torch_threads: int | None) -> int | None:
    """Limit per-task CPU threads to avoid oversubscription on a cluster."""

    requested = torch_threads
    if requested is None:
        env_value = os.environ.get("HENGQIN_TORCH_THREADS") or os.environ.get("OMP_NUM_THREADS")
        if env_value:
            try:
                requested = max(1, int(env_value))
            except ValueError:
                requested = None
    if requested is None:
        # Parallel schedulers generally assign one CPU core per GPU task.  A
        # single PyTorch thread is a conservative default and can be raised by
        # --torch-threads or HENGQIN_TORCH_THREADS when more cores are granted.
        requested = 1
    try:
        torch.set_num_threads(int(requested))
        # set_num_interop_threads can only be called before parallel work starts.
        try:
            torch.set_num_interop_threads(int(requested))
        except RuntimeError:
            pass
    except (RuntimeError, ValueError):
        return None
    return int(requested)


def _load_dates(date_file: Path | None, n_acquisitions: int) -> pd.Series | None:
    if date_file is None:
        return None
    if not date_file.exists():
        raise FileNotFoundError(f"Date file not found: {date_file}")
    date_frame = pd.read_csv(date_file)
    if date_frame.shape[1] < 1:
        raise ValueError("Date file must contain at least one column")
    dates = pd.to_datetime(date_frame.iloc[:, 0], errors="coerce")
    if len(dates) != n_acquisitions:
        raise ValueError("--date-file must contain one date per acquisition")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("--date-file dates must be valid, unique, and strictly increasing")
    return dates


def _select_variant(data: tuple, use_coords: bool, use_geo: bool):
    """Select feature columns for a covariate ablation."""

    if use_coords and use_geo:
        return data
    cols = [0]
    if use_coords:
        cols.extend([1, 2])
    if use_geo:
        cols.append(3)
    x, y, ids, starts = data
    return x[..., cols], y, ids, starts


def _rate_denominators(starts: np.ndarray, horizon: int, dates: pd.Series | None) -> np.ndarray:
    if dates is None:
        return np.full(len(starts), float(horizon), dtype=np.float64)
    values = []
    for target_start in starts:
        # target_start is the zero-based index immediately after the input
        # window; target values run through target_start + horizon - 1.
        first_observed = max(int(target_start) - 1, 0)
        last_target = int(target_start) + int(horizon) - 1
        elapsed = (dates.iloc[last_target] - dates.iloc[first_observed]).days
        values.append(float(max(elapsed, 1)))
    return np.asarray(values, dtype=np.float64)


def _base_row(job_id: str, model: str, seed: int, horizon: int) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "job_model": model,
        "seed": int(seed),
        "horizon": int(horizon),
    }


def run_single_job(
    *,
    model: str,
    seed: int,
    horizon: int,
    data_path: Path,
    output_dir: Path,
    date_file: Path | None = None,
    epochs: int | None = None,
    mc_samples: int | None = None,
    recursive_steps: int | None = None,
    smoke: bool = False,
    smoke_points: int = 16,
    device_name: str = "auto",
    torch_threads: int | None = None,
    save_checkpoint: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train/evaluate one fixed reviewer experiment combination.

    The returned dictionary is also written as ``job_metadata_<job_id>.json``.
    ``smoke=True`` is intended for temporary directories; callers should still
    remove that directory in a ``finally`` block (``run_job_cli`` does this).
    """

    if model not in MODEL_SPECS:
        raise ValueError(f"Unknown model {model!r}; choose one of {sorted(MODEL_SPECS)}")
    if int(horizon) < 1:
        raise ValueError("horizon must be a positive integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    model = str(model)
    seed = int(seed)
    horizon = int(horizon)
    job_id = make_job_id(model, seed, horizon)
    data_path = Path(data_path).resolve()
    output_dir = Path(output_dir).resolve()
    result_directory = result_folder_name(model, seed, horizon)
    if output_dir.name != result_directory:
        raise ValueError(
            f"output directory must end with {result_directory!r}; got {output_dir.name!r}"
        )
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing and not overwrite:
            raise FileExistsError(
                f"Output directory is non-empty: {output_dir}. Use --overwrite to rerun."
            )
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"job_status_{job_id}.json"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_json(
        {
            "job_id": job_id,
            "result_directory": result_directory,
            "model": model,
            "seed": seed,
            "horizon": horizon,
            "status": "running",
            "started_at": started_at,
        },
        status_path,
    )

    thread_count = _configure_threads(torch_threads)
    started_clock = time.perf_counter()
    try:
        # Keep the full reviewer horizon set in the configuration even though
        # this task evaluates one horizon.  WindowStore uses the maximum
        # configured horizon to define the common historical scaler cutoff;
        # retaining all four horizons makes every split numerically
        # comparable to the monolithic reviewer experiment.
        cfg = ExperimentConfig(horizons=DEFAULT_HORIZONS, seeds=(seed,))
        if epochs is not None:
            cfg.epochs = max(1, int(epochs))
        if mc_samples is not None:
            cfg.mc_samples = max(1, int(mc_samples))
        if recursive_steps is not None:
            cfg.recursive_steps = max(0, int(recursive_steps))
        if smoke:
            cfg.epochs = 1
            cfg.mc_samples = min(cfg.mc_samples, 2)
            cfg.recursive_steps = 0

        coords, geology, deformation, sequence_columns, frame = load_export(data_path)
        point_ids = (
            frame["PS_ID"].to_numpy()
            if "PS_ID" in frame.columns
            else np.arange(len(frame))
        )
        dates = _load_dates(date_file, deformation.shape[1])
        if smoke and len(coords) > int(smoke_points):
            keep = np.linspace(0, len(coords) - 1, int(smoke_points)).astype(int)
            coords = coords[keep]
            geology = geology[keep]
            deformation = deformation[keep]
            point_ids = point_ids[keep]

        if device_name == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_name == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested but CUDA is unavailable")
            device = torch.device("cuda")
        elif device_name == "cpu":
            device = torch.device("cpu")
        else:
            raise ValueError("device must be auto, cpu, or cuda")

        set_seed(seed)
        store = WindowStore(coords, geology, deformation, cfg, seed)
        train_starts, val_starts, test_starts = store.starts(horizon)
        train = store.make(store.train_nodes, train_starts, horizon)
        val = store.make(store.val_nodes, val_starts, horizon)
        test = store.make(store.test_nodes, test_starts, horizon)
        tx, ty, ids, starts = test
        true = store.y_scaler.inverse_transform(ty.numpy().reshape(-1, 1)).reshape(ty.shape)

        # Baselines are evaluated on exactly the same windows.  They are cheap
        # and are stored beside the selected model so paired tests remain local
        # to each independent scheduler task.
        metric_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        baseline_errors: dict[str, np.ndarray] = {}
        for baseline_name in ("persistence", "linear_trend"):
            prediction = baseline(
                tx.numpy(),
                "linear" if baseline_name == "linear_trend" else baseline_name,
                horizon,
            )
            prediction = store.y_scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(prediction.shape)
            row = _base_row(job_id, model, seed, horizon)
            row.update({"model": baseline_name, **metrics(prediction, true)})
            metric_rows.append(row)
            baseline_errors[baseline_name] = np.mean(np.abs(prediction - true), axis=1)
            for j, (point_index, target_start) in enumerate(zip(ids, starts)):
                for step in range(horizon):
                    item = _base_row(job_id, model, seed, horizon)
                    item.update(
                        {
                            "model": baseline_name,
                            "point_index": int(point_index),
                            "ps_id": _jsonable(point_ids[int(point_index)]),
                            "target_start": int(target_start),
                            "step": step + 1,
                            "prediction_mm": float(prediction[j, step]),
                            "target_mm": float(true[j, step]),
                        }
                    )
                    prediction_rows.append(item)

        cls, input_dim, use_coords, use_geo = MODEL_SPECS[model]
        train_variant = _select_variant(train, use_coords, use_geo)
        val_variant = _select_variant(val, use_coords, use_geo)
        test_variant = _select_variant(test, use_coords, use_geo)
        test_x = test_variant[0]
        network = fit_model(cls(input_dim, horizon, cfg).to(device), train_variant, val_variant, cfg, device)
        parameter_count = int(sum(parameter.numel() for parameter in network.parameters()))
        mean, lower, upper = mc_predict(network, test_x, cfg.mc_samples, device)
        mean = store.y_scaler.inverse_transform(mean.reshape(-1, 1)).reshape(mean.shape)
        lower = store.y_scaler.inverse_transform(lower.reshape(-1, 1)).reshape(lower.shape)
        upper = store.y_scaler.inverse_transform(upper.reshape(-1, 1)).reshape(upper.shape)

        metric_row = _base_row(job_id, model, seed, horizon)
        metric_row.update({"model": model, **metrics(mean, true)})
        last_observed = store.y_scaler.inverse_transform(tx.numpy()[:, -1, 0, None]).ravel()
        denominators = _rate_denominators(starts, horizon, dates)
        rate_true = (true[:, -1] - last_observed) / denominators
        rate_pred = (mean[:, -1] - last_observed) / denominators
        metric_row["rate_MAE_mm_per_step"] = float(np.mean(np.abs(rate_pred - rate_true)))
        if dates is not None:
            metric_row["rate_MAE_mm_per_day"] = metric_row["rate_MAE_mm_per_step"] / float(
                np.median(np.diff(dates.astype("int64").to_numpy()) / 86400000000000.0)
            )
        metric_rows.append(metric_row)

        model_errors = np.mean(np.abs(mean - true), axis=1)
        uncertainty_row = _base_row(job_id, model, seed, horizon)
        uncertainty_row.update(
            {
                "model": model,
                "mc_samples": int(cfg.mc_samples),
                "coverage_90": float(np.mean((true >= lower) & (true <= upper))),
                "mean_interval_width_mm": float(np.mean(upper - lower)),
            }
        )

        for j, (point_index, target_start) in enumerate(zip(ids, starts)):
            for step in range(horizon):
                item = _base_row(job_id, model, seed, horizon)
                item.update(
                    {
                        "model": model,
                        "point_index": int(point_index),
                        "ps_id": _jsonable(point_ids[int(point_index)]),
                        "target_start": int(target_start),
                        "step": step + 1,
                        "prediction_mm": float(mean[j, step]),
                        "target_mm": float(true[j, step]),
                        "lower90_mm": float(lower[j, step]),
                        "upper90_mm": float(upper[j, step]),
                    }
                )
                prediction_rows.append(item)

        hazard_rows: list[dict[str, Any]] = []
        predicted_rate = (mean[:, -1] - last_observed) / denominators
        level_percentile = np.argsort(np.argsort(mean[:, -1])) / max(len(mean) - 1, 1) * 100.0
        rate_percentile = np.argsort(np.argsort(predicted_rate)) / max(len(predicted_rate) - 1, 1) * 100.0
        for point_index, predicted_value, rate, level_rank, rate_rank in zip(
            ids, mean[:, -1], predicted_rate, level_percentile, rate_percentile
        ):
            combined = max(float(level_rank), float(rate_rank))
            category = (
                "low"
                if combined < 50
                else "moderate"
                if combined < 75
                else "high"
                if combined < 90
                else "very_high"
            )
            item = _base_row(job_id, model, seed, horizon)
            item.update(
                {
                    "model": model,
                    "point_index": int(point_index),
                    "ps_id": _jsonable(point_ids[int(point_index)]),
                    "predicted_deformation_mm": float(predicted_value),
                    "predicted_rate": float(rate),
                    "rate_unit": "mm/day" if dates is not None else "mm/acquisition_step",
                    "relative_deformation_percentile": float(level_rank),
                    "relative_rate_percentile": float(rate_rank),
                    "relative_hazard_percentile": combined,
                    "relative_hazard_class": category,
                }
            )
            hazard_rows.append(item)

        recursive_rows: list[dict[str, Any]] = []
        # Match the complete runner: recursive scenario forecasts are produced
        # only for the main STG-Informer at the longest reviewer horizon.
        if model == "STG-Informer" and horizon == max(DEFAULT_HORIZONS) and cfg.recursive_steps > horizon:
            recursive_mean, recursive_lower, recursive_upper = recursive_mc_predict(
                network,
                test_x,
                cfg.recursive_steps,
                horizon,
                max(5, cfg.mc_samples // 2),
                device,
            )
            recursive_mean = store.y_scaler.inverse_transform(recursive_mean.reshape(-1, 1)).reshape(recursive_mean.shape)
            recursive_lower = store.y_scaler.inverse_transform(recursive_lower.reshape(-1, 1)).reshape(recursive_lower.shape)
            recursive_upper = store.y_scaler.inverse_transform(recursive_upper.reshape(-1, 1)).reshape(recursive_upper.shape)
            for j, (point_index, target_start) in enumerate(zip(ids, starts)):
                for step in range(cfg.recursive_steps):
                    item = _base_row(job_id, model, seed, horizon)
                    item.update(
                        {
                            "model": model,
                            "point_index": int(point_index),
                            "ps_id": _jsonable(point_ids[int(point_index)]),
                            "origin_target_start": int(target_start),
                            "step": step + 1,
                            "prediction_mm": float(recursive_mean[j, step]),
                            "lower90_mm": float(recursive_lower[j, step]),
                            "upper90_mm": float(recursive_upper[j, step]),
                        }
                    )
                    recursive_rows.append(item)

        wilcoxon_rows: list[dict[str, Any]] = []
        try:
            from scipy.stats import wilcoxon

            for baseline_name, baseline_error in baseline_errors.items():
                try:
                    p_value = float(wilcoxon(model_errors, baseline_error, alternative="less").pvalue)
                except Exception:
                    p_value = float("nan")
                item = _base_row(job_id, model, seed, horizon)
                item.update(
                    {
                        "comparison": f"{model}_vs_{baseline_name}",
                        "p_value": p_value,
                        "n_pairs": int(len(model_errors)),
                        "alternative": "selected model absolute error < baseline absolute error",
                    }
                )
                wilcoxon_rows.append(item)
        except ImportError:
            pass

        prefix = job_id
        common_columns = ["job_id", "job_model", "seed", "horizon"]
        _write_csv(
            output_dir / f"metrics_{prefix}.csv",
            pd.DataFrame(metric_rows),
            common_columns
            + [
                "model",
                "MAE_mm",
                "RMSE_mm",
                "R2",
                "MAPE_percent",
                "WAPE_percent",
                "rate_MAE_mm_per_step",
                "rate_MAE_mm_per_day",
            ],
        )
        _write_csv(
            output_dir / f"predictions_{prefix}.csv",
            pd.DataFrame(prediction_rows),
            common_columns
            + [
                "model",
                "point_index",
                "ps_id",
                "target_start",
                "step",
                "prediction_mm",
                "target_mm",
                "lower90_mm",
                "upper90_mm",
            ],
        )
        _write_csv(
            output_dir / f"uncertainty_{prefix}.csv",
            pd.DataFrame([uncertainty_row]),
            common_columns + ["model", "mc_samples", "coverage_90", "mean_interval_width_mm"],
        )
        _write_csv(
            output_dir / f"relative_hazard_{prefix}.csv",
            pd.DataFrame(hazard_rows),
            common_columns
            + [
                "model",
                "point_index",
                "ps_id",
                "predicted_deformation_mm",
                "predicted_rate",
                "rate_unit",
                "relative_deformation_percentile",
                "relative_rate_percentile",
                "relative_hazard_percentile",
                "relative_hazard_class",
            ],
        )
        _write_csv(
            output_dir / f"recursive_forecasts_{prefix}.csv",
            pd.DataFrame(recursive_rows),
            common_columns
            + [
                "model",
                "point_index",
                "ps_id",
                "origin_target_start",
                "step",
                "prediction_mm",
                "lower90_mm",
                "upper90_mm",
            ],
        )
        _write_csv(
            output_dir / f"wilcoxon_{prefix}.csv",
            pd.DataFrame(wilcoxon_rows),
            common_columns + ["comparison", "p_value", "n_pairs", "alternative"],
        )
        parameter_rows = []
        for baseline_name in ("persistence", "linear_trend"):
            parameter_rows.append(
                {
                    **_base_row(job_id, model, seed, horizon),
                    "model": baseline_name,
                    "input_dim": 4,
                    "hidden": 0,
                    "layers": 0,
                    "heads": 0,
                    "parameters": 0,
                }
            )
        parameter_rows.append(
            {
                **_base_row(job_id, model, seed, horizon),
                "model": model,
                "input_dim": int(input_dim),
                "hidden": int(cfg.hidden),
                "layers": int(cfg.layers),
                "heads": int(cfg.heads),
                "parameters": parameter_count,
            }
        )
        _write_csv(
            output_dir / f"model_parameter_count_{prefix}.csv",
            pd.DataFrame(parameter_rows),
            common_columns + ["model", "input_dim", "hidden", "layers", "heads", "parameters"],
        )

        if save_checkpoint:
            torch.save(network.state_dict(), output_dir / f"model_checkpoint_{prefix}.pt")

        elapsed = time.perf_counter() - started_clock
        metadata = {
            **_base_row(job_id, model, seed, horizon),
            "status": "completed",
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": float(elapsed),
            "data_path": str(data_path),
            "data_sha256": _sha256(data_path),
            "date_file": None if date_file is None else str(Path(date_file).resolve()),
            "sequence_columns": sequence_columns,
            "n_points": int(len(coords)),
            "n_acquisitions": int(deformation.shape[1]),
            "n_train_nodes": int(len(store.train_nodes)),
            "n_validation_nodes": int(len(store.val_nodes)),
            "n_test_nodes": int(len(store.test_nodes)),
            "test_ps_ids": [_jsonable(point_ids[i]) for i in store.test_nodes],
            "test_node_indices": store.test_nodes.tolist(),
            "train_target_starts": train_starts.tolist(),
            "validation_target_starts": val_starts.tolist(),
            "test_target_starts": test_starts.tolist(),
            "time_axis": "calendar dates" if dates is not None else "acquisition index",
            "device": str(device),
            "torch_threads": thread_count,
            "config": asdict(cfg),
            "mc_samples": int(cfg.mc_samples),
            "recursive_steps": int(cfg.recursive_steps),
            "smoke": bool(smoke),
            "checkpoint": f"model_checkpoint_{prefix}.pt" if save_checkpoint else None,
            "result_directory": result_directory,
        }
        _write_json(metadata, output_dir / f"job_metadata_{prefix}.json")
        _write_json(metadata, status_path)
        return metadata
    except Exception as exc:
        failure = {
            "job_id": job_id,
            "model": model,
            "seed": seed,
            "horizon": horizon,
            "status": "failed",
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(failure, status_path)
        raise


def run_job_cli(
    *,
    model: str,
    seed: int,
    horizon: int,
    argv: Sequence[str] | None = None,
) -> int:
    """CLI adapter used by every generated wrapper."""

    job_id = make_job_id(model, seed, horizon)
    result_directory = result_folder_name(model, seed, horizon)
    parser = argparse.ArgumentParser(description=f"Run reviewer job {job_id}")
    parser.add_argument("--data", type=Path, default=HERE / "data" / "insarthick.csv")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HERE,
        help="parent directory for the flat per-task result folders (default: 200EX)",
    )
    parser.add_argument("--date-file", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--mc-samples", type=int, default=None)
    parser.add_argument("--recursive-steps", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="temporary one-epoch validation; output is deleted")
    parser.add_argument("--smoke-points", type=int, default=16)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke and args.smoke_points < 4:
        parser.error("--smoke-points must be at least 4")

    if args.smoke:
        # Do not place smoke files under output-root.  The temporary directory
        # is removed regardless of success or failure.
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{job_id}_smoke_"))
        smoke_output = temp_dir / result_directory
        try:
            run_single_job(
                model=model,
                seed=seed,
                horizon=horizon,
                data_path=args.data,
                output_dir=smoke_output,
                date_file=args.date_file,
                epochs=1,
                mc_samples=min(args.mc_samples or 2, 2),
                recursive_steps=0,
                smoke=True,
                smoke_points=args.smoke_points,
                device_name=args.device,
                torch_threads=args.torch_threads,
                save_checkpoint=False,
                overwrite=True,
            )
            print(f"SMOKE PASS: {job_id}")
            return 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    output_root = Path(args.output_root).resolve()
    output_dir = output_root / result_directory
    metadata = run_single_job(
        model=model,
        seed=seed,
        horizon=horizon,
        data_path=args.data,
        output_dir=output_dir,
        date_file=args.date_file,
        epochs=args.epochs,
        mc_samples=args.mc_samples,
        recursive_steps=args.recursive_steps,
        smoke=False,
        smoke_points=args.smoke_points,
        device_name=args.device,
        torch_threads=args.torch_threads,
        save_checkpoint=not args.no_checkpoint,
        overwrite=args.overwrite,
    )
    print(
        f"COMPLETED: {metadata['job_id']} | device={metadata['device']} | "
        f"elapsed_seconds={metadata['elapsed_seconds']:.2f} | output={output_dir}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one arbitrary 200EX reviewer combination")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS))
    parser.add_argument("--seed", type=int, choices=DEFAULT_SEEDS)
    parser.add_argument("--horizon", type=int, choices=DEFAULT_HORIZONS)
    args, remainder = parser.parse_known_args(argv)
    missing = [name for name in ("model", "seed", "horizon") if getattr(args, name) is None]
    if missing:
        parser.error("--model, --seed and --horizon are required")
    return run_job_cli(model=args.model, seed=args.seed, horizon=args.horizon, argv=remainder)


if __name__ == "__main__":
    raise SystemExit(main())
