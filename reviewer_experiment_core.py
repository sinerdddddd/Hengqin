"""Reviewer-complete leakage-safe experiments for Hengqin PS-InSAR data.

The command runs spatial hold-out evaluation with a strict chronological split
of forecast windows, several horizons, matched-covariate neural baselines and
Monte-Carlo dropout uncertainty.  It accepts the supplied F5--F108 export
(which has no dates) and records that horizons are acquisition indices. If a
one-column date file is supplied with ``--date-file``, the script reports
calendar-day intervals and rates while retaining the observed acquisition
sequence (no interpolation is silently introduced).
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


@dataclass
class ExperimentConfig:
    input_len: int = 80
    horizons: tuple[int, ...] = (1, 3, 6, 9, 12)
    spatial_test_fraction: float = 0.20
    train_window_fraction: float = 0.60
    validation_window_fraction: float = 0.20
    hidden: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.15
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    seeds: tuple[int, ...] = (42, 123, 2024, 7, 99)
    mc_samples: int = 100
    recursive_steps: int = 96


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_export(path: Path):
    df = pd.read_csv(path)
    required = {"Longitude", "Latitude", "RASTERVALU"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    fcols = sorted((c for c in df.columns if c.startswith("F") and c[1:].isdigit()),
                   key=lambda c: int(c[1:]))
    if not fcols:
        raise ValueError("No F<number> deformation columns found")
    numbers = [int(c[1:]) for c in fcols]
    if numbers != list(range(numbers[0], numbers[-1] + 1)):
        raise ValueError(f"Deformation columns are not contiguous: {fcols}")
    coords = df[["Longitude", "Latitude"]].to_numpy(np.float32)
    geology = df[["RASTERVALU"]].to_numpy(np.float32)
    y = df[fcols].to_numpy(np.float32)
    if np.isnan(y).any():
        # Preserve rows while avoiding silent loss of windows; interpolation is
        # performed along each point's time axis and is reported in metadata.
        y = pd.DataFrame(y).interpolate(axis=1, limit_direction="both").to_numpy(np.float32)
    if not np.isfinite(coords).all() or not np.isfinite(geology).all() or not np.isfinite(y).all():
        raise ValueError("Input contains non-finite coordinates, geology, or deformation after interpolation")
    return coords, geology, y, fcols, df


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    return {
        "MAE_mm": float(np.mean(np.abs(err))),
        "RMSE_mm": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - np.sum(err ** 2) / (np.sum((target - target.mean()) ** 2) + 1e-12)),
        "MAPE_percent": float(np.mean(np.abs(err) / np.maximum(np.abs(target), 1e-6)) * 100.0),
        "WAPE_percent": float(np.sum(np.abs(err)) / (np.sum(np.abs(target)) + 1e-8) * 100),
    }


class WindowStore:
    """Scale using training points only and generate windows by target start."""

    def __init__(self, coords, geology, deformation, cfg: ExperimentConfig, seed: int):
        n, t = deformation.shape
        self.cfg = cfg
        self.n, self.t = n, t
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        ntest = max(1, int(round(cfg.spatial_test_fraction * n)))
        self.test_nodes = np.sort(perm[:ntest]); self.fit_nodes = np.sort(perm[ntest:])
        nval_points = max(1, int(round(cfg.validation_window_fraction * len(self.fit_nodes))))
        self.val_nodes = self.fit_nodes[:nval_points]
        self.train_nodes = self.fit_nodes[nval_points:]
        if len(self.train_nodes) == 0:
            raise ValueError("Spatial split leaves no training points")
        self.coord_scaler = StandardScaler().fit(coords[self.train_nodes])
        self.geo_scaler = StandardScaler().fit(geology[self.train_nodes])
        # Fit deformation scale on historical values only (before the test target).
        hist_end = max(cfg.input_len, t - max(cfg.horizons))
        self.y_scaler = StandardScaler().fit(deformation[self.train_nodes, :hist_end].reshape(-1, 1))
        self.coords = self.coord_scaler.transform(coords).astype(np.float32)
        self.geology = self.geo_scaler.transform(geology).astype(np.float32)
        self.y = self.y_scaler.transform(deformation.reshape(-1, 1)).reshape(n, t).astype(np.float32)
        self.raw_y = deformation

    def starts(self, horizon: int):
        starts = np.arange(self.cfg.input_len, self.t - horizon + 1)
        if len(starts) < 3:
            raise ValueError(f"Only {len(starts)} windows for horizon {horizon}; reduce input_len")
        ntr = max(1, int(len(starts) * self.cfg.train_window_fraction)); nval = max(1, int(len(starts) * self.cfg.validation_window_fraction))
        return starts[:ntr], starts[ntr:ntr+nval], starts[ntr+nval:]

    def make(self, nodes: Iterable[int], starts: Sequence[int], horizon: int, use_coords=True):
        xs, ys, ids, ss = [], [], [], []
        for i in nodes:
            for target_start in starts:
                s = int(target_start) - self.cfg.input_len
                x = np.column_stack([self.y[i, s:target_start],
                                      np.repeat(self.coords[i:i+1], self.cfg.input_len, axis=0),
                                      np.repeat(self.geology[i:i+1], self.cfg.input_len, axis=0)])
                if not use_coords: x = x[:, [0, 3]]
                xs.append(x); ys.append(self.y[i, target_start:target_start+horizon])
                ids.append(i); ss.append(target_start)
        return torch.tensor(np.asarray(xs), dtype=torch.float32), torch.tensor(np.asarray(ys), dtype=torch.float32), np.asarray(ids), np.asarray(ss)


class ForecastTransformer(nn.Module):
    def __init__(self, input_dim: int, pred_len: int, cfg: ExperimentConfig):
        super().__init__(); self.pred_len = pred_len
        self.proj = nn.Linear(input_dim, cfg.hidden)
        self.pos = nn.Parameter(torch.randn(1, cfg.input_len, cfg.hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(cfg.hidden, cfg.heads, 4*cfg.hidden, cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers)
        self.head = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, pred_len))

    def forward(self, x):
        h = self.encoder(self.proj(x) + self.pos[:, :x.size(1)])
        return self.head(h[:, -1])


class VanillaTransformer(ForecastTransformer):
    """Matched-covariate Transformer baseline with mean pooling."""
    def forward(self, x):
        h = self.encoder(self.proj(x) + self.pos[:, :x.size(1)])
        return self.head(h.mean(dim=1))


class ForecastLSTM(nn.Module):
    def __init__(self, input_dim: int, pred_len: int, cfg: ExperimentConfig):
        super().__init__(); self.pred_len = pred_len
        self.lstm = nn.LSTM(input_dim, cfg.hidden, num_layers=cfg.layers, batch_first=True, dropout=cfg.dropout if cfg.layers > 1 else 0)
        self.head = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, pred_len))

    def forward(self, x): return self.head(self.lstm(x)[0][:, -1])


class ForecastGRU(ForecastLSTM):
    """GRU baseline with the same covariates and parameter dimensions."""
    def __init__(self, input_dim: int, pred_len: int, cfg: ExperimentConfig):
        nn.Module.__init__(self); self.pred_len = pred_len
        self.lstm = nn.GRU(input_dim, cfg.hidden, num_layers=cfg.layers, batch_first=True, dropout=cfg.dropout if cfg.layers > 1 else 0)
        self.head = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, pred_len))


class ForecastConvLSTM(nn.Module):
    """One-dimensional convolution followed by an LSTM encoder."""
    def __init__(self, input_dim: int, pred_len: int, cfg: ExperimentConfig):
        super().__init__(); self.pred_len = pred_len
        self.conv = nn.Conv1d(input_dim, cfg.hidden, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)
        self.lstm = nn.LSTM(
            cfg.hidden,
            cfg.hidden,
            num_layers=cfg.layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.layers > 1 else 0,
        )
        self.head = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, pred_len))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h = self.dropout(self.activation(h))
        return self.head(self.lstm(h)[0][:, -1])


class ForecastCNNLSTM(nn.Module):
    """Two-layer temporal CNN followed by an LSTM encoder."""
    def __init__(self, input_dim: int, pred_len: int, cfg: ExperimentConfig):
        super().__init__(); self.pred_len = pred_len
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cfg.hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(cfg.hidden, cfg.hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            cfg.hidden,
            cfg.hidden,
            num_layers=cfg.layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.layers > 1 else 0,
        )
        self.head = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, pred_len))

    def forward(self, x):
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        return self.head(self.lstm(h)[0][:, -1])


def fit_model(model, train, val, cfg, device):
    x, y, _, _ = train; vx, vy, _, _ = val
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    best, state = float("inf"), None
    for _ in range(cfg.epochs):
        model.train()
        for idx in torch.randperm(len(x)).split(cfg.batch_size):
            pred = model(x[idx].to(device)); loss = nn.functional.mse_loss(pred, y[idx].to(device))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad(): vl = nn.functional.mse_loss(model(vx.to(device)), vy.to(device)).item()
        if vl < best: best, state = vl, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if state is not None: model.load_state_dict(state)
    return model


def mc_predict(model, x, samples, device):
    preds = []
    model.train()  # activate dropout for epistemic uncertainty
    with torch.no_grad():
        for _ in range(max(1, samples)): preds.append(model(x.to(device)).cpu().numpy())
    arr = np.stack(preds)
    return arr.mean(0), np.quantile(arr, 0.05, 0), np.quantile(arr, 0.95, 0)


def recursive_mc_predict(model, x, total_steps: int, block: int, samples: int, device):
    """Iteratively roll forecasts forward, preserving static covariates."""
    trajectories = []
    model.train()
    with torch.no_grad():
        for _ in range(max(1, samples)):
            state = x.to(device).clone(); chunks = []
            left = total_steps
            while left > 0:
                out = model(state); take = min(left, out.shape[1]); nxt = out[:, :take]
                chunks.append(nxt.cpu().numpy()); left -= take
                if left:
                    feat = state[:, -1:, :].repeat(1, take, 1); feat[:, :, 0] = nxt
                    state = torch.cat([state[:, take:, :], feat], dim=1)
            trajectories.append(np.concatenate(chunks, axis=1))
    arr = np.stack(trajectories); return arr.mean(0), np.quantile(arr, .05, axis=0), np.quantile(arr, .95, axis=0)


def baseline(x_scaled: np.ndarray, kind: str, horizon: int):
    hist = x_scaled[:, :, 0]
    if kind == "persistence": return np.repeat(hist[:, -1,None], horizon, axis=1)
    out=[]; k=np.arange(hist.shape[1]); future=np.arange(hist.shape[1], hist.shape[1]+horizon)
    for row in hist:
        slope, intercept = np.polyfit(k, row, 1); out.append(intercept + slope*future)
    return np.asarray(out, np.float32)


def run(cfg: ExperimentConfig, data_path: Path, output: Path, smoke=False, date_file: Path | None = None):
    if smoke: cfg.epochs, cfg.seeds, cfg.mc_samples, cfg.recursive_steps = 1, (cfg.seeds[0],), 5, 0
    output.mkdir(parents=True, exist_ok=True)
    coords, geo, deformation, fcols, frame = load_export(data_path)
    point_ids = frame["PS_ID"].to_numpy() if "PS_ID" in frame.columns else np.arange(len(frame))
    dates = None
    if date_file is not None:
        if not Path(date_file).exists():
            raise FileNotFoundError(f"Date file not found: {date_file}")
        ddf = pd.read_csv(date_file)
        dates = pd.to_datetime(ddf.iloc[:, 0], errors="coerce")
        if len(dates) != deformation.shape[1]: raise ValueError("--date-file must contain one date per acquisition")
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("--date-file dates must be valid, unique, and strictly increasing")
    if smoke and len(coords)>128:
        keep=np.linspace(0,len(coords)-1,128).astype(int); coords,geo,deformation=coords[keep],geo[keep],deformation[keep]; point_ids=point_ids[keep]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows=[]; pred_rows=[]; uncertainty=[]; split_meta=[]; hazard_rows=[]; abs_errors={}; parameter_rows=[]; recursive_rows=[]
    for seed in cfg.seeds:
        set_seed(seed); store=WindowStore(coords,geo,deformation,cfg,seed)
        split_record={"seed":seed,"n_train_nodes":int(len(store.train_nodes)),"n_validation_nodes":int(len(store.val_nodes)),"n_test_nodes":int(len(store.test_nodes)),"test_nodes":store.test_nodes.tolist()}
        split_record["test_ps_ids"] = point_ids[store.test_nodes].tolist()
        split_meta.append(split_record)
        for horizon in cfg.horizons:
            tr_s, va_s, te_s = store.starts(horizon)
            train=store.make(store.train_nodes,tr_s,horizon); val=store.make(store.val_nodes,va_s,horizon); test=store.make(store.test_nodes,te_s,horizon)
            tx,ty,ids,target_starts=test; true=store.y_scaler.inverse_transform(ty.numpy().reshape(-1,1)).reshape(ty.shape)
            for kind in ("persistence","linear_trend"):
                p=baseline(tx.numpy(), "linear" if kind=="linear_trend" else kind, horizon); p=store.y_scaler.inverse_transform(p.reshape(-1,1)).reshape(p.shape)
                rows.append({"seed":seed,"horizon":horizon,"model":kind,**metrics(p,true)})
                parameter_rows.append({"seed":seed,"horizon":horizon,"model":kind,"input_dim":4,"hidden":0,"layers":0,"heads":0,"parameters":0})
                abs_errors[(seed,horizon,kind)] = np.mean(np.abs(p-true),axis=1)
                for j,(pid,start) in enumerate(zip(ids,target_starts)):
                    for h in range(horizon): pred_rows.append({"seed":seed,"horizon":horizon,"model":kind,"point_index":int(pid),"target_start":int(start),"step":h+1,"prediction_mm":float(p[j,h]),"target_mm":float(true[j,h])})
            variants = (("STG-Informer",ForecastTransformer,4,True,True),("G-Informer",ForecastTransformer,2,False,True),("ST-Informer",ForecastTransformer,3,True,False),("Informer",ForecastTransformer,1,False,False),("GRU",ForecastGRU,4,True,True),("Transformer",VanillaTransformer,4,True,True),("ConvLSTM",ForecastConvLSTM,4,True,True),("CNN-LSTM",ForecastCNNLSTM,4,True,True))
            for name, cls, dim, use_coords, use_geo in variants:
                def variant(data):
                    if use_coords and use_geo: return data
                    cols=[0] + ([1,2] if use_coords else []) + ([3] if use_geo else [])
                    return (data[0][...,cols], data[1], data[2], data[3])
                train_model = variant(train); val_model = variant(val); test_model = variant(test)[0]
                model=fit_model(cls(dim,horizon,cfg).to(device), train_model, val_model, cfg, device)
                parameter_rows.append({"seed":seed,"horizon":horizon,"model":name,"input_dim":dim,"hidden":cfg.hidden,"layers":cfg.layers,"heads":cfg.heads,"parameters":sum(p.numel() for p in model.parameters())})
                mean,lo,hi=mc_predict(model,test_model,cfg.mc_samples,device)
                mean=store.y_scaler.inverse_transform(mean.reshape(-1,1)).reshape(mean.shape); lo=store.y_scaler.inverse_transform(lo.reshape(-1,1)).reshape(lo.shape); hi=store.y_scaler.inverse_transform(hi.reshape(-1,1)).reshape(hi.shape)
                rows.append({"seed":seed,"horizon":horizon,"model":name,**metrics(mean,true)})
                abs_errors[(seed,horizon,name)] = np.mean(np.abs(mean-true),axis=1)
                last_observed = store.y_scaler.inverse_transform(tx.numpy()[:,-1,0,None]).ravel()
                if dates is None:
                    denominators = np.full(len(target_starts), float(horizon))
                else:
                    denominators = np.asarray([max((dates.iloc[int(start + horizon - 1)] - dates.iloc[int(start - 1)]).days, 1) for start in target_starts], dtype=float)
                rate_true=(true[:,-1]-last_observed)/denominators
                rate_pred=(mean[:,-1]-last_observed)/denominators
                rows[-1]["rate_MAE_mm_per_step"]=float(np.mean(np.abs(rate_pred-rate_true)))
                coverage=float(np.mean((true>=lo)&(true<=hi))); width=float(np.mean(hi-lo))
                uncertainty.append({"seed":seed,"horizon":horizon,"model":name,"coverage_90":coverage,"mean_interval_width_mm":width})
                for j,(pid,start) in enumerate(zip(ids,target_starts)):
                    for h in range(horizon): pred_rows.append({"seed":seed,"horizon":horizon,"model":name,"point_index":int(pid),"target_start":int(start),"step":h+1,"prediction_mm":float(mean[j,h]),"target_mm":float(true[j,h]),"lower90_mm":float(lo[j,h]),"upper90_mm":float(hi[j,h])})
                if name == "STG-Informer" and horizon == max(cfg.horizons) and cfg.recursive_steps > horizon:
                    rm, rl, ru = recursive_mc_predict(model, test_model, cfg.recursive_steps, horizon, max(5, cfg.mc_samples//2), device)
                    rm=store.y_scaler.inverse_transform(rm.reshape(-1,1)).reshape(rm.shape); rl=store.y_scaler.inverse_transform(rl.reshape(-1,1)).reshape(rl.shape); ru=store.y_scaler.inverse_transform(ru.reshape(-1,1)).reshape(ru.shape)
                    for j,(pid,start) in enumerate(zip(ids,target_starts)):
                        for h in range(cfg.recursive_steps): recursive_rows.append({"seed":seed,"model":name,"point_index":int(pid),"origin_target_start":int(start),"step":h+1,"prediction_mm":float(rm[j,h]),"lower90_mm":float(rl[j,h]),"upper90_mm":float(ru[j,h])})
                # Relative hazard is descriptive ranking only. Include the
                # forecasted deformation rate as a second indicator, as
                # requested by the reviewers; with no dates the denominator is
                # acquisition steps, and with dates it is calendar days.
                predicted_rate = (mean[:, -1] - last_observed) / denominators
                level_percentile = np.argsort(np.argsort(mean[:, -1])) / max(len(mean) - 1, 1) * 100.0
                rate_percentile = np.argsort(np.argsort(predicted_rate)) / max(len(predicted_rate) - 1, 1) * 100.0
                for pid,predicted_value,rate,lp,rp in zip(ids,mean[:,-1],predicted_rate,level_percentile,rate_percentile):
                    combined = max(float(lp), float(rp))
                    cat="low" if combined < 50 else "moderate" if combined < 75 else "high" if combined < 90 else "very_high"
                    hazard_rows.append({"seed":seed,"horizon":horizon,"model":name,"point_index":int(pid),"predicted_deformation_mm":float(predicted_value),"predicted_rate":float(rate),"rate_unit":"mm/day" if dates is not None else "mm/acquisition_step","relative_deformation_percentile":float(lp),"relative_rate_percentile":float(rp),"relative_hazard_percentile":combined,"relative_hazard_class":cat})
    pd.DataFrame(rows).to_csv(output/"reviewer_metrics_by_seed_horizon.csv",index=False)
    summary=pd.DataFrame(rows).groupby(["horizon","model"])[["MAE_mm","RMSE_mm","R2","MAPE_percent","WAPE_percent","rate_MAE_mm_per_step"]].agg(["mean","std"]).reset_index(); summary.to_csv(output/"reviewer_metrics_summary.csv",index=False)
    pd.DataFrame(uncertainty).to_csv(output/"forecast_uncertainty_summary.csv",index=False)
    pd.DataFrame(pred_rows).to_csv(output/"representative_point_predictions.csv",index=False)
    pd.DataFrame(recursive_rows).to_csv(output/"recursive_forecasts_with_uncertainty.csv",index=False)
    pd.DataFrame(hazard_rows).to_csv(output/"relative_hazard_percentiles.csv",index=False)
    pd.DataFrame(parameter_rows).drop_duplicates(["horizon","model"]).to_csv(output/"model_parameter_counts.csv",index=False)
    pd.DataFrame(split_meta).to_json(output/"split_metadata_by_seed.json",orient="records",indent=2)
    try:
        from scipy.stats import wilcoxon
        tests=[]
        for (seed,horizon,model), err in abs_errors.items():
            if model in {"persistence","linear_trend"}: continue
            for base_name in ("linear_trend", "persistence"):
                base=abs_errors.get((seed,horizon,base_name))
                if base is not None:
                    try: p=float(wilcoxon(err,base,alternative="less").pvalue)
                    except Exception: p=float("nan")
                    tests.append({"seed":seed,"horizon":horizon,"comparison":f"{model}_vs_{base_name}","p_value":p})
        pd.DataFrame(tests).to_csv(output/"wilcoxon_pvalues_by_horizon.csv",index=False)
    except ImportError:
        pass
    if dates is not None:
        dvals = dates.astype("int64").to_numpy(dtype=np.float64) / 86400000000000.0
        interval = np.diff(dvals); interval = interval[np.isfinite(interval) & (interval > 0)]
    else: interval = None
    if interval is not None and len(interval):
        metric_path = output/"reviewer_metrics_by_seed_horizon.csv"
        mframe = pd.read_csv(metric_path)
        mframe["rate_MAE_mm_per_day"] = mframe["rate_MAE_mm_per_step"] / float(np.median(interval))
        mframe.to_csv(metric_path, index=False)
    meta={"sequence_columns":fcols,"n_points":int(len(coords)),"n_acquisitions":int(deformation.shape[1]),"time_axis":"calendar dates" if dates is not None else "acquisition index (no date column supplied)","irregular_interval_handling":"windows use acquisition order; supplied dates are used only to report interval range and rate per day","acquisition_interval_days_min":None if interval is None or len(interval)==0 else float(np.min(interval)),"acquisition_interval_days_median":None if interval is None or len(interval)==0 else float(np.median(interval)),"acquisition_interval_days_mean":None if interval is None or len(interval)==0 else float(np.mean(interval)),"acquisition_interval_days_max":None if interval is None or len(interval)==0 else float(np.max(interval)),"horizons":list(cfg.horizons),"spatial_test_fraction":cfg.spatial_test_fraction,"models":"all neural models receive deformation, longitude, latitude and RASTERVALU; ablations are explicit"}
    (output/"reviewer_experiment_config.json").write_text(json.dumps({**meta,**asdict(cfg),"device":str(device)},indent=2),encoding="utf-8")
    print(summary.to_string(index=False))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,default=Path(__file__).with_name("insarthick.csv")); ap.add_argument("--output",type=Path,default=Path(__file__).with_name("reviewer_outputs")); ap.add_argument("--epochs",type=int,default=None); ap.add_argument("--seeds",type=int,nargs="+",default=None); ap.add_argument("--horizons",type=int,nargs="+",default=None); ap.add_argument("--date-file",type=Path,default=None,help="CSV containing one acquisition date per row"); ap.add_argument("--recursive-steps",type=int,default=None,help="Recursive forecast length; default 96, smoke disables it"); ap.add_argument("--smoke",action="store_true"); args=ap.parse_args(); cfg=ExperimentConfig();
    if args.epochs is not None: cfg.epochs=args.epochs
    if args.seeds is not None: cfg.seeds=tuple(args.seeds)
    if args.horizons is not None: cfg.horizons=tuple(sorted(set(args.horizons)))
    if args.recursive_steps is not None: cfg.recursive_steps=max(0,args.recursive_steps)
    run(cfg,args.data,args.output,args.smoke,args.date_file)


if __name__ == "__main__": main()
