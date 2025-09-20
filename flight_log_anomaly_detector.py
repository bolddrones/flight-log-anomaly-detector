#!/usr/bin/env python3
import argparse, json, math
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.seasonal import STL
    _HAS_STL = True
except Exception:
    _HAS_STL = False

def hampel_indices(y: np.ndarray, k: int = 7, n_sigma: float = 3.0) -> np.ndarray:
    n = len(y)
    flagged = np.zeros(n, dtype=bool)
    y = np.asarray(y, dtype=float)
    for i in range(n):
        a = max(0, i - k)
        b = min(n, i + k + 1)
        window = y[a:b]
        med = np.nanmedian(window)
        mad = np.nanmedian(np.abs(window - med))
        scale = 1.4826 * (mad if mad > 1e-12 else 1e-12)
        if abs(y[i] - med) > n_sigma * scale:
            flagged[i] = True
    return np.where(flagged)[0]

def stl_residual(y: np.ndarray, period: int) -> Optional[np.ndarray]:
    if not _HAS_STL or period is None or period <= 1:
        return None
    try:
        y2 = np.asarray(y, dtype=float)
        if np.isnan(y2).any():
            med = np.nanmedian(y2)
            y2 = np.where(np.isnan(y2), med, y2)
        res = STL(y2, period=period, robust=True).fit()
        return y2 - res.trend - res.seasonal
    except Exception:
        return None

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    ph1 = math.radians(lat1); ph2 = math.radians(lat2)
    dph = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dph/2)**2 + math.cos(ph1)*math.cos(ph2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(max(0.0, min(1.0, a))))

def detect_anomalies(df: pd.DataFrame,
                     time_col: str,
                     cols: List[str],
                     k: int = 7,
                     n_sigma: float = 3.0,
                     stl_cols: Optional[List[str]] = None,
                     stl_period: Optional[int] = None,
                     gps_jump: bool = True,
                     gps_alpha: float = 2.0,
                     gps_margin_m: float = 30.0) -> List[Dict[str, Any]]:
    anomalies = []
    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}'")
    t = pd.to_datetime(df[time_col], errors='coerce', utc=True)
    if t.isna().any():
        raise ValueError("Time column has unparseable values.")
    stl_cols = stl_cols or []
    for col in cols:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors='coerce').values.astype(float)
        x_work = x.copy()
        residual = None
        if col in stl_cols and stl_period and stl_period > 1:
            residual = stl_residual(x, stl_period)
            if residual is not None:
                x_work = residual
        idxs = hampel_indices(x_work, k=k, n_sigma=n_sigma)
        for i in idxs:
            item = {
                "index": int(i),
                "timestamp": t.iloc[i].isoformat(),
                "column": col,
                "value": None if np.isnan(x[i]) else float(x[i]),
                "reason": "hampel"
            }
            if residual is not None:
                item["residual"] = float(residual[i])
            anomalies.append(item)
    if gps_jump and all(c in df.columns for c in ["lat","lon","gps_speed_mps"]):
        lat = pd.to_numeric(df["lat"], errors='coerce').values
        lon = pd.to_numeric(df["lon"], errors='coerce').values
        spd = pd.to_numeric(df["gps_speed_mps"], errors='coerce').values
        dt = np.diff(t.view("int64") // 10**9, prepend=np.nan)
        for i in range(1, len(df)):
            if any(np.isnan([lat[i-1], lon[i-1], lat[i], lon[i], spd[i], dt[i]])):
                continue
            dist = haversine_m(lat[i-1], lon[i-1], lat[i], lon[i])
            allowed = gps_alpha * float(spd[i]) * float(dt[i]) + gps_margin_m
            if dist > allowed:
                anomalies.append({
                    "index": int(i),
                    "timestamp": t.iloc[i].isoformat(),
                    "column": "gps",
                    "value": float(dist),
                    "reason": "gps_jump",
                    "distance_m": float(dist),
                    "allowed_m": float(allowed)
                })
    return sorted(anomalies, key=lambda d: d["index"])

def main():
    p = argparse.ArgumentParser(description="Flight Log Anomaly Detector (Hampel + optional STL)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pdect = sub.add_parser("detect", help="Detect anomalies from a telemetry CSV")
    pdect.add_argument("--csv", required=True)
    pdect.add_argument("--out", required=True)
    pdect.add_argument("--time-col", required=True)
    pdect.add_argument("--cols", required=True, help="Comma-separated numeric columns for Hampel")
    pdect.add_argument("--k", type=int, default=7)
    pdect.add_argument("--n-sigma", type=float, default=3.0)
    pdect.add_argument("--stl-cols", default="", help="Comma-separated columns for STL pre-detrending")
    pdect.add_argument("--stl-period", type=int, default=0)
    pdect.add_argument("--gps-jump", type=int, default=1)
    pdect.add_argument("--gps-alpha", type=float, default=2.0)
    pdect.add_argument("--gps-margin-m", type=float, default=30.0)
    def run_detect(args):
        df = pd.read_csv(args.csv)
        cols = [c.strip() for c in args.cols.split(",") if c.strip()]
        stl_cols = [c.strip() for c in args.stl_cols.split(",") if c.strip()]
        anomalies = detect_anomalies(
            df=df,
            time_col=args.time_col,
            cols=cols,
            k=args.k,
            n_sigma=args.n_sigma,
            stl_cols=stl_cols,
            stl_period=(args.stl_period if args.stl_period>1 else None),
            gps_jump=bool(args.gps_jump),
            gps_alpha=args.gps_alpha,
            gps_margin_m=args.gps_margin_m
        )
        with open(args.out, "w") as f:
            json.dump(anomalies, f, indent=2)
        print(f"Wrote {len(anomalies)} anomalies to {args.out}")
        by_col = {}
        for a in anomalies:
            by_col[a["column"]] = by_col.get(a["column"], 0) + 1
        if by_col:
            print("Counts by column:", by_col)
        else:
            print("No anomalies flagged.")
    pdect.set_defaults(func=run_detect)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
