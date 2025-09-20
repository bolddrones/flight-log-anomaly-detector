# Flight Log Anomaly Detector (Hampel + optional STL)

Flags weird events in UAV telemetry: battery voltage spikes, GPS drift/jumps, IMU outliers.

## What it does

- **Hampel filter (rolling median + MAD)** on chosen numeric columns.
- Optional **STL decomposition** to remove trend/seasonality before Hampel (per-column).
- Optional **GPS jump** detector using lat/lon vs reported speed.
- Outputs an **anomalies JSON** list with timestamps, columns, values, and reasons.

## MVP model

- For each selected column, apply a **windowed Hampel filter**:
  - window size `k` (neighbors on each side, so total window = `2k+1`)
  - threshold `n_sigma` (default 3.0) on MAD-scaled deviation
- If you pass `--stl-cols`, we run STL (if available) and apply Hampel to the residuals.
- If `lat`, `lon`, and `gps_speed_mps` exist, we can also flag **GPS jumps** where
  the per-sample movement **exceeds** a plausible distance given speed.

> This is deliberately simple, fast, and explainable. Treat it as triage; you can review flagged rows.

## Quick start

1) CSV columns you can use (example below):
   - `timestamp` (ISO)
   - `batt_voltage_v`
   - `gps_speed_mps`
   - `lat`, `lon` (optional, for GPS jump detector)
   - `imu_ax`, `imu_ay`, `imu_az` (optional)

2) Detect anomalies and write JSON:

```bash
python flight_log_anomaly_detector.py detect   --csv example_log.csv   --time-col timestamp   --cols batt_voltage_v,gps_speed_mps,imu_ax,imu_ay,imu_az   --out anomalies.json   --k 7 --n-sigma 3.0   --stl-cols batt_voltage_v --stl-period 60   --gps-jump 1 --gps-alpha 2.0 --gps-margin-m 30
```

3) Output: `anomalies.json` like:

```json
[
  {"index": 120, "timestamp": "2025-01-01T12:00:10Z", "column": "batt_voltage_v", "value": 15.9, "reason": "hampel", "residual": -0.8},
  {"index": 245, "timestamp": "2025-01-01T12:02:05Z", "column": "gps", "value": 185.3, "reason": "gps_jump", "distance_m": 185.3, "allowed_m": 70.1}
]
```

## Arguments

```
detect
  --csv <path.csv>                Input telemetry CSV
  --out <anomalies.json>          Output JSON path
  --time-col <name>               Timestamp column (ISO or parseable)
  --cols <c1,c2,...>              Numeric columns to Hampel-scan
  --k <int>                       Half-window size (neighbors). Default 7
  --n-sigma <float>               MAD threshold. Default 3.0
  --stl-cols <c1,c2,...>          Columns to detrend/seasonal-remove with STL before Hampel
  --stl-period <int>              Seasonal period (samples). Example: 60 for 1 Hz * 1 minute
  --gps-jump <0|1>                Enable GPS jump detector (requires lat,lon,gps_speed_mps)
  --gps-alpha <float>             Allowed multiplier on speed*dt (default 2.0)
  --gps-margin-m <float>          Extra fixed margin in meters (default 30)
```

## Notes

- If `statsmodels` isn't installed, STL is skipped with a warning.
- Time deltas are inferred from `--time-col` differences; if missing/invalid, GPS jump is skipped.
- Hampel flags are independent per column; review context in your log viewer.

## Example data

`example_log.csv` contains a synthetic 10-minute flight at 1 Hz with planted anomalies.

## License

MIT
