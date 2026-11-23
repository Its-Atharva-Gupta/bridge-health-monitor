"""Turn one test cycle (thousands of raw readings) into one row of features.

Usage:  python -m bridge_shm.features data/sim     -> writes data/sim/features.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

from . import config as C
from .storage import find_cycles, load_cycle


def load_sag_features(steps):
    """Stiffness, softening, hysteresis and leftover sag from the load steps."""
    load = steps[steps.phase == "load"]
    unload = steps[steps.phase == "unload"]
    F = load.load_g.to_numpy() * C.G
    d = load.sag_mm.to_numpy()

    # stiffness: best straight line through load vs sag (N per mm)
    stiffness = np.polyfit(d, F, 1)[0]

    # softening: fit sag = a*F + b*F^2; nonlinearity = extra sag fraction at max load
    b, a, _ = np.polyfit(F, d, 2)
    nonlinearity = b * F.max() / a

    # hysteresis: area between loading and unloading curves (N*mm = mJ)
    Fu = unload.load_g.to_numpy() * C.G
    du = unload.sag_mm.to_numpy()
    Fu_full = np.concatenate([[F.max()], Fu])
    du_full = np.concatenate([[d[-1]], du])
    grid = np.linspace(0, F.max(), 50)
    d_up = np.interp(grid, F, d)
    order = np.argsort(Fu_full)
    d_down = np.interp(grid, Fu_full[order], du_full[order])
    hysteresis = np.trapezoid(d_down - d_up, grid)

    residual = du[-1] - d[0]                    # sag after unloading vs start
    return dict(stiffness_n_per_mm=stiffness, nonlinearity=nonlinearity,
                hysteresis_mj=hysteresis, residual_sag_mm=residual)


def strain_features(steps):
    nan = dict(strain_per_n_l=np.nan, strain_per_n_r=np.nan, strain_ratio_lr=np.nan)
    if steps[["strain_l_ue", "strain_r_ue"]].isna().all().any():
        return nan                                  # gauges not fitted yet
    load = steps[steps.phase == "load"]
    F = load.load_g.to_numpy() * C.G
    sl = np.polyfit(F, load.strain_l_ue.to_numpy(), 1)[0]
    sr = np.polyfit(F, load.strain_r_ue.to_numpy(), 1)[0]
    return dict(strain_per_n_l=sl, strain_per_n_r=sr, strain_ratio_lr=sl / sr)


def tap_features(tap, fmin=5.0, fmax=300.0):
    """Natural frequency (FFT peak) and damping (decay of the vibration envelope)."""
    t = tap.t_s.to_numpy()
    a = tap.accel_g.to_numpy() - np.median(tap.accel_g)
    fs = 1.0 / np.median(np.diff(t))
    start = int(np.argmax(np.abs(a)))                       # the tap
    seg = a[start:]
    win = seg * np.hanning(seg.size)
    n = 8 * seg.size                                        # zero-pad: finer frequency grid
    spec = np.abs(np.fft.rfft(win, n))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    i = np.flatnonzero(band)[np.argmax(spec[band])]
    # parabolic interpolation around the peak
    y0, y1, y2 = np.log(spec[i - 1:i + 2] + 1e-12)
    f = freqs[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (freqs[1] - freqs[0])

    # damping: isolate the mode, take its envelope, fit the exponential decay
    sos = butter(2, [0.7 * f, 1.3 * f], btype="band", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, seg)))
    # fit only the clean part of the decay: from where the envelope falls below
    # 90% of its peak until it reaches 15% (later = noise / filter artefacts)
    noise = np.median(np.abs(a[: max(start - 10, 10)]))
    skip = int(0.02 * fs)                                   # ignore the first 20 ms (tap transient)
    peak = env[skip:].max()
    floor = max(0.15 * peak, 10 * noise)
    after = np.arange(env.size) >= skip
    lo_idx = np.flatnonzero(after & (env < 0.9 * peak))
    hi_idx = np.flatnonzero(after & (env < floor))
    tt = np.arange(env.size) / fs
    if lo_idx.size and hi_idx.size and hi_idx[0] - lo_idx[0] > 20:
        lo, stop = lo_idx[0], hi_idx[0]
        slope = np.polyfit(tt[lo:stop], np.log(env[lo:stop]), 1)[0]
        zeta = -slope / (2 * np.pi * f)
    else:
        zeta = np.nan
    return dict(freq_hz=f, damping_ratio=zeta)


def ae_features(ae):
    return dict(ae_count_log=np.log1p(len(ae)),
                ae_energy_log=np.log1p(1e3 * ae.energy.sum()) if len(ae) else 0.0)


def extract(cycle_dir):
    meta, steps, tap, ae = load_cycle(cycle_dir)
    row = dict(cycle_id=meta["cycle_id"], plank=meta["plank"], stage=meta["stage"],
               timestamp=meta["timestamp"], temp_c=meta["temp_c"], rh_pct=meta["rh_pct"],
               source=meta.get("source", ""))
    row["sag_at_max_mm"] = steps[steps.phase == "load"].sag_mm.iloc[-1] - steps.sag_mm.iloc[0]
    row.update(load_sag_features(steps))
    row.update(strain_features(steps))
    row.update(tap_features(tap))
    row.update(ae_features(ae))
    return row


def build_table(root):
    rows = [extract(d) for d in find_cycles(root)]
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    a = ap.parse_args()
    df = build_table(a.root)
    out = Path(a.root) / "features.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} cycles -> {out}")
    print(df.groupby("stage")[["sag_at_max_mm"] + C.FEATURES].mean().T.round(4).to_string())
