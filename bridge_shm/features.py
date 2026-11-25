"""Turn one test cycle (thousands of raw readings) into one row of features.

Usage:  python -m bridge_shm.features data/sim     -> writes data/sim/features.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

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


def tilt_features(steps):
    """Left (1/4 span) vs right (3/4 span) MPU tilt per newton. Healthy = about 1."""
    if "tilt2_deg" not in steps or steps[["tilt_deg", "tilt2_deg"]].isna().any().any():
        return dict(tilt_ratio_lr=np.nan)                 # only one MPU fitted
    load = steps[steps.phase == "load"]
    F = load.load_g.to_numpy() * C.G
    sl = np.polyfit(F, np.abs(load.tilt_deg.to_numpy()), 1)[0]
    sr = np.polyfit(F, np.abs(load.tilt2_deg.to_numpy()), 1)[0]
    return dict(tilt_ratio_lr=sl / sr)


def _peak_freq(freqs, spec, i):
    """Parabolic interpolation around spectrum bin i (sub-bin frequency accuracy)."""
    y0, y1, y2 = np.log(spec[i - 1:i + 2] + 1e-12)
    den = y0 - 2 * y1 + y2
    return freqs[i] + (0.5 * (y0 - y2) / den if den != 0 else 0.0) * (freqs[1] - freqs[0])


def _one_channel(a, fs, fmin, fmax):
    """Mode-1 frequency, damping, and the spectrum of one accelerometer."""
    a = a - np.median(a)
    start = int(np.argmax(np.abs(a)))                       # the tap
    seg = a[start:]
    n = 8 * seg.size                                        # zero-pad: finer frequency grid
    # half-Hann window: full weight at the tap (where the ringing is strongest),
    # fading to zero at the end. A normal Hann window would start at zero and
    # throw away most of a quickly decaying vibration.
    win = np.hanning(2 * seg.size)[seg.size:]
    spec = np.abs(np.fft.rfft(seg * win, n))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    band = np.flatnonzero((freqs >= fmin) & (freqs <= fmax))
    # mode 1 = the LOWEST strong peak (mode 2 can be taller, depending on where you tap)
    # (prominence, not just height: noise ripples on the side of a peak must not count)
    pk, _ = find_peaks(spec[band], height=0.25 * spec[band].max(), prominence=0.2 * spec[band].max())
    i = band[pk[0]] if pk.size else band[np.argmax(spec[band])]
    f = _peak_freq(freqs, spec, i)

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
    zeta = np.nan
    if lo_idx.size and hi_idx.size and hi_idx[0] - lo_idx[0] > 20:
        lo, stop = lo_idx[0], hi_idx[0]
        slope = np.polyfit(tt[lo:stop], np.log(env[lo:stop]), 1)[0]
        zeta = -slope / (2 * np.pi * f)
    return f, zeta, freqs, spec


def tap_features(tap, fmin=5.0, fmax=300.0):
    """Natural frequency + damping (mode 1) and the 2nd mode frequency.

    Uses every accelerometer in tap.csv (accel_g = MPU at 1/4, accel2_g = MPU at 3/4).
    Mode 2 is found in the summed power spectrum, which does not care whether the
    two MPUs started recording at exactly the same instant.
    """
    t = tap.t_s.to_numpy()
    fs = 1.0 / np.median(np.diff(t))
    chans = [c for c in ("accel_g", "accel2_g") if c in tap and tap[c].notna().all()]
    res = [_one_channel(tap[c].to_numpy(), fs, fmin, fmax) for c in chans]
    f1 = float(np.mean([r[0] for r in res]))
    zeta = float(np.nanmean([r[1] for r in res])) if any(np.isfinite(r[1]) for r in res) else np.nan
    n = min(r[3].size for r in res)
    freqs = res[0][2][:n]
    power = sum(r[3][:n] ** 2 for r in res)
    band = np.flatnonzero((freqs >= 2.5 * f1) & (freqs <= min(6 * f1, 0.45 * fs)))
    f2 = np.nan
    if band.size > 3:
        i = band[np.argmax(power[band])]
        if 0 < i < power.size - 1:
            f2 = _peak_freq(freqs, power, i)
    return dict(freq_hz=f1, damping_ratio=zeta, freq2_hz=f2)


def ae_features(ae):
    return dict(ae_count_log=np.log1p(len(ae)),
                ae_energy_log=np.log1p(1e3 * ae.energy.sum()) if len(ae) else 0.0)


# Sensors that can be left out (for real rigs missing a sensor, or the sensor study)
DROPPABLE = ("strain", "mpu2", "ae", "tap")


def extract(cycle_dir, drop=()):
    """One feature row. `drop` pretends sensors were absent, e.g. drop=("strain", "mpu2")."""
    meta, steps, tap, ae = load_cycle(cycle_dir)
    if "mpu2" in drop and "tilt2_deg" in steps:
        steps = steps.copy()
        steps["tilt2_deg"] = np.nan
        tap = tap.drop(columns=["accel2_g"], errors="ignore")
        if meta.get("sag_source") == "tilt":                  # sag came from both MPUs: redo with one
            steps["sag_mm"] = np.abs(np.radians(steps.tilt_deg)) * C.SAG_PER_TILT_RAD
    row = dict(cycle_id=meta["cycle_id"], plank=meta["plank"], stage=meta["stage"],
               timestamp=meta["timestamp"], temp_c=meta["temp_c"], rh_pct=meta["rh_pct"],
               source=meta.get("source", ""))
    row["sag_at_max_mm"] = steps[steps.phase == "load"].sag_mm.iloc[-1] - steps.sag_mm.iloc[0]
    row.update(load_sag_features(steps))
    row.update(strain_features(steps))
    row.update(tilt_features(steps))
    tf = tap_features(tap)
    if "tap" in drop:
        tf = {k: np.nan for k in tf}
    row.update(tf)
    af = ae_features(ae)
    if "ae" in drop:
        af = {k: np.nan for k in af}
    row.update(af)
    if "strain" in drop:
        row.update(strain_per_n_l=np.nan, strain_per_n_r=np.nan, strain_ratio_lr=np.nan)
    return row


def build_table(root, drop=()):
    rows = [extract(d, drop) for d in find_cycles(root)]
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
