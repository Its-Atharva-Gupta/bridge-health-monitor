"""Simulated test cycles, so the whole pipeline can be built before real data exists.

Physics (see beam.py): plywood plank resting on two supports, point load at
midspan, damage = zones where the plank is weaker. From that one beam model we get
what every sensor would read: the load cell (under the RIGHT support), both MPU
tilts (1/4 and 3/4 span), sag estimated from those tilts (no sag sensor), the
optional strain gauges, and the tap-test vibration of both MPUs.

Realistic error sources: sensor noise, the load cup sitting slightly off-centre
(different every cycle), tap strength/position, humidity/temperature drift.

IMPORTANT: the damage sizes and the non-geometric effects (crackles, damping,
hysteresis) are educated guesses. The real experiment decides how detectable each
stage is; never put simulated results on the poster.

Usage:  python -m bridge_shm.simulate --out data/sim
"""
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import beam
from . import config as C
from .storage import save_cycle

K0_N_PER_MM = 1.31         # healthy stiffness: ~1 kg at midspan -> ~7.5 mm sag (plywood)
F0_HZ = 50.0               # healthy 1st natural frequency (plank + two MPU boards)
ZETA0 = 0.015              # healthy damping, mode 1
ZETA2 = 0.020              # healthy damping, mode 2
LOAD_POS_SD_MM = 1.5       # how far off-centre the load cup ends up (per cycle)
TILT_NOISE_DEG = 0.02      # MPU6050 tilt noise after averaging 0.5 s
STRAIN_NOISE_UE = 3.0
ACC_NOISE_G = 0.008

# damage: (start mm, end mm, stiffness multiplier) measured from the LEFT support
STAGE = {
    "D0": dict(zones=(), dz=0.000, ae=0.5, beta=0.02, hyst=0.010, resid=0.010),
    "D1": dict(zones=((108, 112, 0.70),), dz=0.002, ae=2.0, beta=0.03, hyst=0.014, resid=0.020),
    "D2": dict(zones=((80, 115, 0.70),), dz=0.004, ae=4.0, beta=0.05, hyst=0.022, resid=0.040),
    "D3": dict(zones=((95, 130, 0.45),), dz=0.007, ae=8.0, beta=0.08, hyst=0.035, resid=0.080),
}


def env_factor(temp_c, rh_pct):
    """Wood gets softer when damp or warm."""
    return 1.0 - 0.003 * (rh_pct - 55.0) - 0.002 * (temp_c - 27.0)


def simulate_cycle(rng, stage, temp_c, rh_pct):
    s = STAGE[stage]
    L = C.SPAN_MM
    zones = s["zones"]
    env = env_factor(temp_c, rh_pct) * rng.normal(1, 0.004)      # + small plank-to-plank scatter
    EI = L ** 3 * K0_N_PER_MM * env / 48.0                       # N*mm^2
    x_load = C.LOAD_X_MM + rng.normal(0, LOAD_POS_SD_MM)
    x, w1, s1, M1 = beam.static(1.0, x_load, zones, EI)          # response to 1 N
    at = lambda arr, X: float(np.interp(X, x, arr))
    th_l, th_r = at(s1, C.MPU_X_MM), at(s1, L - C.MPU_X_MM)     # rad per N (left +, right -)
    eps_l = at(M1, C.GAUGE_L_X_MM) * C.PLANK_T_MM / 2 / (EI * beam.ei_factor(np.array([C.GAUGE_L_X_MM]), zones)[0])
    eps_r = at(M1, C.GAUGE_R_X_MM) * C.PLANK_T_MM / 2 / (EI * beam.ei_factor(np.array([C.GAUGE_R_X_MM]), zones)[0])
    Fmax = max(C.LOAD_STEPS_G) * C.G
    sag_max = at(w1, C.LOAD_X_MM) * Fmax

    rows = []
    loads = C.LOAD_STEPS_G + C.LOAD_STEPS_G[-2::-1]
    for i, g in enumerate(loads):
        phase = "load" if i < len(C.LOAD_STEPS_G) else "unload"
        F = g * C.G
        scale = F * (1 + s["beta"] * F / Fmax)                  # softening at high load
        extra = 0.0                                              # leftover sag while unloading (mm)
        if phase == "unload":
            extra = s["resid"] * (1 - F / Fmax) + s["hyst"] * sag_max * np.sin(np.pi * F / Fmax)
        k = extra / max(at(w1, C.LOAD_X_MM), 1e-12)             # same shape, expressed as extra newtons
        tl = np.degrees((scale + k) * th_l) + rng.normal(0, TILT_NOISE_DEG)
        tr = np.degrees((scale + k) * th_r) + rng.normal(0, TILT_NOISE_DEG)
        sag = np.mean([abs(np.radians(tl)), abs(np.radians(tr))]) * C.SAG_PER_TILT_RAD
        # load cell under the RIGHT support, calibrated with the load at midspan
        g_meas = g * x_load / C.LOAD_X_MM + rng.normal(0, 1.5)
        rows.append(dict(
            phase=phase, load_g=round(g_meas, 2), sag_mm=sag,
            strain_l_ue=eps_l * F * 1e6 + rng.normal(0, STRAIN_NOISE_UE),
            strain_r_ue=eps_r * F * 1e6 + rng.normal(0, STRAIN_NOISE_UE),
            tilt_deg=tl, tilt2_deg=tr,
        ))
    steps = pd.DataFrame(rows)

    # tap test: both modes ring; each MPU sees them weighted by the mode shapes
    o1, o2, nodes, sh1, sh2 = beam.modes(zones)
    h1, _, _, _, _ = beam.modes(())
    f1 = F0_HZ * np.sqrt(env) * o1 / h1
    f2 = F0_HZ * np.sqrt(env) * o2 / h1
    z1 = ZETA0 + s["dz"] + rng.normal(0, 0.0008)
    z2 = ZETA2 + s["dz"] + rng.normal(0, 0.0010)
    x_tap = C.TAP_X_MM + rng.normal(0, 3)
    t = np.arange(0, 3.0, 1 / C.TAP_FS_HZ)
    t0 = rng.uniform(0.2, 0.6)
    tt = np.clip(t - t0, 0, None)
    A = rng.uniform(0.3, 0.8)                                  # tap strength varies

    def ring(xs):
        a = np.zeros_like(t)
        for f, z, sh, wgt in ((f1, z1, sh1, 1.0), (f2, z2, sh2, 0.5)):
            amp = A * wgt * beam.shape_at(nodes, sh, x_tap) * beam.shape_at(nodes, sh, xs)
            a += amp * np.exp(-z * 2 * np.pi * f * tt) * np.sin(2 * np.pi * f * np.sqrt(1 - z * z) * tt)
        a[t < t0] = 0
        return a + rng.normal(0, ACC_NOISE_G, t.size)

    tap = pd.DataFrame(dict(t_s=t, accel_g=ring(C.MPU_X_MM), accel2_g=ring(L - C.MPU_X_MM)))

    # crackles during loading
    n = rng.poisson(s["ae"])
    amp = rng.lognormal(-3, 0.6, n)
    ae = pd.DataFrame(dict(t_s=np.sort(rng.uniform(0, 60, n)), amplitude=amp, energy=amp ** 2))
    return steps, tap, ae


def simulate_dataset(out, seed=0, n_healthy=36, n_damaged=12):
    rng = np.random.default_rng(seed)
    t = datetime(2026, 12, 1, 10, 0)
    plan = [("D0", n_healthy, 3)] + [(st, n_damaged, 1) for st in C.STAGES[1:]]
    cid = 0
    for stage, n, sessions in plan:
        per = int(np.ceil(n / sessions))
        for sess in range(sessions):
            temp_m, rh_m = rng.normal(27, 2), rng.uniform(40, 75)   # each session = a different day
            for _ in range(min(per, n - sess * per)):
                temp, rh = temp_m + rng.normal(0, 0.4), rh_m + rng.normal(0, 1.5)
                steps, tap, ae = simulate_cycle(rng, stage, temp, rh)
                meta = dict(cycle_id=f"c{cid:04d}", plank="sim_plank", stage=stage,
                            timestamp=t.isoformat(), temp_c=round(temp, 2), rh_pct=round(rh, 2),
                            source="simulated", sag_source="tilt")
                save_cycle(out, meta, steps, tap, ae)
                cid += 1
                t += timedelta(minutes=4)
            t += timedelta(days=1)
    return cid


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/sim")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n = simulate_dataset(a.out, a.seed)
    print(f"wrote {n} simulated cycles to {a.out}")
