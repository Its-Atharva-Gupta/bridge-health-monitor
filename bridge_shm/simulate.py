"""Simulated test cycles, so the whole pipeline can be built before the bridge exists.

Physics: simply supported balsa plank, point load at midspan (same numbers as the
bridge design: ~7.2 mm sag at 600 g). Adds sensor noise, humidity/temperature
drift, tap-strength variation and the D0-D3 damage stages.

IMPORTANT: the damage effect sizes below are educated guesses. The real
experiment decides how detectable each stage is; never put simulated results
on the poster.

Usage:  python -m bridge_shm.simulate --out data/sim
"""
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config as C
from .storage import save_cycle

K0_N_PER_MM = 0.823        # healthy stiffness: 5.886 N / 7.15 mm
F0_HZ = 60.0               # healthy natural frequency (plank + MPU6050 mass)
ZETA0 = 0.015              # healthy damping ratio

# per-stage changes (guesses, see module docstring)
STAGE = {
    #       sag x   L/R asym  +damping  crackles  softening  hysteresis  residual mm
    "D0": dict(sag=1.000, asym=0.00, dz=0.000, ae=0.5, beta=0.02, hyst=0.010, resid=0.010),
    "D1": dict(sag=1.007, asym=0.03, dz=0.002, ae=2.0, beta=0.03, hyst=0.014, resid=0.020),
    "D2": dict(sag=1.130, asym=0.07, dz=0.004, ae=4.0, beta=0.05, hyst=0.022, resid=0.040),
    "D3": dict(sag=1.400, asym=0.04, dz=0.007, ae=8.0, beta=0.08, hyst=0.035, resid=0.080),
}


def env_factor(temp_c, rh_pct):
    """Balsa gets softer when damp or warm."""
    return 1.0 - 0.003 * (rh_pct - 55.0) - 0.002 * (temp_c - 27.0)


def simulate_cycle(rng, stage, temp_c, rh_pct):
    s = STAGE[stage]
    k_env = K0_N_PER_MM * env_factor(temp_c, rh_pct)          # undamaged, today's weather
    k = k_env / s["sag"] * rng.normal(1, 0.007)               # whole-bridge stiffness
    L = C.SPAN_MM
    EI = L ** 3 * k / 48.0          # "effective" EI, used for global quantities (tilt)
    # the gauges sit OUTSIDE the damage zone: the moment there is fixed by statics
    # and the local section is undamaged, so their strain only feels the weather
    EI_gauge = L ** 3 * k_env / 48.0
    Fmax = max(C.LOAD_STEPS_G) * C.G

    rows = []
    loads = C.LOAD_STEPS_G + C.LOAD_STEPS_G[-2::-1]
    for i, g in enumerate(loads):
        phase = "load" if i < len(C.LOAD_STEPS_G) else "unload"
        g_meas = g + rng.normal(0, 1.5)                       # load cell noise
        F = g * C.G
        sag = F / k * (1 + s["beta"] * F / Fmax)
        if phase == "unload":
            dmax = Fmax / k
            sag += s["resid"] * (1 - F / Fmax) + s["hyst"] * dmax * np.sin(np.pi * F / Fmax)
        # bending moment at the gauges (right of the load): M = F/2 * (L - x)
        M = F / 2 * (L - C.GAUGE_X_MM)
        eps = M * (C.PLANK_T_MM / 2) / EI_gauge * 1e6         # microstrain
        x = C.MPU_X_MM
        tilt = np.degrees(F * (L ** 2 - 4 * x ** 2) / (16 * EI))
        rows.append(dict(
            phase=phase, load_g=round(g_meas, 2),
            sag_mm=sag + rng.normal(0, 0.03),
            strain_l_ue=eps * (1 + s["asym"]) + rng.normal(0, 3),
            strain_r_ue=eps + rng.normal(0, 3),
            tilt_deg=tilt + rng.normal(0, 0.02),
        ))
    steps = pd.DataFrame(rows)

    # tap test: decaying vibration at the natural frequency (+ a weaker 2nd mode)
    f = F0_HZ * np.sqrt(k / K0_N_PER_MM)
    zeta = ZETA0 + s["dz"] + rng.normal(0, 0.0008)
    t = np.arange(0, C.TAP_SECONDS, 1 / C.TAP_FS_HZ)
    t0 = 0.2
    tt = np.clip(t - t0, 0, None)
    A = rng.uniform(0.3, 0.8)                                 # tap strength varies
    wd = 2 * np.pi * f * np.sqrt(1 - zeta ** 2)
    acc = A * np.exp(-zeta * 2 * np.pi * f * tt) * np.sin(wd * tt)
    f2 = 3.9 * f
    acc += 0.3 * A * np.exp(-0.03 * 2 * np.pi * f2 * tt) * np.sin(2 * np.pi * f2 * tt)
    acc[t < t0] = 0
    acc += rng.normal(0, 0.01, t.size)
    tap = pd.DataFrame(dict(t_s=t, accel_g=acc))

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
                            source="simulated")
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
