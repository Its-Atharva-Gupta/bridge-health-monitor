"""Sensor check + calibration.

  python -m bridge_shm.pi.calibrate check    live readout of every sensor (Ctrl+C to stop)
  python -m bridge_shm.pi.calibrate load     load cell: zero + one known weight at MIDSPAN
  python -m bridge_shm.pi.calibrate sag      sag sensor: several known displacements
  python -m bridge_shm.pi.calibrate strain   strain gauges: zero + sign check

Results are saved into rig_config.json.
"""
import argparse
import time

import numpy as np

from .hx711 import counts_to_microstrain, robust_mean
from .rig import Rig, load_config, save_config


def ask(msg):
    return input(f"\n>>> {msg}\n    (press Enter) ")


def cmd_check(rig):
    print("Live readings. Press on the bridge and watch them change. Ctrl+C to stop.\n")
    t, h = rig.environment()
    print(f"Temperature {t:.1f} C   Humidity {h:.1f} %")
    rig.zero(1.0)
    try:
        while True:
            r = rig.read_step(0.3)
            print("  ".join(f"{k}={v:8.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    except KeyboardInterrupt:
        print()


def cmd_load(rig, cfg, path):
    h = rig.hx.get("load")
    if h is None:
        raise SystemExit("Load cell is disabled in rig_config.json")
    ask("Put the plank, sensors and empty load cup in place. NOTHING else on the bridge.")
    h.tare(3.0)
    grams = float(input("\n>>> Weigh your calibration object on a kitchen scale. Its weight in grams: "))
    ask(f"Put the {grams:g} g object in the cup at the CENTRE of the bridge and let it settle.")
    counts = robust_mean(h.read_many(seconds=3.0)) - h.offset
    if abs(counts) < 1000:
        raise SystemExit(f"Reading barely changed ({counts:.0f} counts). Check wiring / that the free end can bend.")
    cfg["load"]["g_per_count"] = grams / counts
    save_config(path, cfg)
    ask("Remove it, then put it back once more to double-check.")
    check = (robust_mean(h.read_many(seconds=3.0)) - h.offset) * cfg["load"]["g_per_count"]
    print(f"\nSaved. Check reading: {check:.1f} g (should be close to {grams:g} g).")


def cmd_sag(rig, cfg, path):
    if rig.ads is None:
        raise SystemExit("Sag sensor is disabled in rig_config.json")
    ch = cfg["sag"]["channel"]
    print("You will set the sensor to several KNOWN displacements (e.g. stacks of 1 mm cards,\n"
          "or coins you measured). Start at 0 (resting position), then increase.")
    pts = []
    while True:
        s = input("\n>>> Displacement in mm for this position (blank = finished): ").strip()
        if not s:
            break
        mm = float(s)
        v = np.median([rig.ads.read_volts(ch) for _ in range(200)])
        pts.append((v, mm))
        print(f"    {mm:g} mm -> {v:.4f} V")
    if len(pts) < 3:
        raise SystemExit("Need at least 3 positions.")
    v, mm = np.array(pts).T
    v = v - v[0]
    mm = mm - mm[0]
    linear = np.polyfit(v, mm, 1)
    resid1 = np.abs(np.polyval(linear, v) - mm).max()
    poly = linear
    if len(pts) >= 5 and resid1 > 0.1:         # Hall sensors are curved: use a cubic
        poly = np.polyfit(v, mm, 3)
    err = np.abs(np.polyval(poly, v) - mm).max()
    cfg["sag"]["poly_mm_per_volt"] = poly.tolist()
    save_config(path, cfg)
    print(f"\nSaved ({'linear' if len(poly) == 2 else 'curved'} fit). Worst error: {err:.3f} mm")
    if err > 0.2:
        print("That is large. Re-check the displacements and that the sensor is held rigidly.")


def cmd_strain(rig, cfg, path):
    sides = [k for k in ("strain_l", "strain_r") if k in rig.hx]
    if not sides:
        raise SystemExit("Both strain gauges are disabled in rig_config.json")
    ask("Bridge empty and still.")
    for k in sides:
        rig.hx[k].tare(2.0)
    ask("Press DOWN gently on the centre of the bridge with one finger and keep holding")
    for k in sides:
        c = cfg[k]
        counts = robust_mean(rig.hx[k].read_many(seconds=1.5)) - rig.hx[k].offset
        ue = counts_to_microstrain(counts, c["gauge_factor"])
        if abs(ue) < 20:
            print(f"  {k}: barely changed ({ue:.0f} microstrain). Check the gauge wiring/bridge resistors.")
            continue
        # the underside stretches when pressed: that must read POSITIVE
        c["sign"] = 1 if ue > 0 else -1
        print(f"  {k}: {abs(ue):.0f} microstrain -> sign set to {c['sign']:+d}")
    save_config(path, cfg)
    print("Saved.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["check", "load", "sag", "strain"])
    ap.add_argument("--config", default="rig_config.json")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rig = Rig(cfg)
    {"check": lambda: cmd_check(rig),
     "load": lambda: cmd_load(rig, cfg, a.config),
     "sag": lambda: cmd_sag(rig, cfg, a.config),
     "strain": lambda: cmd_strain(rig, cfg, a.config)}[a.what]()


if __name__ == "__main__":
    main()
