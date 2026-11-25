"""Guided test cycles: walks you through loading, unloading and the tap test,
then saves each cycle in the same format the analysis expects.

  python -m bridge_shm.pi.run_test --plank P1 --stage D0 --cycles 5

Try it with no hardware:  python -m bridge_shm.pi.run_test --fake --yes --stage D0 --cycles 3
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..features import extract
from ..storage import save_cycle
from .rig import FakeRig, Rig, load_config


class Prompter:
    def __init__(self, auto):
        self.auto = auto

    def __call__(self, msg):
        print(f"\n>>> {msg}")
        if not self.auto:
            input("    (press Enter) ")


def next_cycle_id(root, plank):
    existing = list((Path(root) / plank).glob("*/c*"))
    nums = [int(p.name[1:]) for p in existing if p.name[1:].isdigit()]
    return f"c{(max(nums) + 1 if nums else 0):04d}"


def run_cycle(rig, ask, plank, stage, root, hold, notes=""):
    ask("Bridge EMPTY (just the cup), nothing touching it. Zeroing sensors...")
    temp, rh = rig.environment()
    rig.zero()
    listener = rig.crackle_listener()
    if listener:
        print("    listening for 1 s of quiet...")
        listener.start()

    rows = []
    loads = C.LOAD_STEPS_G + C.LOAD_STEPS_G[-2::-1]
    for i, g in enumerate(loads):
        phase = "load" if i < len(C.LOAD_STEPS_G) else "unload"
        if i == 0:
            print("    reading empty bridge...")
        else:
            verb = "ADD weights so the cup holds" if phase == "load" else "REMOVE weights so the cup holds"
            ask(f"[{phase} {i}/{len(loads) - 1}] {verb} {g} g in total. Keep your hands off.")
            if not getattr(ask, "auto", False):
                time.sleep(hold)                     # let creep settle the same way every time
        r = rig.read_step()
        r.update(phase=phase, nominal_g=g)
        rows.append(r)
        print(f"    load {r['load_g']:7.1f} g   sag {r['sag_mm']:6.2f} mm   "
              f"tilt L {r['tilt_deg']:6.2f}  R {r['tilt2_deg']:6.2f} deg   "
              f"strain L {r['strain_l_ue']:7.1f}  R {r['strain_r_ue']:7.1f}")

    ae = listener.stop()[0] if listener else pd.DataFrame(columns=["t_s", "amplitude", "energy"])

    while True:
        ask(f"TAP TEST: after pressing Enter, tap the marked spot ({C.TAP_X_MM:.0f} mm from the left "
            "support) ONCE with a pencil, within 1 second.")
        t, acc, acc2 = rig.record_tap(3.0)
        tap = pd.DataFrame(dict(t_s=t, accel_g=acc, accel2_g=acc2 if acc2 is not None else np.nan))
        problem = tap_problem(tap)
        if not problem:
            break
        print(f"    {problem} Let's redo it.")

    steps = pd.DataFrame(rows)
    sag_source = steps.pop("sag_source").iloc[0] if "sag_source" in steps else "sensor"
    steps = steps[["phase", "nominal_g", "load_g", "sag_mm", "strain_l_ue", "strain_r_ue",
                   "tilt_deg", "tilt2_deg"]]
    meta = dict(cycle_id=next_cycle_id(root, plank), plank=plank, stage=stage,
                timestamp=datetime.now().isoformat(timespec="seconds"),
                temp_c=round(float(temp), 2), rh_pct=round(float(rh), 2),
                source="fake" if isinstance(rig, FakeRig) else "real",
                sag_source=sag_source, notes=notes)
    d = save_cycle(root, meta, steps, tap, ae)

    f = extract(d)
    print(f"\n  Saved {d}")
    print(f"  stiffness {f['stiffness_n_per_mm']:.3f} N/mm | sag@max {f['sag_at_max_mm']:.2f} mm | "
          f"freq {f['freq_hz']:.1f} / {f['freq2_hz']:.1f} Hz | damping {f['damping_ratio']:.4f} | "
          f"tilt L/R {f['tilt_ratio_lr']:.3f} | crackles {len(ae)} | "
          f"{temp:.1f} C {rh:.0f}%")
    sanity(steps)
    return d


def tap_problem(tap):
    """Reasons to redo the tap test right away (None = fine)."""
    chans = [c for c in ("accel_g", "accel2_g") if tap[c].notna().all()]
    if max(np.abs(tap[c] - tap[c].median()).max() for c in chans) < 0.05:
        return "No clear tap detected (signal too small)."
    if len(chans) == 2:
        # both MPUs sit on the same plank, so they must agree on its frequency
        from ..features import _one_channel
        fs = 1.0 / np.median(np.diff(tap.t_s))
        f1, f2 = (_one_channel(tap[c].to_numpy(), fs, 5.0, 300.0)[0] for c in chans)
        if abs(f1 - f2) > 0.03 * max(f1, f2):
            return f"The two MPUs disagree on the frequency ({f1:.1f} vs {f2:.1f} Hz)."
    return None


def sanity(steps):
    """Warn about obvious problems right away, while you can still redo the cycle."""
    load = steps[steps.phase == "load"]
    if load.load_g.notna().all():
        err = (load.load_g - load.nominal_g).abs().max()
        if err > 20:
            print(f"  ! Load cell differs from the nominal weights by up to {err:.0f} g "
                  f"(weights off-centre? needs calibration?)")
    if (np.diff(load.sag_mm.to_numpy()) <= 0).any():
        print("  ! Sag did not increase at every load step: something slipped or a sensor is loose.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plank", default="P1")
    ap.add_argument("--stage", required=True, choices=C.STAGES)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--data", default="data/real")
    ap.add_argument("--config", default="rig_config.json")
    ap.add_argument("--notes", default="")
    ap.add_argument("--fake", action="store_true", help="simulated sensors (no hardware)")
    ap.add_argument("--yes", action="store_true", help="don't wait for Enter (fake mode demos)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    rig = FakeRig(cfg, a.stage) if a.fake else Rig(cfg)
    ask = Prompter(a.yes)
    ask.auto = a.yes
    root = "data/fake" if a.fake and a.data == "data/real" else a.data
    print(f"Plank {a.plank}, stage {a.stage} ({C.STAGE_NAMES[a.stage]}), {a.cycles} cycle(s) -> {root}")
    for n in range(a.cycles):
        print(f"\n========== cycle {n + 1} of {a.cycles} ==========")
        run_cycle(rig, ask, a.plank, a.stage, root, C.HOLD_SECONDS, a.notes)
    print("\nDone.")


if __name__ == "__main__":
    main()
