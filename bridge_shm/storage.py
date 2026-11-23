"""How one test cycle is saved on disk (same format for simulated and real data).

data/<plank>/<stage>/<cycle_id>/
    meta.json   cycle id, plank, stage, time, temperature, humidity, source
    steps.csv   one row per load step: phase, load_g, sag_mm, strain_l_ue, strain_r_ue, tilt_deg
    tap.csv     t_s, accel_g        (MPU6050 vertical acceleration after a tap)
    ae.csv      t_s, amplitude, energy   (one row per detected crackle; may be empty)

Plain CSV/JSON on purpose: you can open every file in Excel or a text editor.
"""
import json
from pathlib import Path

import pandas as pd


def save_cycle(root, meta, steps, tap, ae):
    d = Path(root) / meta["plank"] / meta["stage"] / meta["cycle_id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    steps.to_csv(d / "steps.csv", index=False)
    tap.to_csv(d / "tap.csv", index=False)
    ae.to_csv(d / "ae.csv", index=False)
    return d


def load_cycle(d):
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    steps = pd.read_csv(d / "steps.csv")
    tap = pd.read_csv(d / "tap.csv")
    ae_path = d / "ae.csv"
    ae = pd.read_csv(ae_path) if ae_path.stat().st_size > 0 else pd.DataFrame(columns=["t_s", "amplitude", "energy"])
    return meta, steps, tap, ae


def find_cycles(root):
    return sorted(p.parent for p in Path(root).rglob("meta.json"))
