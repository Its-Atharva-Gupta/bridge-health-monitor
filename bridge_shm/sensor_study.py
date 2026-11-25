"""Which sensors actually matter? Re-run the whole analysis with sensors left out.

Same test cycles every time; a "missing" sensor is simply ignored (and sag is
re-estimated from one MPU when the second is left out). Works on simulated data
(several seeds, averaged) or on your real data folder.

  python -m bridge_shm.sensor_study                 simulated, 5 datasets
  python -m bridge_shm.sensor_study --data data/real
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .evaluate import false_alarm_rate
from .features import build_table
from .model import MahalanobisDetector, SagThresholdDetector
from .simulate import simulate_dataset

SCENARIOS = [
    ("2 MPUs + strain gauges + piezo", ()),
    ("2 MPUs + piezo (no strain gauges)", ("strain",)),
    ("1 MPU + piezo (no strain gauges)", ("strain", "mpu2")),
    ("2 MPUs, no strain, no piezo", ("strain", "ae")),
    ("2 MPUs, no strain, no tap test", ("strain", "tap")),
]


def score(df, cls):
    healthy = df[df.stage == "D0"].reset_index(drop=True)
    det = cls().fit(healthy)
    row = {"False alarms": false_alarm_rate(cls, healthy)}
    for st in C.STAGES[1:]:
        sub = df[df.stage == st]
        row[f"{st} caught"] = det.is_anomaly(sub).mean() if len(sub) else np.nan
    return row


def study(roots):
    rows = []
    for name, drop in SCENARIOS:
        per = [score(build_table(r, drop), MahalanobisDetector) for r in roots]
        rows.append({"Sensors": name, **pd.DataFrame(per).mean().to_dict()})
    base = [score(build_table(r, ("strain",)), SagThresholdDetector) for r in roots]
    rows.append({"Sensors": "Sag threshold only (no ML), 2 MPUs", **pd.DataFrame(base).mean().to_dict()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder of real cycles (default: simulate)")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    if a.data:
        table = study([Path(a.data)])
    else:
        with tempfile.TemporaryDirectory() as tmp:
            roots = []
            for s in range(a.seeds):
                r = Path(tmp) / f"seed{s}"
                simulate_dataset(r, seed=s)
                roots.append(r)
            table = study(roots)
    pct = table.copy()
    for c in pct.columns[1:]:
        pct[c] = (100 * pct[c]).round(0).astype(int).astype(str) + "%"
    print(pct.to_string(index=False))


if __name__ == "__main__":
    main()
