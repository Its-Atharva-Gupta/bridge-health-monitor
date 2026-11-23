"""Train on healthy cycles, test on everything, and make the poster plots.

Usage:  python -m bridge_shm.evaluate data/sim        (results go to data/sim/results/)

Honest-testing rules used here:
  * False-alarm rate: healthy cycles are split into 6 groups; each group is scored
    by a model trained on the OTHER groups (it never saw them).
  * Detection rate: damaged cycles are scored by a model trained on all healthy ones.
  * The model never sees a damaged cycle while training.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config as C
from .features import build_table
from .model import DETECTORS, MahalanobisDetector

STAGE_COLORS = {"D0": "#2a9d8f", "D1": "#e9c46a", "D2": "#f4a261", "D3": "#e76f51"}


def false_alarm_rate(cls, healthy, folds=6):
    kf = KFold(folds, shuffle=True, random_state=0)
    flags = np.zeros(len(healthy), bool)
    for tr, te in kf.split(healthy):
        det = cls().fit(healthy.iloc[tr])
        flags[te] = det.is_anomaly(healthy.iloc[te])
    return flags.mean()


def evaluate(df):
    healthy = df[df.stage == "D0"].reset_index(drop=True)
    damaged = df[df.stage != "D0"]
    rows = []
    for cls in DETECTORS:
        det = cls().fit(healthy)
        row = {"model": cls.name, "False alarms (healthy)": false_alarm_rate(cls, healthy)}
        for st in C.STAGES[1:]:
            sub = damaged[damaged.stage == st]
            row[f"Detected {st}"] = det.is_anomaly(sub).mean() if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plots(df, out):
    healthy = df[df.stage == "D0"]
    det = MahalanobisDetector().fit(healthy)
    df = df.copy()
    df["distance"] = det.score(df)
    # healthy cycles get their leave-one-out score (scored by a model that never saw them)
    df.loc[df.stage == "D0", "distance"] = det.loo_scores_
    df["health"] = np.interp(df.distance, [det.typical_, det.threshold_, 2 * det.threshold_, 4 * det.threshold_],
                             [100, 60, 30, 0])
    df["status"] = np.where(df.distance < det.threshold_, "Healthy",
                            np.where(df.distance < 2 * det.threshold_, "Warning", "Damaged"))

    # Figure 1: anomaly score per test cycle
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for st in C.STAGES:
        s = df[df.stage == st]
        ax.scatter(s.index, s.distance, s=22, color=STAGE_COLORS[st], label=f"{st}: {C.STAGE_NAMES[st]}")
    ax.axhline(det.threshold_, color="k", ls="--", lw=1)
    ax.text(0, det.threshold_ * 1.05, "alarm threshold", fontsize=9)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Test cycle (in order)")
    ax.set_ylabel("Anomaly score (Mahalanobis distance)")
    ax.set_title("Every test cycle, scored against the healthy bridge")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig1_score_per_cycle.png", dpi=150); plt.close(fig)

    # Figure 2: sag alone vs the model, per stage
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, lab in ((axs[0], "sag_at_max_mm", "Sag at 600 g (mm)"), (axs[1], "health", "Health score (0-100)")):
        data = [df[df.stage == st][col] for st in C.STAGES]
        bp = ax.boxplot(data, tick_labels=C.STAGES, patch_artist=True, widths=0.6)
        for patch, st in zip(bp["boxes"], C.STAGES):
            patch.set_facecolor(STAGE_COLORS[st])
        ax.set_ylabel(lab)
    axs[0].set_title("What you'd see by measuring sag only")
    axs[1].set_title("What the model sees")
    fig.tight_layout(); fig.savefig(out / "fig2_sag_vs_model.png", dpi=150); plt.close(fig)
    return df, det


def main(root):
    root = Path(root)
    out = root / "results"
    out.mkdir(exist_ok=True)
    df = build_table(root)
    df.to_csv(root / "features.csv", index=False)

    table = evaluate(df)
    scored, det = plots(df, out)
    scored.to_csv(out / "scores.csv", index=False)

    # Figure 3: model comparison
    fig, ax = plt.subplots(figsize=(9, 4))
    cols = [c for c in table.columns if c != "model"]
    x = np.arange(len(cols)); w = 0.8 / len(table)
    for i, r in table.iterrows():
        ax.bar(x + i * w, [100 * r[c] for c in cols], w, label=r["model"])
    ax.set_xticks(x + w * (len(table) - 1) / 2, cols)
    ax.set_ylabel("% of test cycles flagged"); ax.set_ylim(0, 105)
    ax.set_title("Lower is better for healthy, higher is better for damage")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig3_model_comparison.png", dpi=150); plt.close(fig)

    pct = table.copy()
    for c in cols:
        pct[c] = (100 * pct[c]).round(0).astype(int).astype(str) + "%"
    summary = pct.to_string(index=False)
    (out / "summary.txt").write_text(summary + "\n")
    print(summary)
    print("\nMean health score by stage:")
    print(scored.groupby("stage").health.mean().round(0).to_string())
    print("\nWhy the model flags a D1 cycle (most unusual features, in healthy std-devs):")
    d1 = df[df.stage == "D1"]
    if len(d1):
        for f, z in det.explain(d1.iloc[[0]]):
            print(f"  {f:20s} {z:+.1f}")
    print(f"\nplots + scores written to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    main(ap.parse_args().root)
