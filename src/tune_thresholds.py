import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import COSTS, HUMAN_ACCURACY, DEFECT_RATE

GRID = np.append(np.round(np.arange(0, 1.0001, 0.005), 3), 1.01)


def evaluate(d, g, t_low, t_high, p=DEFECT_RATE):
    """Expected outcomes per 1,000 parts for a pass / review / act policy.
    d = scores of defective parts, g = scores of good parts."""
    d_pass = np.mean(d < t_low)
    d_rev = np.mean((d >= t_low) & (d < t_high))
    d_act = np.mean(d >= t_high)
    g_rev = np.mean((g >= t_low) & (g < t_high))
    g_act = np.mean(g >= t_high)

    human_miss = 1 - HUMAN_ACCURACY
    cost_d = (d_pass * COSTS["miss"]
              + d_rev * (COSTS["review"] + human_miss * COSTS["miss"])
              + d_act * COSTS["catch"])
    cost_g = g_rev * COSTS["review"] + g_act * COSTS["false_stop"]

    n_d, n_g = 1000 * p, 1000 * (1 - p)
    return {
        "t_low": t_low,
        "t_high": t_high,
        "cost_per_1000": n_d * cost_d + n_g * cost_g,
        "missed": n_d * (d_pass + d_rev * human_miss),
        "false_stops": n_g * g_act,
        "reviews": n_d * d_rev + n_g * g_rev,
    }


def best_policy(d, g, p=DEFECT_RATE, allow_review=True):
    """Search all threshold pairs. Ties go to the most cautious option
    (lowest t_low, highest t_high = widest review band)."""
    best, best_key = None, None
    for lo in GRID:
        highs = GRID[GRID >= lo] if allow_review else [lo]
        for hi in highs:
            r = evaluate(d, g, lo, hi, p)
            key = (round(r["cost_per_1000"], 4), round(r["reviews"], 4), lo, -hi)
            if best_key is None or key < best_key:
                best, best_key = r, key
    return best


def main():
    df = pd.read_csv("outputs/scores.csv")
    d = df[df.is_defect == 1].score.values
    g = df[df.is_defect == 0].score.values

    policies = {
        "1. Default (0.5, no review)": evaluate(d, g, 0.5, 0.5),
        "2. Best single threshold": best_policy(d, g, allow_review=False),
        "3. Review everything": evaluate(d, g, 0.0, 1.01),
        "4. Best three tiers": best_policy(d, g, allow_review=True),
    }

    table = pd.DataFrame(policies).T[
        ["t_low", "t_high", "cost_per_1000", "missed", "false_stops", "reviews"]]
    pd.options.display.float_format = "{:,.2f}".format
    print(f"\nEXPECTED OUTCOMES PER 1,000 PARTS  (defect rate {DEFECT_RATE:.0%})")
    print(table.to_string())

    best = policies["4. Best three tiers"]
    default = policies["1. Default (0.5, no review)"]
    saving = 1 - best["cost_per_1000"] / default["cost_per_1000"]
    print(f"\nChosen tiers:  PASS < {best['t_low']:.3f}  |  REVIEW  |  ACT >= {best['t_high']:.3f}")
    print(f"Cost vs default threshold: {saving:.0%} lower")

    # --- Sensitivity: how the best tiers change with the defect rate ---
    print("\nSENSITIVITY: best tiers at different defect rates")
    rows = []
    for p in [0.005, 0.01, 0.02, 0.05, 0.10]:
        r = best_policy(d, g, p=p)
        rows.append({"defect_rate": f"{p:.1%}", "t_low": r["t_low"], "t_high": r["t_high"],
                     "cost_per_1000": r["cost_per_1000"], "reviews": r["reviews"]})
    print(pd.DataFrame(rows).to_string(index=False))

    # --- Save thresholds for the decision layer (Step 6) ---
    out = {
        "t_low": float(best["t_low"]),
        "t_high": float(best["t_high"]),
        "defect_rate": DEFECT_RATE,
        "costs": COSTS,
        "human_accuracy": HUMAN_ACCURACY,
        "expected_cost_per_1000": round(float(best["cost_per_1000"]), 2),
    }
    Path("outputs/thresholds.json").write_text(json.dumps(out, indent=2))
    print("\nSaved outputs/thresholds.json")

    # --- Chart: the three tiers on the score distribution ---
    fig, ax = plt.subplots(figsize=(9, 4.2))
    lo, hi = best["t_low"], min(best["t_high"], 1.0)
    ax.axvspan(0, lo, color="#2F6FDB", alpha=0.07)
    ax.axvspan(lo, hi, color="#888888", alpha=0.18)
    ax.axvspan(hi, 1.0, color="#F28C28", alpha=0.10)
    bins = np.linspace(0, 1, 41)
    ax.hist(g, bins=bins, color="#2F6FDB", alpha=0.9, label=f"good ({len(g)})")
    ax.hist(d, bins=bins, color="#F28C28", alpha=0.9, label=f"defective ({len(d)})")
    top = ax.get_ylim()[1]
    ax.text(lo / 2, top * 0.93, "PASS", ha="center", fontsize=11, fontweight="bold", color="#2F6FDB")
    ax.text((lo + hi) / 2, top * 0.93, "REVIEW", ha="center", fontsize=11, fontweight="bold", color="#555555")
    ax.text((hi + 1) / 2, top * 0.93, "ACT", ha="center", fontsize=11, fontweight="bold", color="#C46A10")
    ax.set_xlim(0, 1)
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("number of parts")
    ax.set_title(f"Cost-based tiers: pass < {lo:.2f}  ·  review  ·  act ≥ {hi:.2f}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="center left")
    plt.tight_layout()
    fig.savefig("outputs/tiers.png", dpi=120)
    print("Saved outputs/tiers.png")

    print("\nNote: thresholds were tuned on the same 115 test images they are scored on,")
    print("so these numbers are optimistic. A real deployment would tune on a separate set.")


if __name__ == "__main__":
    main()
