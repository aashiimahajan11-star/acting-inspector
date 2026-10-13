"""Step 11: measure the humans, re-tune the thresholds, propose the change."""
import argparse
import json
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from audit import AuditLog
from config import COSTS, DEFECT_RATE, HUMAN_ACCURACY
from tune_thresholds import GRID

THRESHOLDS = Path("outputs/thresholds.json")
PROPOSAL = Path("outputs/thresholds_proposed.json")

COST_GOOD_SCRAPPED = 5          # a good part wrongly scrapped by a reviewer
REVIEW_CAPACITY_PER_1000 = 125  # one inspector: ~240 reviews per 1,920-part shift
PRIOR_WEIGHT = 10               # the starting assumption counts as 10 reviews
PRIOR_CATCH = 0.90              # assumed: reviewers catch 90% of defects
PRIOR_FALSE_SCRAP = 0.05        # assumed: reviewers scrap 5% of good parts
MIN_VERDICTS = 20               # fewer than this = low evidence
MAX_STEP = 0.05                 # max threshold change per update


def _secs(note):
    m = re.search(r"took (\d+)s", note or "")
    return float(m.group(1)) if m else np.nan


def load_verdicts():
    db = sqlite3.connect("outputs/audit.db")
    ev = pd.read_sql(
        "SELECT part_id, event_type, actor, verdict, note FROM human_events "
        "WHERE event_type IN ('review_verdict', 'override')", db)
    truth = pd.read_csv("outputs/scores.csv").set_index("part_id")
    ev["is_defect"] = ev.part_id.map(truth.is_defect)
    ev["said_defect"] = (ev.verdict == "defective").astype(int)
    ev["correct"] = ev.said_defect == ev.is_defect
    ev["secs"] = ev.note.map(_secs)
    return ev


def measure_humans(ev):
    rv = ev[ev.event_type == "review_verdict"]
    d, g = rv[rv.is_defect == 1], rv[rv.is_defect == 0]
    caught, scrapped = int(d.said_defect.sum()), int(g.said_defect.sum())
    return {
        "n": len(rv), "n_defects": len(d), "caught": caught,
        "n_good": len(g), "scrapped": scrapped,
        "raw_accuracy": float(rv.correct.mean()) if len(rv) else float("nan"),
        "median_secs": float(rv.secs.median()),
        # smoothed: blend measurements with the starting assumption
        "catch_rate": (caught + PRIOR_CATCH * PRIOR_WEIGHT) / (len(d) + PRIOR_WEIGHT),
        "false_scrap_rate": (scrapped + PRIOR_FALSE_SCRAP * PRIOR_WEIGHT) / (len(g) + PRIOR_WEIGHT),
    }


def evaluate(d, g, lo, hi, catch, false_scrap, p=DEFECT_RATE):
    d_pass = np.mean(d < lo)
    d_rev = np.mean((d >= lo) & (d < hi))
    d_act = np.mean(d >= hi)
    g_rev = np.mean((g >= lo) & (g < hi))
    g_act = np.mean(g >= hi)
    cost_d = (d_pass * COSTS["miss"]
              + d_rev * (COSTS["review"] + (1 - catch) * COSTS["miss"])
              + d_act * COSTS["catch"])
    cost_g = (g_rev * (COSTS["review"] + false_scrap * COST_GOOD_SCRAPPED)
              + g_act * COSTS["false_stop"])
    n_d, n_g = 1000 * p, 1000 * (1 - p)
    return {"t_low": float(lo), "t_high": float(hi),
            "cost": n_d * cost_d + n_g * cost_g,
            "missed": n_d * (d_pass + d_rev * (1 - catch)),
            "false_stops": n_g * g_act,
            "reviews": n_d * d_rev + n_g * g_rev}


def search(d, g, catch, false_scrap, capacity=None):
    best, best_key = None, None
    for lo in GRID:
        for hi in GRID[GRID >= lo]:
            r = evaluate(d, g, lo, hi, catch, false_scrap)
            if capacity is not None and r["reviews"] > capacity:
                continue
            key = (round(r["cost"], 4), round(r["reviews"], 4), lo, -hi)
            if best_key is None or key < best_key:
                best, best_key = r, key
    return best


def apply_proposal(approver):
    if not approver:
        print("Refused: --approver is required. Threshold changes need a named person.")
        return
    prop = json.loads(PROPOSAL.read_text())
    cur = json.loads(THRESHOLDS.read_text())
    version = cur.get("version", 1)
    backup = THRESHOLDS.with_name(f"thresholds_v{version}.json")
    shutil.copy(THRESHOLDS, backup)
    new = dict(cur)
    new.update(t_low=prop["proposed"]["t_low"], t_high=prop["proposed"]["t_high"],
               version=version + 1, approved_by=approver,
               approved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               human_catch_rate=prop["humans"]["catch_rate"], previous_version=str(backup))
    THRESHOLDS.write_text(json.dumps(new, indent=2))
    AuditLog().record_human_event(
        "threshold_change", approver, reference=f"v{version} -> v{version + 1}",
        verdict="approved",
        note=(f"t_low {cur['t_low']:.3f} -> {new['t_low']:.3f}, "
              f"t_high {cur['t_high']:.3f} -> {new['t_high']:.3f}"))
    print(f"Applied thresholds v{version + 1}, approved by {approver}.")
    print(f"  t_low  {cur['t_low']:.3f} -> {new['t_low']:.3f}")
    print(f"  t_high {cur['t_high']:.3f} -> {new['t_high']:.3f}")
    print(f"Backup of v{version}: {backup}")
    print(f"To undo:  cp {backup} {THRESHOLDS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--approver")
    args = ap.parse_args()
    if args.apply:
        return apply_proposal(args.approver)

    ev = load_verdicts()
    if ev.empty:
        print("No human verdicts yet: review some parts in the app first (Step 10).")
        return
    h = measure_humans(ev)

    print("1. HOW GOOD ARE THE HUMANS?  (checked against the true labels)")
    print(f"  Review verdicts:          {h['n']}")
    print(f"  Raw accuracy:             {h['raw_accuracy']:.0%}")
    print(f"  Defects caught:           {h['caught']} of {h['n_defects']}")
    print(f"  Good parts scrapped:      {h['scrapped']} of {h['n_good']}")
    print(f"  Median review time:       {h['median_secs']:.0f} s")
    print(f"  Smoothed catch rate:      {h['catch_rate']:.0%}   (assumed in Step 5: {HUMAN_ACCURACY:.0%})")
    print(f"  Smoothed false-scrap rate:{h['false_scrap_rate']:.0%}   (assumed in Step 5: 0%)")

    ov = ev[ev.event_type == "override"]
    if len(ov):
        right = int((ov.is_defect == 0).sum())
        print(f"\n  Overrides of ACT ('part is actually good'): {len(ov)}, "
              f"of which {right} were really good parts")

    cur = json.loads(THRESHOLDS.read_text())
    scores = pd.read_csv("outputs/scores.csv")
    d = scores[scores.is_defect == 1].score.values
    g = scores[scores.is_defect == 0].score.values
    lo, hi = cur["t_low"], cur["t_high"]
    cr, fs = h["catch_rate"], h["false_scrap_rate"]

    policies = {
        "A. Current tiers, ASSUMED humans": evaluate(d, g, lo, hi, HUMAN_ACCURACY, 0.0),
        "B. Current tiers, MEASURED humans": evaluate(d, g, lo, hi, cr, fs),
        "C. Re-tuned for measured humans": search(d, g, cr, fs),
        "D. Re-tuned + review capacity": search(d, g, cr, fs, capacity=REVIEW_CAPACITY_PER_1000),
    }
    target = policies["D. Re-tuned + review capacity"] or policies["C. Re-tuned for measured humans"]

    # Guardrail: limit how far a threshold can move in one update
    new_lo = float(np.clip(target["t_low"], lo - MAX_STEP, lo + MAX_STEP))
    new_hi = float(np.clip(target["t_high"], hi - MAX_STEP, hi + MAX_STEP))
    new_hi = max(new_hi, new_lo)
    policies["E. PROPOSED (max-step guardrail)"] = evaluate(d, g, new_lo, new_hi, cr, fs)

    table = pd.DataFrame({k: v for k, v in policies.items() if v}).T[
        ["t_low", "t_high", "cost", "missed", "false_stops", "reviews"]]
    pd.options.display.float_format = "{:,.2f}".format
    print(f"\n2. POLICIES, PER 1,000 PARTS  (review capacity {REVIEW_CAPACITY_PER_1000})")
    print(table.to_string())

    low_evidence = h["n"] < MIN_VERDICTS
    proposal = {
        "status": "awaiting approval",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "current": {"t_low": lo, "t_high": hi},
        "proposed": {"t_low": new_lo, "t_high": new_hi},
        "expected": policies["E. PROPOSED (max-step guardrail)"],
        "humans": h,
        "low_evidence": low_evidence,
    }
    PROPOSAL.write_text(json.dumps(proposal, indent=2, default=float))

    print("\n3. PROPOSAL")
    print(f"  t_low  {lo:.3f} -> {new_lo:.3f}")
    print(f"  t_high {hi:.3f} -> {new_hi:.3f}")
    if low_evidence:
        print(f"  ⚠ LOW EVIDENCE: only {h['n']} verdicts (want at least {MIN_VERDICTS}).")
    print(f"  Saved to {PROPOSAL}. NOT applied.")
    print("  To apply:  python src/feedback_loop.py --apply --approver Aashi")


if __name__ == "__main__":
    main()
