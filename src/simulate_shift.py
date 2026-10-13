"""Simulate a night shift: dashboard-only factory vs acting factory."""
import json
from collections import deque
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import COSTS, HUMAN_ACCURACY, PAUSE_SECONDS
from decision import DecisionEngine

# ---------------- Shift settings (illustrative assumptions) ----------------
SHIFT_START = "23:00"
SHIFT_HOURS = 8
CYCLE_S = 15                 # one part every 15 s -> up to 1,920 parts per shift
BASE_DEFECT_RATE = 0.01      # normal background defect rate
FAULT_DEFECT_RATE = 0.70     # cracked die: most parts come out defective
FAULT_AT_MIN = 240           # story shift: fault starts at 03:00
ESCALATE_N = 3               # 3 ACT decisions ...
ESCALATE_WINDOW_MIN = 10     # ... within 10 minutes -> stop line, call engineer
RESPONSE_MIN = 20            # engineer arrives and fixes the fault after 20 min
REVIEW_CAPACITY = 240        # one inspector, ~2 min per part, 8 hours
SCORE_NOISE = 0.02           # new parts are never identical to the test images
DASHBOARD_THRESHOLD = 0.5    # dashboard world flags at anomalib's default
COST_DOWNTIME_PER_MIN = 50   # lost output while the line is stopped
COST_SCRAP = 3               # material + machine time per defective part produced
COST_PULL = 15               # finding a flagged part in packed boxes next morning

N_SLOTS = SHIFT_HOURS * 3600 // CYCLE_S
END_S = N_SLOTS * CYCLE_S


def clock(t_s):
    start = datetime.strptime(SHIFT_START, "%H:%M")
    return (start + timedelta(seconds=float(t_s))).strftime("%H:%M:%S")


def make_parts(seed, good_scores, defect_scores):
    """Pre-draw every random value, so both factories see the same parts."""
    rng = np.random.default_rng(seed)
    return {
        "u": rng.random(N_SLOTS),
        "good": rng.choice(good_scores, N_SLOTS),
        "defect": rng.choice(defect_scores, N_SLOTS),
        "noise": rng.normal(0, SCORE_NOISE, N_SLOTS),
        "review_u": rng.random(N_SLOTS),
    }


def part_at(parts, k, fault_active):
    p = FAULT_DEFECT_RATE if fault_active else BASE_DEFECT_RATE
    is_def = bool(parts["u"][k] < p)
    base = parts["defect"][k] if is_def else parts["good"][k]
    return is_def, float(np.clip(base + parts["noise"][k], 0, 1))


def new_result():
    return dict(produced=0, defects=0, shipped=0, flagged=0, downtime_min=0.0,
                reviews=0, pauses=0, escalations=0, false_stops=0,
                contained_at=None, timeline=[], events=[])


def costs(r):
    c = {
        "shipped defects": r["shipped"] * COSTS["miss"],
        "scrap": r["defects"] * COST_SCRAP,
        "downtime": r["downtime_min"] * COST_DOWNTIME_PER_MIN,
        "reviews": r["reviews"] * COSTS["review"],
        "pulling flagged parts": r["flagged"] * COST_PULL,
    }
    c["TOTAL"] = sum(c.values())
    return c


def run_dashboard(parts, fault_at_s):
    """Model flags parts, but nobody acts until the 07:00 handover."""
    r = new_result()
    for k in range(N_SLOTS):
        t = k * CYCLE_S
        is_def, score = part_at(parts, k, t >= fault_at_s)
        r["produced"] += 1
        r["defects"] += is_def
        if score >= DASHBOARD_THRESHOLD:
            r["flagged"] += 1          # pulled from the boxes next morning
        elif is_def:
            r["shipped"] += 1          # never flagged: goes to the customer
        r["timeline"].append((t, r["defects"]))
    return r


def run_acting(parts, fault_at_s, engine):
    """The decision layer acts in real time, with escalation."""
    r = new_result()
    fault_fixed, stopped_until, recent = False, 0.0, deque()
    first_act_logged = False
    t = 0.0
    while t < END_S:
        if t < stopped_until:                       # line is stopped
            stop_end = min(stopped_until, END_S)
            r["downtime_min"] += (stop_end - t) / 60
            t = stop_end
            continue

        k = int(t // CYCLE_S)
        fault_active = t >= fault_at_s and not fault_fixed
        is_def, score = part_at(parts, k, fault_active)
        r["produced"] += 1
        r["defects"] += is_def
        d = engine.decide(f"part_{k}", score)

        if d.tier == "ACT":
            if not is_def:
                r["false_stops"] += 1
            recent.append(t)
            while recent and t - recent[0] > ESCALATE_WINDOW_MIN * 60:
                recent.popleft()
            if fault_active and not first_act_logged:
                r["events"].append(f"{clock(t)}  First ACT (score {score:.2f}): line paused "
                                   f"{PAUSE_SECONDS}s, part held, operator alerted")
                first_act_logged = True
            if len(recent) >= ESCALATE_N:
                r["escalations"] += 1
                stopped_until = t + RESPONSE_MIN * 60
                r["events"].append(f"{clock(t)}  {ESCALATE_N} ACTs in {ESCALATE_WINDOW_MIN} min: "
                                   f"ESCALATION, line stopped, engineer called")
                if fault_active:
                    fault_fixed = True
                    r["contained_at"] = t
                    r["events"].append(f"{clock(stopped_until)}  Engineer replaces the die, line restarts")
                else:
                    r["events"].append(f"{clock(stopped_until)}  Engineer finds no fault, line restarts")
                recent.clear()
            else:
                r["pauses"] += 1
                stopped_until = t + PAUSE_SECONDS
        elif d.tier == "REVIEW":
            r["reviews"] += 1
            if is_def and parts["review_u"][k] > HUMAN_ACCURACY:
                r["shipped"] += 1              # the inspector missed it
        elif is_def:
            r["shipped"] += 1                  # passed, but was defective

        r["timeline"].append((t, r["defects"]))
        t += CYCLE_S
    return r


def summary_table(dash, act):
    rows = {
        "Parts produced": (dash["produced"], act["produced"]),
        "Defective parts produced": (dash["defects"], act["defects"]),
        "Fault contained at": ("07:00 (next shift)",
                               clock(act["contained_at"]) if act["contained_at"] else "-"),
        "Defects shipped to customer": (dash["shipped"], act["shipped"]),
        "Line downtime (min)": (round(dash["downtime_min"], 1), round(act["downtime_min"], 1)),
        "Automatic 30s pauses": (0, act["pauses"]),
        "Escalations (engineer called)": (0, act["escalations"]),
        "False stops (good part paused line)": (0, act["false_stops"]),
        "Parts sent to human review": (0, act["reviews"]),
        "Review backlog at 07:00": (0, max(0, act["reviews"] - REVIEW_CAPACITY)),
    }
    cd, ca = costs(dash), costs(act)
    for k in cd:
        rows[f"Cost: {k} (€)"] = (f"{cd[k]:,.0f}", f"{ca[k]:,.0f}")
    return pd.DataFrame(rows, index=["Dashboard only", "Acting system"]).T


def main():
    df = pd.read_csv("outputs/scores.csv")
    good = df[df.is_defect == 0].score.values
    bad = df[df.is_defect == 1].score.values
    engine = DecisionEngine(review_capacity=10**9)   # capacity modelled separately here

    # ---------------- 1. The story shift ----------------
    fault_s = FAULT_AT_MIN * 60
    parts = make_parts(seed=1, good_scores=good, defect_scores=bad)
    dash = run_dashboard(parts, fault_s)
    act = run_acting(parts, fault_s, engine)

    print(f"NIGHT SHIFT {SHIFT_START}–07:00   (die cracks at {clock(fault_s)})\n")
    print("What the acting system did:")
    print(f"  {clock(fault_s)}  Die cracks: 70% of parts now defective")
    for e in act["events"]:
        print(f"  {e}")
    print(f"\n  Dashboard-only factory: the first red box appears at about {clock(fault_s)},")
    print("  but nobody looks until the 07:00 handover.\n")
    print(summary_table(dash, act).to_string())

    # ---------------- 2. Fifty shifts, fault at random times ----------------
    rng = np.random.default_rng(123)
    rows = []
    for i in range(50):
        fs = float(rng.uniform(30, 450)) * 60
        p = make_parts(seed=1000 + i, good_scores=good, defect_scores=bad)
        d_r, a_r = run_dashboard(p, fs), run_acting(p, fs, engine)
        rows.append({"dash_cost": costs(d_r)["TOTAL"], "act_cost": costs(a_r)["TOTAL"],
                     "dash_shipped": d_r["shipped"], "act_shipped": a_r["shipped"],
                     "dash_defects": d_r["defects"], "act_defects": a_r["defects"],
                     "act_reviews": a_r["reviews"]})
    m = pd.DataFrame(rows)
    saving = 1 - m.act_cost.mean() / m.dash_cost.mean()
    print("\n50 SHIFTS, FAULT AT A RANDOM TIME   (average, with range)")
    for label, col in [("Total cost (€)", "cost"), ("Defects shipped", "shipped"),
                       ("Defective parts produced", "defects")]:
        d_col, a_col = m[f"dash_{col}"], m[f"act_{col}"]
        print(f"  {label:26s} dashboard {d_col.mean():>9,.0f} ({d_col.min():,.0f}–{d_col.max():,.0f})"
              f"   acting {a_col.mean():>7,.0f} ({a_col.min():,.0f}–{a_col.max():,.0f})")
    print(f"  {'Parts sent to review':26s} acting {m.act_reviews.mean():,.0f} per shift "
          f"(capacity {REVIEW_CAPACITY})")
    print(f"\n  Acting system cost: {saving:.0%} lower on average")

    # ---------------- 3. Chart: the 3 a.m. story ----------------
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for r, label, color in [(dash, "Dashboard only", "#8A8A8A"), (act, "Acting system", "#2F6FDB")]:
        ts, ys = zip(*r["timeline"])
        ax.plot(np.array(ts) / 3600, ys, color=color, linewidth=2.4, label=label)
    ax.axvline(fault_s / 3600, color="#C0392B", linestyle="--", linewidth=1)
    ax.text(fault_s / 3600 + 0.08, ax.get_ylim()[1] * 0.9, "die cracks\n03:00", color="#C0392B", fontsize=9)
    ax.text(SHIFT_HOURS - 0.05, dash["defects"], f"{dash['defects']} defective parts",
            ha="right", va="bottom", fontsize=9, color="#555555")
    ax.text(SHIFT_HOURS - 0.05, act["defects"], f"{act['defects']}",
            ha="right", va="bottom", fontsize=9, color="#2F6FDB")
    ax.set_xticks(range(SHIFT_HOURS + 1))
    ax.set_xticklabels([clock(h * 3600)[:5] for h in range(SHIFT_HOURS + 1)])
    ax.set_xlim(0, SHIFT_HOURS)
    ax.set_xlabel("time")
    ax.set_ylabel("defective parts produced (cumulative)")
    ax.set_title("One night shift: the same model, with and without the ability to act")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    fig.savefig("outputs/shift_timeline.png", dpi=130)

    out = {"story_shift": {"dashboard": costs(dash), "acting": costs(act)},
           "fifty_shifts": {"dash_cost_mean": float(m.dash_cost.mean()),
                            "act_cost_mean": float(m.act_cost.mean()),
                            "saving": float(saving)}}
    with open("outputs/shift_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nSaved outputs/shift_timeline.png and outputs/shift_results.json")
    print("All settings are illustrative assumptions (top of simulate_shift.py).")


if __name__ == "__main__":
    main()
