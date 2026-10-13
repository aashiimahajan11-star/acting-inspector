from datetime import timedelta

import pandas as pd

from actions import Plant, now_utc
from config import PAUSE_SECONDS
from decision import DecisionEngine

scores = pd.read_csv("outputs/scores.csv").set_index("part_id")
plant = Plant(reset=True)
engine = DecisionEngine()

demo_parts = [
    scores[scores.is_defect == 0].score.idxmin(),  # most clearly good part
    "color_004",                                   # the borderline defect
    "flip_000",                                    # the clearest defect
]

for part_id in demo_parts:
    row = scores.loc[part_id]
    d = engine.decide(part_id, float(row.score))
    print(f"\n▶ {part_id}  score {d.score:.2f}  →  {d.tier}")
    print(f"  reason: {d.reason}")
    results = plant.execute(d, heatmap_path=row.heatmap_path)
    if not results:
        print("  (no action: part continues down the line)")
    for r in results:
        mark = "✓" if r["ok"] else "✗"
        print(f"  {mark} [{r['type']}] {r['action']}: {r['detail']}")

print("\nPLANT STATE NOW")
for k, v in plant.summary().items():
    print(f"  {k:20s} {v}")

later = now_utc() + timedelta(seconds=PAUSE_SECONDS + 1)
print(f"\n…{PAUSE_SECONDS + 1} seconds later, line status: {plant.line_status(now=later)}")

pending = [p for p in plant.state["pending_approvals"] if p["status"] == "awaiting approval"]
if pending:
    print(f"\nA supervisor reviews {pending[0]['id']} and approves it:")
    print("  " + plant.approve(pending[0]["id"], approver="Aashi"))

print("\nFINAL PLANT STATE")
for k, v in plant.summary().items():
    print(f"  {k:20s} {v}")
print("\nFull state saved in outputs/plant_state.json")
