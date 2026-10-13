import pandas as pd

import actions
from actions import Plant
from audit import AuditLog
from decision import DecisionEngine
from pipeline import process_part, approve, reject

actions.send_alert = lambda *a, **k: False   # batch run: don't send 91 phone alerts

SHIFT = "demo_shift_1"
scores = pd.read_csv("outputs/scores.csv").sample(frac=1, random_state=7)  # line order
engine, plant, audit = DecisionEngine(), Plant(reset=True), AuditLog(reset=True)

for r in scores.itertuples():
    process_part(engine, plant, audit, r.part_id, r.score, r.image_path, r.heatmap_path, SHIFT)
print(f"Processed {len(scores)} parts. Audit log: outputs/audit.db\n")

print("DECISIONS BY TIER")
for row in audit.query("SELECT tier, COUNT(*) AS n FROM decisions GROUP BY tier ORDER BY n DESC"):
    print(f"  {row['tier']:7s} {row['n']}")

print("\nACTIONS LOGGED")
for row in audit.query("""SELECT action, type, COUNT(*) AS n, SUM(ok) AS succeeded
                          FROM actions GROUP BY action, type ORDER BY n DESC"""):
    print(f"  {row['action']:17s} {row['type']:19s} {row['n']:3d}  ({row['succeeded']} succeeded)")

pending = [p for p in plant.state["pending_approvals"] if p["status"] == "awaiting approval"]
print(f"\n{len(pending)} quarantine requests waiting. A supervisor handles three:")
print("  " + approve(plant, audit, pending[0]["id"], "Aashi", note="Defect clearly visible on heatmap"))
print("  " + approve(plant, audit, pending[1]["id"], "Aashi"))
print("  " + reject(plant, audit, pending[2]["id"], "Aashi", note="Isolated defect, rest of batch looks fine"))

print("\nWHY DID THE SYSTEM DO THAT?\n")
for part_id in ["color_004", pending[0]["part_id"], pending[2]["part_id"]]:
    print(audit.explain(part_id))
    print()

print("BORDERLINE DECISIONS (within 0.03 of a threshold)")
for row in audit.query("""SELECT part_id, ROUND(score, 3) AS score, tier, ROUND(margin, 3) AS margin
                          FROM decisions WHERE margin < 0.03 ORDER BY margin"""):
    print(f"  {row['part_id']:12s} score {row['score']:.3f}  {row['tier']:6s}  margin {row['margin']:.3f}")
