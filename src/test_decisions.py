import math
import pandas as pd
from decision import DecisionEngine

# --- 1. Rule checks on hand-picked scores ---
engine = DecisionEngine()
print(f"Thresholds loaded: PASS < {engine.t_low}  |  REVIEW  |  ACT >= {engine.t_high}\n")

checks = [
    ("clearly good",   0.20,          "PASS"),
    ("borderline",     0.45,          "REVIEW"),
    ("clear defect",   0.90,          "ACT"),
    ("missing score",  None,          "REVIEW"),
    ("broken score",   float("nan"),  "REVIEW"),
    ("out of range",   7.0,           "REVIEW"),
]
for name, score, expected in checks:
    d = engine.decide(name, score)
    status = "✓" if d.tier == expected else "✗ FAILED"
    print(f"  {status}  {name:14s} score={score!s:5s} -> {d.tier:6s} ({d.reason})")
    assert d.tier == expected
assert "quarantine_batch" not in engine.decide("x", 0.95).auto_actions
print("\nAll rule checks passed. Quarantine always needs approval.\n")

# --- 2. Review capacity: a tiny queue overflows safely ---
small = DecisionEngine(review_capacity=2)
for i in range(3):
    d = small.decide(f"borderline_{i}", 0.45)
    print(f"  queue test {i + 1}: {d.auto_actions}  ({d.reason})")
print()

# --- 3. All 115 test parts ---
df = pd.read_csv("outputs/scores.csv").sample(frac=1, random_state=42)  # shuffle like a real line
engine = DecisionEngine()
decisions = [engine.decide(r.part_id, r.score).to_dict() for r in df.itertuples()]
out = pd.DataFrame(decisions).merge(df[["part_id", "defect_type", "is_defect"]], on="part_id")
out.to_csv("outputs/decisions.csv", index=False)

out["truth"] = out.is_defect.map({0: "good", 1: "defective"})
print("DECISIONS vs TRUTH (115 test parts)")
print(pd.crosstab(out.truth, out.tier, margins=True, margins_name="total").to_string())

print("\nParts sent to REVIEW (the hard cases):")
rev = out[out.tier == "REVIEW"].sort_values("score")
print(rev[["part_id", "score", "truth", "margin"]].to_string(index=False))

print("\nExample ACT decision:")
ex = out[out.tier == "ACT"].iloc[0]
print(f"  part:           {ex.part_id}  (score {ex.score:.2f})")
print(f"  reason:         {ex.reason}")
print(f"  auto actions:   {ex.auto_actions}")
print(f"  needs approval: {ex.needs_approval}")

print("\nSaved outputs/decisions.csv")
