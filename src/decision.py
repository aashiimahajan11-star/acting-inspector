"""Decision layer: turns an anomaly score into PASS / REVIEW / ACT."""
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

THRESHOLDS_FILE = Path("outputs/thresholds.json")
MODEL_VERSION = "patchcore_metal_nut_v1"

# Every action the system can take, and whether it can be undone.
# Reversible actions may run automatically; irreversible ones need a person.
ACTIONS = {
    "pause_line":       {"reversible": True,  "description": "Pause the line for 30 seconds"},
    "hold_part":        {"reversible": True,  "description": "Divert the part to the hold bin"},
    "open_ticket":      {"reversible": True,  "description": "Open a maintenance ticket with the heatmap"},
    "alert_operator":   {"reversible": True,  "description": "Message the on-call operator"},
    "add_to_review":    {"reversible": True,  "description": "Add the part to the human review queue"},
    "quarantine_batch": {"reversible": False, "description": "Quarantine all parts since the last good check"},
}


@dataclass
class Decision:
    part_id: str
    score: float
    tier: str              # PASS | REVIEW | ACT
    reason: str
    auto_actions: list     # run immediately (all reversible)
    needs_approval: list   # wait for a person (irreversible)
    margin: float          # distance to the nearest threshold (small = borderline)
    model_version: str
    t_low: float
    t_high: float
    decided_at: str

    def to_dict(self):
        return asdict(self)


class DecisionEngine:
    def __init__(self, thresholds_file=THRESHOLDS_FILE, review_capacity=50):
        t = json.loads(Path(thresholds_file).read_text())
        self.t_low = t["t_low"]
        self.t_high = t["t_high"]
        self.review_capacity = review_capacity
        self.review_queue = 0

        # Safety check: every automatic action must be reversible
        for name, a in ACTIONS.items():
            if name == "quarantine_batch":
                assert not a["reversible"]

    def _make(self, part_id, score, tier, reason, auto, approval):
        valid = isinstance(score, (int, float)) and not math.isnan(score)
        margin = min(abs(score - self.t_low), abs(score - self.t_high)) if valid else 0.0
        # Guardrail: never let an irreversible action run automatically
        for a in auto:
            assert ACTIONS[a]["reversible"], f"{a} is irreversible and cannot run automatically"
        return Decision(
            part_id=part_id,
            score=float(score) if valid else float("nan"),
            tier=tier, reason=reason,
            auto_actions=auto, needs_approval=approval,
            margin=round(margin, 4),
            model_version=MODEL_VERSION,
            t_low=self.t_low, t_high=self.t_high,
            decided_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def decide(self, part_id, score):
        # 1. Fail safe: a missing or invalid score is never a PASS
        if score is None or not isinstance(score, (int, float)) or math.isnan(score) \
                or score < 0 or score > 1.0001:
            return self._make(part_id, score if isinstance(score, (int, float)) else float("nan"),
                              "REVIEW", "Invalid or missing score: failing safe",
                              ["hold_part", "add_to_review"], [])

        # 2. High confidence of a defect: act now, ask before anything irreversible
        if score >= self.t_high:
            return self._make(part_id, score, "ACT",
                              f"Score {score:.2f} ≥ act threshold {self.t_high:.2f}",
                              ["pause_line", "hold_part", "open_ticket", "alert_operator"],
                              ["quarantine_batch"])

        # 3. Uncertain: a person decides
        if score >= self.t_low:
            if self.review_queue < self.review_capacity:
                self.review_queue += 1
                return self._make(part_id, score, "REVIEW",
                                  f"Score {score:.2f} is between {self.t_low:.2f} and {self.t_high:.2f}: uncertain",
                                  ["hold_part", "add_to_review"], [])
            return self._make(part_id, score, "REVIEW",
                              f"Uncertain score {score:.2f} but review queue is full "
                              f"({self.review_capacity}): holding part, alerting supervisor",
                              ["hold_part", "alert_operator"], [])

        # 4. Confidently normal
        return self._make(part_id, score, "PASS",
                          f"Score {score:.2f} < pass threshold {self.t_low:.2f}", [], [])

    def review_done(self):
        """Call when a person finishes a review, to free a queue slot."""
        self.review_queue = max(0, self.review_queue - 1)
