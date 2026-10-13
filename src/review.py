"""Human review: verdicts and overrides, written to the plant state and the audit log."""


def _take_from_queue(plant, part_id):
    for item in plant.state["review_queue"]:
        if item["part_id"] == part_id:
            plant.state["review_queue"].remove(item)
            return item
    raise KeyError(f"{part_id} is not in the review queue")


def _unhold(plant, part_id):
    if part_id in plant.state["hold_bin"]:
        plant.state["hold_bin"].remove(part_id)


def record_verdict(plant, audit, part_id, verdict, reviewer, note="", seconds=None):
    """A reviewer decides on a held, uncertain part."""
    assert verdict in ("good", "defective")
    item = _take_from_queue(plant, part_id)
    _unhold(plant, part_id)
    if verdict == "good":
        outcome = "released back to the line"
    else:
        plant.state.setdefault("scrapped", []).append(part_id)
        outcome = "scrapped"
    plant.save()
    if seconds is not None:
        note = f"{note} [took {seconds:.0f}s]".strip()
    audit.record_human_event(
        "review_verdict", reviewer, part_id=part_id,
        decision_id=audit.latest_decision_id(part_id),
        reference=f"score={item['score']:.4f}", verdict=verdict, note=note or None)
    return outcome


def record_override(plant, audit, approval_id, reviewer, note=""):
    """The model said ACT, but a person says the part is good."""
    p = next(p for p in plant.state["pending_approvals"] if p["id"] == approval_id)
    plant.reject(approval_id, reviewer)
    _unhold(plant, p["part_id"])
    plant.save()
    audit.record_human_event(
        "override", reviewer, part_id=p["part_id"],
        decision_id=audit.latest_decision_id(p["part_id"]),
        reference=approval_id, verdict="good",
        note=note or "Model said ACT; reviewer says the part is good")
    return f"{p['part_id']} released; quarantine {approval_id} cancelled"
