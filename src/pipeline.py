"""One function per part: decide -> log decision -> act -> log actions."""


def process_part(engine, plant, audit, part_id, score,
                 image_path=None, heatmap_path=None, shift_id=None):
    d = engine.decide(part_id, score)
    # Log the decision BEFORE acting: if an action fails, the intent is still on record
    decision_id = audit.record_decision(d, image_path, heatmap_path, shift_id)
    results = plant.execute(d, heatmap_path=heatmap_path)
    audit.record_actions(decision_id, results)
    return d, decision_id, results


def _find_request(plant, approval_id):
    return next(p for p in plant.state["pending_approvals"] if p["id"] == approval_id)


def approve(plant, audit, approval_id, approver, note=None):
    p = _find_request(plant, approval_id)
    msg = plant.approve(approval_id, approver)
    audit.record_human_event("approval", approver, part_id=p["part_id"],
                             decision_id=audit.latest_decision_id(p["part_id"]),
                             reference=approval_id, verdict="approved", note=note)
    return msg


def reject(plant, audit, approval_id, approver, note=None):
    p = _find_request(plant, approval_id)
    msg = plant.reject(approval_id, approver)
    audit.record_human_event("approval", approver, part_id=p["part_id"],
                             decision_id=audit.latest_decision_id(p["part_id"]),
                             reference=approval_id, verdict="rejected", note=note)
    return msg
