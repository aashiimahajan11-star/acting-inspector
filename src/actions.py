"""Action layer: carries out decisions on a simulated production line."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from config import ALERTS_ENABLED, ALERT_TOPIC, PAUSE_SECONDS

STATE_FILE = Path("outputs/plant_state.json")
NTFY_URL = "https://ntfy.sh"


def now_utc():
    return datetime.now(timezone.utc)


def iso(t):
    return t.isoformat(timespec="seconds")


EMPTY_STATE = {
    "line": {"status": "running", "paused_until": None, "pause_count": 0},
    "hold_bin": [],
    "review_queue": [],
    "tickets": [],
    "pending_approvals": [],
    "quarantined_batches": [],
    "alerts": [],
}


def send_alert(title, message, priority="default", tags="", attachment=None):
    """Push a notification to the phone. Never raises: a failed alert must not stop the line."""
    if not ALERTS_ENABLED or not ALERT_TOPIC:
        print(f"    [alert not sent: alerts disabled] {title}")
        return False
    url = f"{NTFY_URL}/{ALERT_TOPIC}"
    params = {"title": title, "priority": priority, "tags": tags}
    try:
        if attachment and Path(attachment).exists():
            params.update(message=message, filename=Path(attachment).name)
            with open(attachment, "rb") as f:
                r = requests.put(url, data=f, params=params, timeout=10)
        else:
            r = requests.post(url, data=message.encode("utf-8"), params=params, timeout=10)
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"    [alert failed, logged only: {e}]")
        return False


class Plant:
    """A simulated production line that the decision layer can act on."""

    def __init__(self, state_file=STATE_FILE, reset=False):
        self.state_file = Path(state_file)
        if reset or not self.state_file.exists():
            self.state = json.loads(json.dumps(EMPTY_STATE))
            self.save()
        else:
            self.state = json.loads(self.state_file.read_text())

    def save(self):
        self.state_file.write_text(json.dumps(self.state, indent=2))

    # ---------- line status (auto-resumes after a pause) ----------
    def line_status(self, now=None):
        now = now or now_utc()
        line = self.state["line"]
        if line["status"] == "paused" and now >= datetime.fromisoformat(line["paused_until"]):
            line["status"], line["paused_until"] = "running", None
            self.save()
        return line["status"]

    # ---------- automatic actions (all reversible) ----------
    def pause_line(self, d, **_):
        until = now_utc() + timedelta(seconds=PAUSE_SECONDS)
        line = self.state["line"]
        line["status"], line["paused_until"] = "paused", iso(until)
        line["pause_count"] += 1
        return f"Line paused until {until:%H:%M:%S} UTC (resumes automatically)"

    def hold_part(self, d, **_):
        if d.part_id not in self.state["hold_bin"]:
            self.state["hold_bin"].append(d.part_id)
        return f"{d.part_id} moved to hold bin ({len(self.state['hold_bin'])} held)"

    def add_to_review(self, d, heatmap_path=None, **_):
        self.state["review_queue"].append({
            "part_id": d.part_id, "score": round(d.score, 4), "reason": d.reason,
            "heatmap_path": heatmap_path, "added_at": iso(now_utc()),
        })
        return f"{d.part_id} added to review queue ({len(self.state['review_queue'])} waiting)"

    def open_ticket(self, d, heatmap_path=None, **_):
        tid = f"TCK-{len(self.state['tickets']) + 1:04d}"
        self.state["tickets"].append({
            "id": tid, "part_id": d.part_id, "score": round(d.score, 4),
            "reason": d.reason, "heatmap_path": heatmap_path,
            "status": "open", "created_at": iso(now_utc()),
        })
        return f"Ticket {tid} opened"

    def alert_operator(self, d, heatmap_path=None, **_):
        title = f"{d.tier}: {d.part_id} (score {d.score:.2f})"
        body = [d.reason.replace("≥", ">=")]
        if d.tier == "ACT":
            body.append(f"Line paused for {PAUSE_SECONDS}s. Part held. Ticket opened.")
        if d.needs_approval:
            body.append("Needs your approval: " + ", ".join(d.needs_approval))
        priority, tags = ("high", "rotating_light") if d.tier == "ACT" else ("default", "warning")
        delivered = send_alert(title, "\n".join(body), priority, tags, heatmap_path)
        self.state["alerts"].append({"title": title, "delivered": delivered, "sent_at": iso(now_utc())})
        if not delivered:
            raise RuntimeError("alert NOT delivered")
        return "Alert delivered to phone"

    # ---------- irreversible actions: only via human approval ----------
    def request_approval(self, d, action):
        pid = f"APR-{len(self.state['pending_approvals']) + 1:04d}"
        self.state["pending_approvals"].append({
            "id": pid, "action": action, "part_id": d.part_id, "score": round(d.score, 4),
            "status": "awaiting approval", "requested_at": iso(now_utc()),
        })
        return f"{action} is waiting for human approval ({pid})"

    def approve(self, approval_id, approver):
        for p in self.state["pending_approvals"]:
            if p["id"] == approval_id:
                if p["status"] != "awaiting approval":
                    return f"{approval_id} already {p['status']}"
                p.update(status="approved", decided_by=approver, decided_at=iso(now_utc()))
                if p["action"] == "quarantine_batch":
                    self.state["quarantined_batches"].append({
                        "approval_id": approval_id, "trigger_part": p["part_id"],
                        "approved_by": approver, "at": iso(now_utc()),
                    })
                self.save()
                return f"{p['action']} approved by {approver} and carried out"
        raise KeyError(f"No approval request {approval_id}")

    def reject(self, approval_id, approver):
        for p in self.state["pending_approvals"]:
            if p["id"] == approval_id and p["status"] == "awaiting approval":
                p.update(status="rejected", decided_by=approver, decided_at=iso(now_utc()))
                self.save()
                return f"{p['action']} rejected by {approver}"
        raise KeyError(f"No open approval request {approval_id}")

    def release_part(self, part_id):
        """Undo a hold (e.g. after a reviewer confirms the part is good)."""
        if part_id in self.state["hold_bin"]:
            self.state["hold_bin"].remove(part_id)
            self.save()
            return f"{part_id} released from hold bin"
        return f"{part_id} was not in the hold bin"

    # ---------- run a decision ----------
    def execute(self, decision, heatmap_path=None):
        results = []
        for action in decision.auto_actions:
            try:
                detail, ok = getattr(self, action)(decision, heatmap_path=heatmap_path), True
            except Exception as e:  # one failed action must not block the others
                detail, ok = f"FAILED: {e}", False
            results.append({"action": action, "type": "auto", "ok": ok, "detail": detail})
        for action in decision.needs_approval:
            results.append({"action": action, "type": "approval_requested", "ok": True,
                            "detail": self.request_approval(decision, action)})
        self.save()
        return results

    def summary(self):
        s = self.state
        return {
            "line": self.line_status(),
            "pauses": s["line"]["pause_count"],
            "hold_bin": len(s["hold_bin"]),
            "review_queue": len(s["review_queue"]),
            "tickets": len(s["tickets"]),
            "awaiting_approval": sum(p["status"] == "awaiting approval" for p in s["pending_approvals"]),
            "quarantined_batches": len(s["quarantined_batches"]),
            "alerts_delivered": sum(a["delivered"] for a in s["alerts"]),
        }
