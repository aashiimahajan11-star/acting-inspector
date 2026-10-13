"""Audit trail: an append-only record of every decision, action and human input."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("outputs/audit.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    decided_at    TEXT NOT NULL,
    shift_id      TEXT,
    part_id       TEXT NOT NULL,
    image_path    TEXT,
    heatmap_path  TEXT,
    score         REAL,
    tier          TEXT NOT NULL,
    reason        TEXT,
    margin        REAL,
    model_version TEXT,
    t_low         REAL,
    t_high        REAL
);
CREATE TABLE IF NOT EXISTS actions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id  INTEGER NOT NULL REFERENCES decisions(id),
    action       TEXT NOT NULL,
    type         TEXT NOT NULL,
    ok           INTEGER NOT NULL,
    detail       TEXT,
    executed_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS human_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id  INTEGER REFERENCES decisions(id),
    part_id      TEXT,
    event_type   TEXT NOT NULL,
    reference    TEXT,
    actor        TEXT NOT NULL,
    verdict      TEXT,
    note         TEXT,
    at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_part ON decisions(part_id);
CREATE INDEX IF NOT EXISTS idx_human_part ON human_events(part_id);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AuditLog:
    """Append-only: there are deliberately no update or delete methods."""

    def __init__(self, db_path=DB_PATH, reset=False):
        db_path = Path(db_path)
        if reset and db_path.exists():
            db_path.unlink()
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ---------- writing ----------
    def record_decision(self, d, image_path=None, heatmap_path=None, shift_id=None):
        cur = self.conn.execute(
            """INSERT INTO decisions (decided_at, shift_id, part_id, image_path, heatmap_path,
                   score, tier, reason, margin, model_version, t_low, t_high)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (d.decided_at, shift_id, d.part_id, image_path, heatmap_path, d.score, d.tier,
             d.reason, d.margin, d.model_version, d.t_low, d.t_high))
        self.conn.commit()
        return cur.lastrowid

    def record_actions(self, decision_id, results):
        self.conn.executemany(
            "INSERT INTO actions (decision_id, action, type, ok, detail, executed_at) VALUES (?, ?, ?, ?, ?, ?)",
            [(decision_id, r["action"], r["type"], int(r["ok"]), r["detail"], now()) for r in results])
        self.conn.commit()

    def record_human_event(self, event_type, actor, part_id=None, decision_id=None,
                           reference=None, verdict=None, note=None):
        self.conn.execute(
            """INSERT INTO human_events (decision_id, part_id, event_type, reference, actor, verdict, note, at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (decision_id, part_id, event_type, reference, actor, verdict, note, now()))
        self.conn.commit()

    # ---------- reading ----------
    def query(self, sql, params=()):
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def latest_decision_id(self, part_id):
        row = self.conn.execute(
            "SELECT id FROM decisions WHERE part_id = ? ORDER BY id DESC LIMIT 1", (part_id,)).fetchone()
        return row["id"] if row else None

    def explain(self, part_id):
        """A plain-language account of everything that happened to one part."""
        d = self.conn.execute(
            "SELECT * FROM decisions WHERE part_id = ? ORDER BY id DESC LIMIT 1", (part_id,)).fetchone()
        if d is None:
            return f"{part_id}: no record"
        acts = self.query("SELECT * FROM actions WHERE decision_id = ? ORDER BY id", (d["id"],))
        humans = self.query("SELECT * FROM human_events WHERE part_id = ? ORDER BY id", (part_id,))

        lines = [
            f"{part_id}",
            f"  decided   {d['decided_at']}  (model {d['model_version']}, shift {d['shift_id']})",
            f"  score     {d['score']:.3f} -> {d['tier']}   "
            f"[pass < {d['t_low']:.3f} | act >= {d['t_high']:.3f}]  margin {d['margin']:.3f}",
            f"  reason    {d['reason']}",
            f"  image     {d['image_path']}",
            f"  heatmap   {d['heatmap_path']}",
        ]
        if acts:
            for i, a in enumerate(acts):
                label = "  actions   " if i == 0 else "            "
                lines.append(f"{label}{'✓' if a['ok'] else '✗'} {a['type']:18s} {a['action']}: {a['detail']}")
        else:
            lines.append("  actions   none")
        if humans:
            for i, h in enumerate(humans):
                label = "  humans    " if i == 0 else "            "
                note = f'  "{h["note"]}"' if h["note"] else ""
                lines.append(f"{label}{h['at']}  {h['actor']} {h['verdict']} {h['reference'] or ''}{note}")
        else:
            lines.append("  humans    none yet")
        return "\n".join(lines)
