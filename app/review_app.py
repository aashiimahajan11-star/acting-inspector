import hashlib
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from actions import Plant            # noqa: E402
from audit import AuditLog           # noqa: E402
from pipeline import approve, reject  # noqa: E402
from review import record_verdict, record_override  # noqa: E402

st.set_page_config(page_title="Review station", page_icon="🔍", layout="wide")


def anon(part_id):
    """Neutral ID for reviewers: the real file name contains the true label."""
    if not part_id:
        return part_id
    return "P-" + hashlib.md5(part_id.encode()).hexdigest()[:5].upper()


def load_heatmap(path):
    """Crop off the titles: they contain the true label, which a reviewer must not see."""
    img = Image.open(path)
    w, h = img.size
    return img.crop((0, int(h * 0.12), w, h))


def flash(msg):
    st.session_state.flash = msg


plant = Plant()
audit = AuditLog()
truth = pd.read_csv("outputs/scores.csv").set_index("part_id")["defect_type"].to_dict()

# ---------------- Sidebar ----------------
st.sidebar.header("Reviewer")
reviewer = st.sidebar.text_input("Your name", value="Aashi")
show_truth = st.sidebar.toggle("Demo: show the true label", value=False)
st.sidebar.divider()
st.sidebar.caption("To reset the demo data, run:\n\n`python src/demo_audit.py`")

# ---------------- Header ----------------
st.title("🔍 Inspection review station")
if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

s = plant.summary()
reviewed = audit.query("SELECT COUNT(*) AS n FROM human_events WHERE event_type = 'review_verdict'")[0]["n"]
m = st.columns(5)
m[0].metric("Line", s["line"].upper())
m[1].metric("Parts on hold", s["hold_bin"])
m[2].metric("Waiting for review", s["review_queue"])
m[3].metric("Approvals waiting", s["awaiting_approval"])
m[4].metric("Reviewed so far", reviewed)

tab_review, tab_approvals, tab_history = st.tabs(
    [f"Review queue ({s['review_queue']})", f"Approvals ({s['awaiting_approval']})", "History"])

# ---------------- Tab 1: review queue ----------------
with tab_review:
    queue = plant.state["review_queue"]
    if not queue:
        st.success("The review queue is empty. Nothing is waiting.")
    else:
        item = queue[0]
        pid = item["part_id"]
        if st.session_state.get("shown_part") != pid:
            st.session_state.shown_part = pid
            st.session_state.shown_at = time.time()

        st.subheader(f"Part {anon(pid)}  ·  1 of {len(queue)} waiting")
        left, right = st.columns([3, 2])
        with left:
            st.image(load_heatmap(item["heatmap_path"]),
                     caption="Left: the part. Right: where the model thinks it looks unusual (brighter = more unusual).")
        with right:
            st.metric("Anomaly score", f"{item['score']:.3f}")
            st.write(f"**Why it's here:** {item['reason']}")
            dec = audit.query("SELECT * FROM decisions WHERE part_id = ? ORDER BY id DESC LIMIT 1", (pid,))
            if dec:
                st.caption(f"Pass below {dec[0]['t_low']:.3f} · act at {dec[0]['t_high']:.3f} · "
                           f"model {dec[0]['model_version']}")
            if show_truth:
                st.info(f"Answer key (demo only): **{truth.get(pid, '?')}**  (`{pid}`)")
            note = st.text_input("Note (optional)", key=f"note_{pid}")
            b1, b2 = st.columns(2)
            if b1.button("✅  Good: release", type="primary", key=f"good_{pid}"):
                secs = time.time() - st.session_state.shown_at
                flash(f"{anon(pid)}: {record_verdict(plant, audit, pid, 'good', reviewer, note, secs)}")
                st.rerun()
            if b2.button("❌  Defective: scrap", key=f"bad_{pid}"):
                secs = time.time() - st.session_state.shown_at
                flash(f"{anon(pid)}: {record_verdict(plant, audit, pid, 'defective', reviewer, note, secs)}")
                st.rerun()

# ---------------- Tab 2: approvals ----------------
with tab_approvals:
    pending = [p for p in plant.state["pending_approvals"] if p["status"] == "awaiting approval"]
    if not pending:
        st.success("No quarantine requests are waiting.")
    else:
        st.caption(f"{len(pending)} quarantine requests waiting. Showing the first 5. "
                   "(A real system would group these by batch.)")
        for p in pending[:5]:
            dec = audit.query("SELECT heatmap_path FROM decisions WHERE part_id = ? ORDER BY id DESC LIMIT 1",
                              (p["part_id"],))
            with st.container(border=True):
                c1, c2 = st.columns([2, 3])
                if dec and dec[0]["heatmap_path"]:
                    c1.image(load_heatmap(dec[0]["heatmap_path"]))
                c2.write(f"**{p['id']}** · part `{anon(p['part_id'])}` · score **{p['score']:.2f}**")
                c2.caption("The model was confident this part is defective and paused the line. "
                           "Quarantining the batch cannot be undone, so it needs a person.")
                if show_truth:
                    c2.info(f"Answer key (demo only): **{truth.get(p['part_id'], '?')}**")
                a, b, c = c2.columns(3)
                if a.button("Approve quarantine", key=f"ap_{p['id']}"):
                    flash(approve(plant, audit, p["id"], reviewer))
                    st.rerun()
                if b.button("Reject", key=f"rj_{p['id']}"):
                    flash(reject(plant, audit, p["id"], reviewer))
                    st.rerun()
                if c.button("Part is actually good", key=f"ov_{p['id']}"):
                    msg = record_override(plant, audit, p["id"], reviewer)
                    flash(msg.replace(p["part_id"], anon(p["part_id"])))
                    st.rerun()

# ---------------- Tab 3: history ----------------
with tab_history:
    rows = audit.query("""SELECT at, actor, event_type, part_id, verdict, reference, note
                          FROM human_events ORDER BY id DESC LIMIT 100""")
    if rows:
        hist = pd.DataFrame(rows)
        if not show_truth:
            hist["part_id"] = hist["part_id"].map(anon)
        st.dataframe(hist, hide_index=True)
    else:
        st.info("No human decisions yet.")
