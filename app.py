
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
DB = BASE / "data" / "mind_timeline.db"
DB.parent.mkdir(exist_ok=True)

CATEGORIES = ["observation", "interpretation", "imagination", "dream", "memory", "perception", "other"]
EVIDENCE = ["unknown", "reported", "supported", "unverified", "contradicted"]

def now():
    return datetime.now(timezone.utc).isoformat()

def conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS events(
            id TEXT PRIMARY KEY, title TEXT NOT NULL, date TEXT NOT NULL,
            category TEXT NOT NULL, context TEXT, significance TEXT,
            confidence REAL, evidence_status TEXT, tags TEXT, notes TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS relationships(
            id TEXT PRIMARY KEY, source_id TEXT NOT NULL, target_id TEXT NOT NULL,
            relationship_type TEXT NOT NULL, note TEXT, created_at TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS history(
            id TEXT PRIMARY KEY, action TEXT NOT NULL, entity_type TEXT NOT NULL,
            entity_id TEXT, snapshot TEXT, created_at TEXT NOT NULL)""")

def events():
    with conn() as c:
        return [dict(x) for x in c.execute("SELECT * FROM events ORDER BY date DESC").fetchall()]

def relationships():
    with conn() as c:
        return [dict(x) for x in c.execute("SELECT * FROM relationships ORDER BY created_at DESC").fetchall()]

def history():
    with conn() as c:
        return [dict(x) for x in c.execute("SELECT * FROM history ORDER BY created_at DESC").fetchall()]

def audit(action, entity_type, entity_id, snapshot):
    with conn() as c:
        c.execute("INSERT INTO history VALUES (?,?,?,?,?,?)",
                  (str(uuid.uuid4()), action, entity_type, entity_id, json.dumps(snapshot, ensure_ascii=False), now()))

def add_event(data):
    eid = str(uuid.uuid4())
    t = now()
    row = {
        "id": eid, "title": data["title"], "date": data["date"],
        "category": data["category"], "context": data["context"],
        "significance": data["significance"], "confidence": data["confidence"],
        "evidence_status": data["evidence_status"], "tags": data["tags"],
        "notes": data["notes"], "created_at": t, "updated_at": t
    }
    with conn() as c:
        c.execute("""INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                  tuple(row.values()))
    audit("create", "event", eid, row)
    return row

def delete_event(eid):
    with conn() as c:
        old = c.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
        if not old: return
        old = dict(old)
        c.execute("DELETE FROM relationships WHERE source_id=? OR target_id=?", (eid,eid))
        c.execute("DELETE FROM events WHERE id=?", (eid,))
    audit("delete", "event", eid, old)

def add_relationship(source, target, rtype, note):
    if source == target:
        st.error("An event cannot be related to itself.")
        return
    rid = str(uuid.uuid4())
    with conn() as c:
        c.execute("INSERT INTO relationships VALUES(?,?,?,?,?,?)",
                  (rid, source, target, rtype, note, now()))
    audit("create", "relationship", rid, {"id":rid,"source_id":source,"target_id":target,
                                           "relationship_type":rtype,"note":note})

def export_json():
    return {"version":"1.0","events":events(),"relationships":relationships(),"history":history()}

def anonymous_export():
    keep = ["id","date","category","confidence","evidence_status","created_at","updated_at"]
    return {"version":"1.0-anonymous",
            "events":[{k:e.get(k) for k in keep} for e in events()],
            "relationships":relationships()}

def clinician_summary():
    es = events()
    lines = ["Mind Timeline — Neutral Summary", "", "This is a structured record, not a diagnosis.", ""]
    for e in sorted(es, key=lambda x:x["date"]):
        lines += [
            f"Date: {e['date']}",
            f"Category: {e['category']}",
            f"Title: {e['title']}",
            f"Context: {e.get('context') or ''}",
            f"Significance: {e.get('significance') or ''}",
            f"Evidence status: {e.get('evidence_status') or 'unknown'}",
            f"Confidence: {e.get('confidence') if e.get('confidence') is not None else ''}",
            f"Notes: {e.get('notes') or ''}",
            "---"
        ]
    return "\n".join(lines)

init_db()

st.set_page_config(page_title="Mind Timeline v1.0", page_icon="🧠", layout="wide")
st.title("🧠 Mind Timeline v1.0")
st.caption("Privacy-first journaling and communication tool • Experience ≠ interpretation ≠ diagnosis")

tabs = st.tabs(["Timeline", "Add Event", "Relationships", "Reports & Export", "History"])

with tabs[0]:
    es = events()
    col1, col2 = st.columns(2)
    with col1:
        q = st.text_input("Search", placeholder="Search title, context, notes...")
    with col2:
        cat = st.selectbox("Filter category", ["all"] + CATEGORIES)
    filtered = [e for e in es if (not q or q.lower() in json.dumps(e, ensure_ascii=False).lower())
                and (cat == "all" or e["category"] == cat)]
    st.write(f"**{len(filtered)} event(s)**")
    for e in filtered:
        with st.expander(f"{e['date']} — {e['title']} [{e['category']}]"):
            st.write(f"**Context:** {e.get('context') or '—'}")
            st.write(f"**Significance:** {e.get('significance') or '—'}")
            st.write(f"**Evidence:** {e.get('evidence_status') or 'unknown'}")
            st.write(f"**Confidence:** {e.get('confidence') if e.get('confidence') is not None else '—'}")
            st.write(f"**Tags:** {e.get('tags') or '—'}")
            st.write(f"**Notes:** {e.get('notes') or '—'}")
            if st.button("Delete", key="del_"+e["id"]):
                delete_event(e["id"])
                st.rerun()

with tabs[1]:
    with st.form("add_event", clear_on_submit=True):
        title = st.text_input("Title *")
        date = st.date_input("Date")
        category = st.selectbox("Category", CATEGORIES)
        context = st.text_area("Context")
        significance = st.text_area("Significance")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.5, 0.05)
        evidence_status = st.selectbox("Evidence status", EVIDENCE)
        tags = st.text_input("Tags")
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Save Event")
    if submitted:
        if not title.strip():
            st.error("Title is required.")
        else:
            add_event({"title":title.strip(),"date":date.isoformat(),"category":category,
                       "context":context,"significance":significance,"confidence":confidence,
                       "evidence_status":evidence_status,"tags":tags,"notes":notes})
            st.success("Event saved.")
            st.rerun()

with tabs[2]:
    es = events()
    if len(es) < 2:
        st.info("Add at least two events to create a relationship.")
    else:
        labels = {f"{e['date']} — {e['title']}": e["id"] for e in es}
        source_label = st.selectbox("From event", list(labels))
        target_options = [x for x in labels if x != source_label]
        target_label = st.selectbox("To event", target_options)
        rtype = st.selectbox("Relationship type",
                             ["related","before","after","supports","contrasts","context_for","caused_by","custom"])
        note = st.text_area("Relationship note")
        if st.button("Create relationship"):
            add_relationship(labels[source_label], labels[target_label], rtype, note)
            st.success("Relationship created.")
            st.rerun()
        rs = relationships()
        st.subheader("Existing relationships")
        for r in rs:
            sm = next((e["title"] for e in es if e["id"]==r["source_id"]), r["source_id"])
            tm = next((e["title"] for e in es if e["id"]==r["target_id"]), r["target_id"])
            st.write(f"**{sm}** → `{r['relationship_type']}` → **{tm}** — {r.get('note') or ''}")

with tabs[3]:
    st.subheader("Neutral clinician summary")
    summary = clinician_summary()
    st.text_area("Preview", summary, height=400)
    st.download_button("Download clinician summary", summary, "mind_timeline_clinician_summary.txt", "text/plain")

    st.subheader("JSON export")
    raw = json.dumps(export_json(), indent=2, ensure_ascii=False)
    st.download_button("Download full JSON", raw, "mind_timeline_v1.0.json", "application/json")

    st.subheader("Anonymous structured export")
    anon = json.dumps(anonymous_export(), indent=2, ensure_ascii=False)
    st.download_button("Download anonymous JSON", anon, "mind_timeline_anonymous_v1.0.json", "application/json")
    st.info("Anonymous export removes title, context, significance, tags and notes, but should still be reviewed before external sharing.")

with tabs[4]:
    hs = history()
    if hs:
        df = pd.DataFrame(hs)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No history records yet.")
