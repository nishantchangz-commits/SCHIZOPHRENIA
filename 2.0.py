
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

import streamlit as st
from database.db import init_db, seed, get_conn
from services.project_service import list_projects,create_project,delete_project
from services.event_service import list_events,create_event,update_event,delete_event
from services.relationship_service import list_relationships,create_relationship,delete_relationship
from pages import dashboard,timeline,mind_map,events,relationships,analytics,reports,import_export,privacy,settings

init_db(); seed()
st.set_page_config(page_title="Mind Timeline v2.0",page_icon="🧠",layout="wide")

projects=list_projects()
if not projects:
    create_project("My Timeline","Default project",True)
    projects=list_projects()

st.sidebar.title("🧠 Mind Timeline v2.0")
project_labels={p["name"]:p["id"] for p in projects}
selected_name=st.sidebar.selectbox("Project",list(project_labels))
project_id=project_labels[selected_name]
project=next(p for p in projects if p["id"]==project_id)

st.sidebar.caption("Local-first journaling & communication")
if st.sidebar.button("➕ New project"):
    create_project("New Timeline","",True); st.rerun()

if st.sidebar.button("🗑️ Delete current project"):
    if len(projects)>1:
        delete_project(project_id); st.rerun()
    else:
        st.sidebar.error("Keep at least one project.")

events_data=list_events(project_id)
rels=list_relationships(project_id)
with get_conn() as c:
    history_data=[dict(r) for r in c.execute(
        "SELECT * FROM history WHERE project_id=? ORDER BY created_at DESC",(project_id,)).fetchall()]

page=st.sidebar.radio("Navigate",[
    "Dashboard","Timeline","Mind Map","Events","Relationships","Analytics",
    "Reports & Export","Backup / Restore","Privacy","Settings"
])

if page=="Dashboard": dashboard.render(events_data,rels)
elif page=="Timeline": timeline.render(events_data)
elif page=="Mind Map": mind_map.render(events_data,rels)
elif page=="Events": events.render(project_id,events_data,create_event,update_event,delete_event)
elif page=="Relationships": relationships.render(project_id,events_data,rels,create_relationship,delete_relationship)
elif page=="Analytics": analytics.render(events_data)
elif page=="Reports & Export": reports.render(project,events_data,rels,history_data)
elif page=="Backup / Restore": import_export.render(project_id)
elif page=="Privacy": privacy.render(project)
elif page=="Settings": settings.render(project)

st.sidebar.divider()
st.sidebar.caption("Experience ≠ interpretation ≠ diagnosis")
