import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="Nexus: Reinforcement Learning Dashboard", layout="wide")

st.title("📊 Nexus: Real-time Adaptive Learning Dashboard")
st.markdown("Student performance tracking using Reinforcement Learning")

# --- SIDEBAR: API Status ---
st.sidebar.header("System Status")
try:
    # Check if FastAPI is running
    response = requests.get("http://127.0.0.1:8000/")
    if response.status_code == 200:
        st.sidebar.success("Backend: Online")
        
except:
    st.sidebar.error("Backend: Offline (Run main.py)")

# --- MAIN SECTION: A/B Testing Results ---
# --- SECTION 1: CLASS STATS (Puri Class) ---
st.header("🏫 Class Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Avg Score (RL Group)", "85%", "12%")
with col2:
    st.metric("Avg Score (Static)", "65%", "-5%")
with col3:
    st.metric("Total Students", "1,240")

# Bar Chart for Class
class_df = pd.DataFrame({
    "Group": ["Static", "RL-Powered"],
    "Completion %": [68, 91]
})
st.bar_chart(class_df.set_index("Group"))

# --- REWARD TRACKING (RL Learning Curve) ---
st.divider()
st.subheader("📈 RL Agent Learning Progress")
# Mock training data to show progress
training_data = pd.DataFrame({
    'Episode': range(0, 200, 10),
    'Reward': [-10, -5, 2, 8, 15, 25, 40, 60, 85, 95, 100, 105, 110, 112, 115, 118, 120, 122, 125, 128]
})
st.line_chart(training_data.set_index('Episode'), y_label='Reward', x_label='Episode')

# --- INTERACTIVE SECTION ---
st.divider()
st.subheader("🧪 Test the API")
u_id = st.number_input("Enter User ID", min_value=1, value=100)
score = st.slider("Mock Quiz Score", 0.0, 2.0, 0.4)

if st.button("Get AI Analysis & Recommendation"):
    payload = {"user_id": u_id, "quiz_score": score, "event_type": "quiz_failed"}
    try:
        res = requests.post("http://127.0.0.1:8000/log_event", json=payload)
        rec = res.json()["recommendation"]
        group = res.json()["group"]
        st.info(f"**Group:** {group} | **AI Recommendation:** {rec}")
        # Mock History Graph for this user
        history = pd.DataFrame({
            "Attempt": [1, 2, 3],
            "Score": [0.2, 0.35, score]
        })
        st.line_chart(history.set_index("Attempt"), y_label="Quiz Score", x_label="Attempt")
    except:
        st.warning("Please run main.py first to test the API.")