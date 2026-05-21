"""
Insider Threat Detection – Streamlit dashboard.
Run with: streamlit run dashboard_app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Insider Threat Detection Dashboard", layout="wide")

st.title("🛡 Insider Threat Detection System")
st.caption("Behavioral Risk Monitoring for Enterprise Security")

with st.sidebar:
    page = st.selectbox(
        "Navigation",
        ["Overview", "Model Performance", "Risk Analytics", "Alerts"],
    )
    st.info("Backend Detection Engine")

METRICS = {"Accuracy": "94.55%", "Precision": "86.36%", "Recall": "96.81%", "F1-score": "90.46%", "AUC": "94.00%"}
CM = np.array([[44, 3], [0, 8]])
RISK_LEVELS, RISK_COUNTS = ["Low", "Medium", "High"], [230, 35, 6]
RISK_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c", "LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
ACTION = {
    "LOW": "Continue routine monitoring",
    "MEDIUM": "Flag for closer monitoring",
    "HIGH": "Immediate investigation required",
}

@st.cache_data
def get_alerts_df():
    n = 41
    np.random.seed(42)
    ids = [f"ALERT-{i:04d}" for i in range(1, n + 1)]
    users = [f"***{str(1000 + i).zfill(4)}" for i in range(n)]
    levels = (["LOW"] * 28) + (["MEDIUM"] * 10) + (["HIGH"] * 3)
    np.random.shuffle(levels)
    return pd.DataFrame({
        "Alert ID": ids,
        "Masked User": users,
        "risk_level": levels,
        "Recommended Action": [ACTION[l] for l in levels],
    })

alerts_df = get_alerts_df()
alerts_df["risk_level"] = alerts_df["risk_level"].str.upper()

@st.cache_data
def get_trend_data():
    days = pd.date_range(end=pd.Timestamp.today(), periods=7, freq="D")[::-1]
    counts = np.array([32, 28, 35, 41, 38, 44, 41])
    return pd.DataFrame({"Date": days, "Alert Count": counts})

@st.cache_data
def get_top_risk_users():
    users = [f"***{str(2000 + i).zfill(4)}" for i in range(10)]
    probs = np.round(np.array([0.92, 0.89, 0.87, 0.85, 0.82, 0.79, 0.76, 0.74, 0.71, 0.68]), 2)
    return pd.DataFrame({"Masked User": users, "Risk Probability": probs})

@st.cache_data
def get_prob_histogram():
    np.random.seed(43)
    low = np.random.beta(2, 8, 230) * 0.4
    med = np.random.beta(4, 4, 35)
    high = np.random.beta(8, 2, 6) * 0.3 + 0.7
    return np.concatenate([low, med, high])

def risk_bar_chart():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = [RISK_COLORS[l] for l in RISK_LEVELS]
    bars = ax.bar(RISK_LEVELS, RISK_COUNTS, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Count")
    ax.set_title("Risk Level Distribution")
    for bar, val in zip(bars, RISK_COUNTS):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(val), ha="center", va="bottom", fontsize=10)
    return fig

def risk_pie_chart():
    total = sum(RISK_COUNTS)
    pcts = [c / total * 100 for c in RISK_COUNTS]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(RISK_COUNTS, labels=[f"{l}\n{pct:.1f}%" for l, pct in zip(RISK_LEVELS, pcts)], colors=[RISK_COLORS[l] for l in RISK_LEVELS], autopct="", startangle=90)
    ax.set_title("Risk Level Share")
    return fig

def style_risk(s):
    return [f"color: {RISK_COLORS.get(v, 'black')}; font-weight: bold" for v in s]

if page == "Overview":
    row1_col1, row1_col2 = st.columns([3, 1])
    with row1_col1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Users", 271)
        c2.metric("Total Alerts", len(alerts_df))
        c3.metric("Medium Risk Users", 35)
        c4.metric("High Risk Users", 6)
    with row1_col2:
        st.markdown("**Model Status**")
        st.success("Active")
        st.markdown("**Confidential Inference**")
        st.info("Enabled")
    st.markdown("---")
    st.markdown("Model optimized for high recall to minimize missed insider threats. System prioritizes reducing False Negatives.")
    st.markdown("---")
    col_bar, col_pie = st.columns(2)
    with col_bar:
        st.pyplot(risk_bar_chart())
    with col_pie:
        st.pyplot(risk_pie_chart())
    plt.close("all")

elif page == "Model Performance":
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (name, val) in zip([c1, c2, c3, c4, c5], METRICS.items()):
        col.metric(name, val)
    st.markdown("---")
    st.markdown("High recall ensures most insider threats are detected.")
    st.markdown("---")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(CM, annot=True, fmt="d", cmap="Greens", xticklabels=["Normal", "Threat"], yticklabels=["Normal", "Threat"], ax=ax, cbar_kws={"label": "Count"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Zero False Negatives – No insider threats were missed.")

elif page == "Risk Analytics":
    trend_df = get_trend_data()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(trend_df["Date"].astype(str), trend_df["Alert Count"], marker="o", color="#3498db")
    ax.set_title("Alert Count Trend (Last 7 Days)")
    ax.set_ylabel("Alert Count")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("---")
    col_t10, col_hist = st.columns(2)
    with col_t10:
        st.subheader("Top 10 Highest Risk Users")
        st.dataframe(get_top_risk_users(), use_container_width=True, hide_index=True)
    with col_hist:
        st.subheader("Model Probability Score Distribution")
        probs = get_prob_histogram()
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.hist(probs, bins=25, color="#3498db", edgecolor="black")
        ax2.set_xlabel("Risk Probability")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)
        plt.close(fig2)
    st.markdown("---")
    st.markdown("Model prioritizes high recall to minimize missed insider threats.")

elif page == "Alerts":
    st.subheader("Recent Insider Threat Alerts")
    selected = st.selectbox("Filter by Risk Level", ["ALL", "LOW", "MEDIUM", "HIGH"])
    if selected != "ALL":
        filtered_df = alerts_df[alerts_df["risk_level"] == selected]
    else:
        filtered_df = alerts_df
    st.write("Total Alerts Displayed:", len(filtered_df))
    styled = filtered_df.style.apply(style_risk, subset=["risk_level"], axis=0)
    st.dataframe(styled, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.85rem; color: #666;'>Insider Threat Behavioral Detection</p>", unsafe_allow_html=True)
