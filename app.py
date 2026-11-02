"""
Smart Electricity Consumption Prediction and Monitoring System
==============================================================
app.py  –  Streamlit web application (enhanced version)

Run with:
    streamlit run app.py
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import json
import os
from datetime import datetime, timedelta

# ── third-party ───────────────────────────────────────────────────────────────
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* gradient header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: #e94560; margin-bottom: 0.3rem; }
    .main-header p  { color: #a8b2d8; margin: 0; font-size: 1rem; }

    /* metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d5a8e);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        text-align: center;
        border-left: 4px solid #e94560;
        margin-bottom: 0.5rem;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #ffd700; }
    .metric-card .label { font-size: 0.85rem; color: #a8b2d8; margin-top: 0.2rem; }

    /* alert box */
    .alert-high {
        background: linear-gradient(135deg, #ff4757, #c0392b);
        border-radius: 10px; padding: 1rem 1.5rem;
        color: white; font-weight: 600;
    }
    .alert-ok {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        border-radius: 10px; padding: 1rem 1.5rem;
        color: white; font-weight: 600;
    }

    /* section heading */
    .section-title {
        font-size: 1.3rem; font-weight: 700;
        color: #e94560; border-left: 4px solid #e94560;
        padding-left: 0.7rem; margin: 1.5rem 0 1rem 0;
    }

    /* sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
FILES_NEEDED = [
    "electricity_model.keras",
    "scaler.pkl",
    "daily_usage.csv",
]


def all_files_exist() -> bool:
    return all(os.path.exists(f) for f in FILES_NEEDED)


@st.cache_resource(show_spinner="Loading model …")
def load_model_and_scaler():
    model  = tf.keras.models.load_model("electricity_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


@st.cache_data(show_spinner="Loading data …")
def load_daily_data() -> pd.DataFrame:
    df = pd.read_csv("daily_usage.csv", index_col=0, parse_dates=True)
    df.columns = ["Consumption_kWh"]
    df = df.dropna()
    return df


def predict_next_day(model, scaler, last_7: np.ndarray) -> float:
    scaled = scaler.transform(last_7.reshape(-1, 1))
    X      = scaled.reshape(1, 7, 1)
    pred   = model.predict(X, verbose=0)
    value  = scaler.inverse_transform(pred)[0][0]
    return max(0.0, float(value))


def compute_bill(kwh: float, rate: float, tax_pct: float = 0.0) -> float:
    return kwh * rate * (1 + tax_pct / 100)


def anomaly_flags(series: pd.Series, sigma: float = 2.0) -> pd.Series:
    mean = series.mean()
    std  = series.std()
    return series > (mean + sigma * std)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>⚡ Smart Electricity Consumption Predictor</h1>
        <p>LSTM-powered energy monitoring · Prediction · Bill estimation · Anomaly detection</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# GATE: model must exist
# ─────────────────────────────────────────────────────────────────────────────
if not all_files_exist():
    st.error(
        "⚠️  Required files not found.  "
        "Please run `python train.py` first to train the model.",
        icon="🚨",
    )
    st.code("python train.py", language="bash")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
model, scaler = load_model_and_scaler()
data          = load_daily_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  –  user settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️  Settings")

    rate = st.number_input(
        "Electricity price per kWh (₹)",
        min_value=1.0, max_value=50.0, value=8.0, step=0.5,
    )
    tax_pct = st.slider(
        "Tax / surcharge (%)", min_value=0, max_value=30, value=5
    )
    sigma_threshold = st.slider(
        "Anomaly sensitivity (σ)", min_value=1.0, max_value=4.0,
        value=2.0, step=0.5,
        help="Lower value = more sensitive alerts",
    )
    chart_days = st.selectbox(
        "History window for charts",
        [30, 60, 90, 180, 365, "All"],
        index=2,
    )

    st.divider()
    st.markdown("### 📋  Dataset info")
    st.write(f"**Total days:** {len(data):,}")
    st.write(f"**From:** {data.index.min().date()}")
    st.write(f"**To:**   {data.index.max().date()}")

    # Show training metrics if available
    if os.path.exists("training_history.json"):
        st.divider()
        st.markdown("### 🏆  Model metrics (test set)")
        with open("training_history.json") as fh:
            m = json.load(fh)
        st.write(f"**MAE:**  {m.get('test_mae_kwh', 'N/A')} kWh")
        st.write(f"**RMSE:** {m.get('test_rmse_kwh', 'N/A')} kWh")
        st.write(f"**MAPE:** {m.get('test_mape_pct', 'N/A')} %")
        if "improvement_over_naive_pct" in m:
            st.success(
                f"LSTM is **{m['improvement_over_naive_pct']} %** better "
                "than a naïve baseline!"
            )

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
last_7      = data["Consumption_kWh"].tail(7).values
prediction  = predict_next_day(model, scaler, last_7)

avg_30       = data["Consumption_kWh"].tail(30).mean()
total_30     = data["Consumption_kWh"].tail(30).sum()
bill_30      = compute_bill(total_30, rate, tax_pct)
pred_bill    = compute_bill(prediction, rate, tax_pct)

delta_pct    = ((prediction - avg_30) / avg_30) * 100

# ─────────────────────────────────────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Energy Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="value">{prediction:.2f} kWh</div>
            <div class="label">⚡ Tomorrow's Prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="value">{avg_30:.2f} kWh</div>
            <div class="label">📅 30-Day Daily Average</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="value">₹{bill_30:,.0f}</div>
            <div class="label">🧾 Est. Bill (Last 30 Days)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    arrow = "📈" if delta_pct > 0 else "📉"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="value">{delta_pct:+.1f}%</div>
            <div class="label">{arrow} vs 30-Day Average</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – DAILY CONSUMPTION CHART (interactive Plotly)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📈 Daily Electricity Consumption</div>', unsafe_allow_html=True)

if chart_days == "All":
    display_data = data.copy()
else:
    display_data = data.tail(int(chart_days))

anomalies = anomaly_flags(display_data["Consumption_kWh"], sigma=sigma_threshold)
anom_data = display_data[anomalies]

fig_daily = go.Figure()

fig_daily.add_trace(
    go.Scatter(
        x=display_data.index,
        y=display_data["Consumption_kWh"],
        mode="lines",
        name="Daily Usage",
        line=dict(color="#4fc3f7", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.08)",
    )
)

# 30-day rolling average
rolling = display_data["Consumption_kWh"].rolling(30, min_periods=1).mean()
fig_daily.add_trace(
    go.Scatter(
        x=display_data.index,
        y=rolling,
        mode="lines",
        name="30-Day Rolling Avg",
        line=dict(color="#ffd700", width=2, dash="dot"),
    )
)

# Anomaly markers
if not anom_data.empty:
    fig_daily.add_trace(
        go.Scatter(
            x=anom_data.index,
            y=anom_data["Consumption_kWh"],
            mode="markers",
            name="⚠️ High Usage",
            marker=dict(color="#ff4757", size=8, symbol="triangle-up"),
        )
    )

# Tomorrow prediction marker
tomorrow = data.index[-1] + timedelta(days=1)
fig_daily.add_trace(
    go.Scatter(
        x=[tomorrow],
        y=[prediction],
        mode="markers+text",
        name="🔮 Tomorrow (Predicted)",
        marker=dict(color="#2ecc71", size=14, symbol="star"),
        text=[f"{prediction:.1f} kWh"],
        textposition="top center",
    )
)

fig_daily.update_layout(
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,30,50,0.6)",
    font=dict(color="white"),
    legend=dict(
        bgcolor="rgba(30,58,95,0.8)",
        bordercolor="#4fc3f7",
        borderwidth=1,
    ),
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.08)",
        title="Consumption (kWh)",
    ),
    hovermode="x unified",
)

st.plotly_chart(fig_daily, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – WEEKLY & MONTHLY BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📅 Weekly & Monthly Breakdown</div>', unsafe_allow_html=True)

col_w, col_m = st.columns(2)

# ---- Weekly bar chart (last 7 days) ----
with col_w:
    weekly = data["Consumption_kWh"].tail(7).copy()
    weekly.index = [d.strftime("%a %d %b") for d in weekly.index]

    fig_week = px.bar(
        x=weekly.index, y=weekly.values,
        labels={"x": "Day", "y": "kWh"},
        title="Last 7 Days",
        color=weekly.values,
        color_continuous_scale="Blues",
    )
    fig_week.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,30,50,0.6)",
        font=dict(color="white"),
        coloraxis_showscale=False,
        showlegend=False,
        height=300,
    )
    st.plotly_chart(fig_week, use_container_width=True)
    st.info(f"🗓️  Total last 7 days: **{weekly.sum():.2f} kWh**  ≈  ₹{compute_bill(weekly.sum(), rate, tax_pct):,.0f}")

# ---- Monthly pie / trend ----
with col_m:
    monthly = data["Consumption_kWh"].tail(30)
    monthly_resample = (
        data["Consumption_kWh"]
        .tail(90)
        .resample("W")
        .sum()
        .reset_index()
    )
    monthly_resample.columns = ["Week", "kWh"]

    fig_month = px.bar(
        monthly_resample,
        x="Week", y="kWh",
        title="Weekly Totals (Last 90 Days)",
        color="kWh",
        color_continuous_scale="Oranges",
    )
    fig_month.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,30,50,0.6)",
        font=dict(color="white"),
        coloraxis_showscale=False,
        height=300,
    )
    st.plotly_chart(fig_month, use_container_width=True)
    st.info(f"📆  Total last 30 days: **{monthly.sum():.2f} kWh**  ≈  ₹{compute_bill(monthly.sum(), rate, tax_pct):,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – BILL ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🧾 Electricity Bill Estimator</div>', unsafe_allow_html=True)

b1, b2, b3 = st.columns(3)

with b1:
    daily_bill = compute_bill(avg_30, rate, tax_pct)
    st.metric("Estimated Daily Bill", f"₹{daily_bill:.2f}")

with b2:
    weekly_bill = compute_bill(data["Consumption_kWh"].tail(7).sum(), rate, tax_pct)
    st.metric("Estimated Weekly Bill", f"₹{weekly_bill:.2f}")

with b3:
    st.metric("Estimated Monthly Bill", f"₹{bill_30:,.2f}")

# Projection gauge using Plotly
fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number+delta",
        value=bill_30,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Monthly Bill (₹)", "font": {"size": 18, "color": "white"}},
        delta={"reference": avg_30 * 30 * rate, "relative": False},
        number={"prefix": "₹", "font": {"size": 28, "color": "#ffd700"}},
        gauge={
            "axis": {"range": [0, bill_30 * 1.5], "tickcolor": "white"},
            "bar": {"color": "#e94560"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 2,
            "bordercolor": "#4fc3f7",
            "steps": [
                {"range": [0, bill_30 * 0.5],  "color": "rgba(46,204,113,0.3)"},
                {"range": [bill_30 * 0.5, bill_30 * 0.8], "color": "rgba(255,215,0,0.3)"},
                {"range": [bill_30 * 0.8, bill_30 * 1.5], "color": "rgba(255,71,87,0.3)"},
            ],
        },
    )
)
fig_gauge.update_layout(
    paper_bgcolor="rgba(15,30,50,0.6)",
    font=dict(color="white"),
    height=280,
)
st.plotly_chart(fig_gauge, use_container_width=True)
st.caption(
    "💡 Estimated using your selected per-unit rate and tax. "
    "Actual bills may include slab pricing, fixed charges, and government levies."
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔍 Anomaly Detection</div>', unsafe_allow_html=True)

last_30   = data["Consumption_kWh"].tail(30)
mean_u    = last_30.mean()
std_u     = last_30.std()
threshold = mean_u + sigma_threshold * std_u
latest_u  = last_30.iloc[-1]

a1, a2 = st.columns([2, 1])

with a1:
    # Consumption band chart
    fig_anom = go.Figure()
    fig_anom.add_trace(
        go.Scatter(
            x=last_30.index, y=last_30.values,
            mode="lines+markers",
            name="Daily Usage",
            line=dict(color="#4fc3f7"),
            marker=dict(size=5),
        )
    )
    fig_anom.add_hline(
        y=mean_u, line_dash="dot", line_color="#ffd700",
        annotation_text="Mean", annotation_position="right",
    )
    fig_anom.add_hline(
        y=threshold, line_dash="dash", line_color="#ff4757",
        annotation_text=f"Threshold ({sigma_threshold:.1f}σ)",
        annotation_position="right",
    )
    # Shade anomaly region
    fig_anom.add_hrect(
        y0=threshold, y1=last_30.max() * 1.1,
        fillcolor="rgba(255,71,87,0.1)", line_width=0,
    )
    fig_anom.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,30,50,0.6)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="kWh"),
        title="Last 30 Days with Anomaly Band",
    )
    st.plotly_chart(fig_anom, use_container_width=True)

with a2:
    st.metric("Mean (last 30 d)", f"{mean_u:.2f} kWh")
    st.metric("Std Dev", f"{std_u:.2f} kWh")
    st.metric("Alert Threshold", f"{threshold:.2f} kWh")
    st.metric("Latest Reading", f"{latest_u:.2f} kWh")

    if latest_u > threshold:
        st.markdown(
            '<div class="alert-high">⚠️  HIGH USAGE ALERT!<br>'
            f'Latest reading exceeds the {sigma_threshold:.1f}σ threshold.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="alert-ok">✅  Normal<br>No unusual consumption detected.</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – PREDICTION DETAILS & NEXT 7-DAY FORECAST
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔮 Forecast Details</div>', unsafe_allow_html=True)

f1, f2 = st.columns([1, 2])

with f1:
    st.markdown("#### Tomorrow's Prediction")
    st.markdown(
        f"""
        | Metric | Value |
        |--------|-------|
        | Predicted kWh | **{prediction:.3f}** |
        | Estimated cost | **₹{pred_bill:.2f}** |
        | vs 30-day avg | **{delta_pct:+.1f}%** |
        | Last actual reading | **{latest_u:.3f} kWh** |
        """
    )

with f2:
    # Rolling 7-day forecast (auto-regressive)
    st.markdown("#### Auto-regressive 7-Day Outlook")
    buffer = list(last_7)
    future_preds = []
    for _ in range(7):
        inp   = np.array(buffer[-7:])
        nxt   = predict_next_day(model, scaler, inp)
        future_preds.append(nxt)
        buffer.append(nxt)

    future_dates = [data.index[-1] + timedelta(days=i + 1) for i in range(7)]
    future_df    = pd.DataFrame(
        {"Date": future_dates, "Predicted kWh": future_preds}
    )
    future_df["Date"] = future_df["Date"].dt.strftime("%a, %d %b")
    future_df["Est. Cost (₹)"] = future_df["Predicted kWh"].apply(
        lambda x: compute_bill(x, rate, tax_pct)
    )
    future_df = future_df.set_index("Date")
    future_df = future_df.round(2)

    fig_fut = px.bar(
        future_df.reset_index(),
        x="Date", y="Predicted kWh",
        color="Predicted kWh",
        color_continuous_scale="Teal",
        title="",
    )
    fig_fut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,30,50,0.6)",
        font=dict(color="white"),
        coloraxis_showscale=False,
        height=220,
        margin=dict(t=10),
    )
    st.plotly_chart(fig_fut, use_container_width=True)
    st.dataframe(future_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – EVALUATION CHART (if available)
# ─────────────────────────────────────────────────────────────────────────────
if os.path.exists("evaluation_chart.png"):
    st.markdown('<div class="section-title">📉 Model Evaluation (Test Set)</div>', unsafe_allow_html=True)
    st.image("evaluation_chart.png", use_container_width=True)
    st.caption("Left: Training vs Validation Loss  |  Right: Actual vs Predicted on unseen test data")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – RAW DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🗂️  Raw Data Explorer"):
    n_rows = st.slider("Rows to show", 10, 200, 30)
    view   = st.radio("View", ["Most recent", "Oldest"], horizontal=True)
    show_df = data.tail(n_rows) if view == "Most recent" else data.head(n_rows)
    st.dataframe(show_df, use_container_width=True)
    st.download_button(
        "⬇️  Download CSV",
        data=data.to_csv(),
        file_name="daily_electricity_usage.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style='text-align:center; color:#4a5568; font-size:0.85rem;'>
        ⚡ Smart Electricity Consumption Prediction System
        &nbsp;|&nbsp; Built with LSTM + Streamlit
        &nbsp;|&nbsp; Dataset: UCI Household Electric Power Consumption
    </div>
    """,
    unsafe_allow_html=True,
)
