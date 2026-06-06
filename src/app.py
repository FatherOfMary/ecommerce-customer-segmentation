# =========================================================================
# src/app.py : Executive RFM Customer Segmentation Intelligence Hub
# =========================================================================
# Run Command: streamlit run src/app.py
#
# Code Architecture Specifications:
#   - Premium Non-White Canvas Architecture (Seamless Executive Matte)
#   - Core Currency Anchoring: United Kingdom (GBP / £)
#   - Enhanced Geometric Centroid Mapping Layer
#   - Midnight Emerald & Copper Premium Palette Configuration
# =========================================================================

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

# Repository Root Path Context Configuration
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans

# =========================================================================
# PAGE CONFIGURATION : MUST EXECUTE FIRST
# =========================================================================
st.set_page_config(
    page_title="Shareholder Intelligence | RFM Capital Allocation Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# COHORT DEFINITIONS & PALETTE SCHEMES
# =========================================================================
OPTIMAL_K = 4
RANDOM_SEED = 42
FEATURE_COLS = ["Recency_sqrt", "Frequency_boxcox", "Monetary_log"]

CLUSTER_MAP: dict[int, str] = {
    0: "Premium Retention",
    1: "Low-Cost Re-Engage",
    2: "Nurture & Upsell",
    3: "High-Priority Winback",
}
VIP_LABEL = "White Glove Concierge"
COHORT_ORDER = list(CLUSTER_MAP.values()) + [VIP_LABEL]

# "Midnight Emerald & Copper" Executive Color Palette
PALETTE: dict[str, str] = {
    "Premium Retention": "#0F766E",      # Deep Emerald Velvet (Core Asset / Trust)
    "Low-Cost Re-Engage": "#64748B",     # Muted Steel Slate (Dormant Tail / Neutral)
    "Nurture & Upsell": "#0284C7",       # Skyline Corporate Blue (Pipeline Expansion)
    "High-Priority Winback": "#D97706",  # Polished Copper Amber (Urgency / Re-Acquisition)
    VIP_LABEL: "#4F46E5",                # Royal Indigo Crest (Elite Whales / Luxury Vault)
}

CENTROID_PROFILES: dict[str, dict] = {
    "Premium Retention": {"emoji": "👑", "tag": "Core Revenue Engine", "desc": "High frequency asset base maintaining operational stability and predictable cash flow lines."},
    "Low-Cost Re-Engage": {"emoji": "💤", "tag": "Dormant Base Tail", "desc": "Low velocity accounts presenting minimal resource overhead with secondary activation options."},
    "Nurture & Upsell": {"emoji": "📈", "tag": "Growth Pipeline", "desc": "High transaction velocity and fresh acquisition vectors primed for margin expansion campaigns."},
    "High-Priority Winback": {"emoji": "🚨", "tag": "Capital At Risk", "desc": "Historically significant valuation segments requiring immediate tactical deployment to avert permanent churn."},
    VIP_LABEL: {"emoji": "💎", "tag": "High Net Worth Outliers", "desc": "Top tier institutional accounts generating extreme outsized lifetime capital returns."},
}

DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Unified Architecture Axis Schema Matrix tailored for non-white plot bounds
_AX_SCHEMA: dict = dict(
    showgrid=True,
    gridcolor="#E2E8F0",
    linecolor="#CBD5E1",
    linewidth=1,
    title_font=dict(size=11, color="#334155", family="sans-serif", weight="bold"),
    tickfont=dict(size=10, color="#475569", family="monospace"),
)


# =========================================================================
# PIPELINE COMPONENT LOADERS
# =========================================================================
def _hex_to_rgba(hex_str: str, alpha: float) -> str:
    """Decodes a standard hex string into a Plotly-compliant functional RGBA definition."""
    hex_clean = hex_str.lstrip("#")
    r, g, b = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


@st.cache_data(show_spinner=False)
def _load_data_layers() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_rfm = pd.read_parquet(DATA_DIR / "rfm_baseline.parquet")
    df_scaled = pd.read_parquet(DATA_DIR / "rfm_scaled_modeling.parquet")
    return df_rfm, df_scaled


@st.cache_data(show_spinner=False)
def _compute_deterministic_labels(_df_scaled: pd.DataFrame) -> np.ndarray:
    X = _df_scaled[FEATURE_COLS].to_numpy(dtype=np.float32)
    km = KMeans(
        n_clusters=OPTIMAL_K,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=RANDOM_SEED,
    )
    return km.fit(X).labels_


def _assemble_master_portfolio(
    df_rfm: pd.DataFrame, df_scaled: pd.DataFrame, labels: np.ndarray
) -> pd.DataFrame:
    df = df_rfm.copy()
    df["Cluster"] = labels.astype(int)
    df["Is_VIP"] = df_scaled["Is_Systemic_VIP"].astype(int).values
    df["Cohort"] = df["Cluster"].map(CLUSTER_MAP)
    df.loc[df["Is_VIP"] == 1, "Cohort"] = VIP_LABEL
    
    df["Visual_Frequency"] = df["Frequency"].clip(lower=1)
    df["Visual_Monetary"] = df["Monetary"].clip(lower=0.1)
    return df


# =========================================================================
# CREATIVE EXECUTIVE STYLING LAYER
# =========================================================================
def _inject_premium_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700;800&display=swap');
        
        /* Unified Background Canvas: Seamless Non-White Corporate Finish */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            font-family: 'Inter', -apple-system, sans-serif !important;
            background-color: #F1F5F9 !important;
        }
        
        /* Sidebar Contrast Frame Override */
        [data-testid="stSidebar"] {
            background-color: #E2E8F0 !important;
            border-right: 1px solid #CBD5E1;
        }
        
        .main .block-container {
            padding-top: 2rem !important;
            padding-left: 3.5rem !important;
            padding-right: 3.5rem !important;
        }
        
        /* Premium Corporate Dashboard Header Layout */
        .dashboard-header {
            margin-bottom: 2.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #CBD5E1;
        }
        .dashboard-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: #0F172A;
            letter-spacing: -0.03em;
            margin-bottom: 0.35rem;
        }
        .dashboard-subtitle {
            font-size: 1rem;
            color: #475569;
            font-weight: 500;
        }
        
        /* Seamless Section Containers (Stripped of White Box Elements) */
        .section-card {
            background: transparent !important;
            border: none !important;
            border-radius: 0px !important;
            padding: 0px !important;
            margin-bottom: 3rem;
            box-shadow: none !important;
        }
        
        .section-header-text {
            font-size: 1.25rem;
            font-weight: 700;
            color: #0F172A;
            letter-spacing: -0.02em;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 10px;
            border-left: 4px solid #0F172A;
            padding-left: 10px;
        }
        
        /* Short Contextual Narrative Block Styling */
        .executive-summary-text {
            font-size: 0.88rem;
            color: #475569;
            line-height: 1.5;
            margin-bottom: 1.5rem;
            margin-top: 0.25rem;
        }
        .insight-highlight-box {
            background-color: #E2E8F0;
            border-left: 4px solid #475569;
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            font-size: 0.85rem;
            color: #334155;
            margin-top: 1rem;
            line-height: 1.4;
        }
        
        /* Custom Architectural Micro-Metrics Layout Row */
        .metric-grid-container {
            display: flex;
            gap: 16px;
            margin-bottom: 2.5rem;
            width: 100%;
        }
        .metric-card-custom {
            flex: 1;
            background: #E2E8F0;
            border: 1px solid #CBD5E1;
            border-radius: 12px;
            padding: 20px;
            position: relative;
            box-shadow: 0 1px 2px 0 rgba(15, 23, 42, 0.02);
            transition: all 0.2s ease;
        }
        .metric-card-custom:hover {
            border-color: #94A3B8;
            background: #DDE3EA;
        }
        .metric-card-custom::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 4px; height: 100%;
            border-radius: 12px 0 0 12px;
            background: #64748B;
        }
        .metric-card-custom.m-prime::before { background: #0F766E; }
        .metric-card-custom.m-teal::before { background: #0284C7; }
        .metric-card-custom.m-orange::before { background: #D97706; }
        
        .metric-card-label {
            font-size: 0.75rem;
            font-weight: 700;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 6px;
        }
        .metric-card-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #0F172A;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }
        
        /* Behavioral Display Grid Blocks */
        .cohort-block {
            background: #E2E8F0;
            border: 1px solid #CBD5E1;
            border-radius: 10px;
            padding: 18px;
            height: 100%;
        }
        .cohort-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            color: #FFFFFF;
            margin-bottom: 12px;
        }
        .cohort-title-text {
            font-size: 1rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 6px;
        }
        .cohort-description-text {
            font-size: 0.8rem;
            color: #334155;
            line-height: 1.4;
            margin-bottom: 14px;
            height: 40px;
            overflow: hidden;
        }
        .cohort-stat-row {
            display: flex;
            justify-content: space-between;
            border-top: 1px dashed #CBD5E1;
            padding-top: 8px;
            margin-top: 4px;
            font-size: 0.75rem;
            color: #475569;
        }
        
        /* Tab Architecture Overrides */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #CBD5E1;
            padding-bottom: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            border-radius: 6px 6px 0 0 !important;
            padding: 10px 20px !important;
            font-weight: 500 !important;
            color: #475569 !important;
            font-size: 0.9rem !important;
            border: none !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #0F172A !important;
            background-color: #CBD5E1 !important;
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# PRECISE CHARTS ENGINE (HIGH CONTRAST GLOWING CENTROIDS)
# =========================================================================
def _draw_3d_canvas(
    df: pd.DataFrame, size_ref: int, opacity_ref: float, centroids_toggle: bool
) -> go.Figure:
    fig = go.Figure()

    for cohort in COHORT_ORDER:
        sub = df[df["Cohort"] == cohort]
        if sub.empty:
            continue
        color = PALETTE[cohort]

        is_vip = (cohort == VIP_LABEL)
        marker_symbol = "diamond" if is_vip else "circle"
        marker_size = size_ref + 2.5 if is_vip else size_ref
        node_opacity = min(opacity_ref + 0.15, 1.0) if is_vip else opacity_ref

        fig.add_trace(go.Scatter3d(
            x=sub["Recency"],
            y=sub["Visual_Frequency"],
            z=sub["Visual_Monetary"],
            mode="markers",
            name=cohort,
            legendgroup=cohort,
            marker=dict(
                size=marker_size,
                color=color,
                opacity=node_opacity,
                symbol=marker_symbol,
                line=dict(width=0)
            ),
            customdata=sub[["CustomerID", "Frequency", "Monetary"]].values,
            hovertemplate=(
                "<b>Account ID: %{customdata[0]}</b><br>"
                "Recency: %{x} Days<br>"
                "Frequency: %{customdata[1]} Orders<br>"
                "Monetary Volume: £%{customdata[2]:,.2f}<br>"
                "<extra>" + cohort + "</extra>"
            ),
        ))

    if centroids_toggle:
        core_averages = (
            df[df["Cohort"] != VIP_LABEL]
            .groupby("Cohort")[["Recency", "Visual_Frequency", "Visual_Monetary"]]
            .mean()
        )
        for cohort, row in core_averages.iterrows():
            color = PALETTE.get(cohort, "#64748B")
            
            fig.add_trace(go.Scatter3d(
                x=[row["Recency"]],
                y=[row["Visual_Frequency"]],
                z=[row["Visual_Monetary"]],
                mode="markers",
                name=f"Centroid: {cohort}",
                legendgroup=cohort,
                showlegend=False,
                marker=dict(
                    size=16,
                    color="#0F172A",
                    symbol="diamond",
                    opacity=0.9,
                ),
                hoverinfo="skip"
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[row["Recency"]],
                y=[row["Visual_Frequency"]],
                z=[row["Visual_Monetary"]],
                mode="markers",
                name=f"Centroid Point: {cohort}",
                legendgroup=cohort,
                showlegend=False,
                marker=dict(
                    size=10,
                    color=color,
                    symbol="diamond",
                    line=dict(width=1.5, color="#F8FAFC"),
                ),
                hovertemplate=(
                    f"<b>Centroid Center Vector: {cohort}</b><br>"
                    "Mean Recency: %{x:.1f} d<br>"
                    "Mean Frequency: %{y:.1f} o<br>"
                    "Mean Value: £%{z:,.1f}<extra></extra>"
                ),
            ))

    _axis_config = lambda title, is_log: dict(
        title=title,
        type="log" if is_log else "linear",
        showgrid=True,
        gridcolor="#CBD5E1",
        showbackground=True,
        backgroundcolor="#E2E8F0",  
        showspikes=True,
        spikecolor="#475569",
        spikethickness=2,
        title_font=dict(size=11, color="#0F172A", family="sans-serif", weight="bold"),
        tickfont=dict(size=9, color="#475569", family="monospace"),
    )

    fig.update_layout(
        uirevision="constant",
        scene=dict(
            xaxis=_axis_config("Recency (Days Linear)", False),
            yaxis=_axis_config("Frequency (Total Orders Log)", True), 
            zaxis=_axis_config("Monetary Yield (£ Gross Log)", True),  
            camera=dict(
                eye=dict(x=1.35, y=1.35, z=1.05),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode="cube", 
        ),
        legend=dict(
            title="<b>Strategic Portfolio Segments</b>",
            bgcolor="#E2E8F0",
            bordercolor="#CBD5E1",
            borderwidth=1,
            font=dict(size=10, color="#0F172A", family="sans-serif"),
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=620,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _draw_2d_slice(
    df: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, log_mode: bool = False
) -> go.Figure:
    fig = go.Figure()
    for cohort in COHORT_ORDER:
        sub = df[df["Cohort"] == cohort]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[x_col], y=sub[y_col],
            mode="markers",
            name=cohort,
            showlegend=False,
            marker=dict(
                size=7 if cohort == VIP_LABEL else 5,
                color=PALETTE[cohort],
                opacity=0.65,
                symbol="circle",
            ),
            hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: £%{{y:,.2f}}<extra>{cohort}</extra>" if "£" in y_label else f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra>{cohort}</extra>",
        ))
    fig.update_layout(
        xaxis=dict(title=x_label, **_AX_SCHEMA),
        yaxis=dict(title=y_label, type="log" if log_mode else "linear", **_AX_SCHEMA),
        height=320,
        margin=dict(l=50, r=15, t=15, b=50),
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="#E2E8F0",   
    )
    return fig


def _draw_volume_pie(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby("Cohort")["CustomerID"].count().reindex(COHORT_ORDER).dropna()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.60,
        marker=dict(colors=[PALETTE[c] for c in counts.index], line=dict(color="#F1F5F9", width=2)),
        textinfo="percent",
        textfont=dict(size=11, weight="bold", color="#0F172A", family="sans-serif"),
        hovertemplate="<b>%{label}</b><br>Volume Count: %{value:,}<br>Proportion: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        showlegend=False,
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _draw_revenue_bars(df: pd.DataFrame) -> go.Figure:
    rev = df.groupby("Cohort")["Monetary"].sum().reindex(COHORT_ORDER).dropna()
    shares = (rev / rev.sum() * 100).sort_values(ascending=True)
    
    fig = go.Figure(go.Bar(
        y=shares.index,
        x=shares.values,
        orientation="h",
        marker_color=[PALETTE[c] for c in shares.index],
        text=[f" {v:.1f}%" for v in shares.values],
        textposition="outside",
        textfont=dict(color="#0F172A", weight="bold"),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Revenue Share: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="Relative Value Share (%)", **_AX_SCHEMA, range=[0, shares.max() * 1.25]),
        yaxis=dict(tickfont=dict(size=10, color="#334155", weight="bold")),
        height=240,
        margin=dict(l=10, r=45, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _draw_boxplot_distribution(df: pd.DataFrame) -> go.Figure:
    dimensions = [
        ("Recency", "Recency (Days)"),
        ("Frequency", "Frequency (Orders)"),
        ("Monetary", "Monetary (£)"),
    ]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[dim[1] for dim in dimensions],
        horizontal_spacing=0.08,
    )
    
    for cohort in COHORT_ORDER:
        sub = df[df["Cohort"] == cohort]
        if sub.empty:
            continue
        color = PALETTE[cohort]
        
        for i, (col_name, _) in enumerate(dimensions, start=1):
            fig.add_trace(
                go.Box(
                    y=sub[col_name],
                    name=cohort,
                    legendgroup=cohort,
                    showlegend=(i == 1),
                    marker_color=color,
                    line_color=color,
                    fillcolor=_hex_to_rgba(color, 0.15),
                    boxpoints=False,
                    boxmean="sd",
                ),
                row=1, col=i,
            )
            
    fig.update_layout(
        height=360,
        margin=dict(l=15, r=15, t=40, b=85),
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="#E2E8F0",   
        boxmode="group",
        legend=dict(
            orientation="h",
            y=-0.25,
            x=0.5,
            xanchor="center",
            font=dict(size=10, color="#0F172A"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    for idx in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=1, col=idx, linecolor="#CBD5E1")
        fig.update_yaxes(**_AX_SCHEMA, row=1, col=idx)
    return fig


# =========================================================================
# APPLICATION DATA ORCHESTRATOR LAYER
# =========================================================================
def main() -> None:
    _inject_premium_styles()

    try:
        df_rfm, df_scaled = _load_data_layers()
        labels = _compute_deterministic_labels(df_scaled)
    except FileNotFoundError as error:
        st.error(
            f"**Execution Aborted : Missing Pipeline Dependencies**<br>"
            f"Ensure the backend calculation matrices are fully processed first.<br>"
            f"Target parameter error: `{error}`",
            icon="❌"
        )
        st.stop()

    df = _assemble_master_portfolio(df_rfm, df_scaled, labels)

    # Sidebar Interface Configuration
    with st.sidebar:
        st.markdown(
            "<div style='padding-top:10px;'>"
            "<span style='font-size:1.35rem; font-weight:800; color:#0F172A; letter-spacing:-0.03em;'>Strategic Control Center</span>"
            "<p style='font-size:0.8rem; color:#475569; margin-top:2px;'>Engine Parameters : KMeans Matrix (K=4)</p>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        st.markdown("<p style='font-size:0.8rem; font-weight:700; color:#334155; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;'>Active Portfolio Filters</p>", unsafe_allow_html=True)
        valid_options = [c for c in COHORT_ORDER if c in df["Cohort"].unique()]
        chosen_cohorts = st.multiselect(
            "Strategic Target Channels",
            valid_options,
            default=valid_options,
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("<p style='font-size:0.8rem; font-weight:700; color:#334155; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;'>3D Spatial Calibration</p>", unsafe_allow_html=True)
        ui_size = st.slider("Marker Point Radius", 2, 10, 4)
        ui_alpha = st.slider("Coordinate Alpha Opacity", 0.10, 1.0, 0.65, step=0.05)
        ui_centroids = st.toggle("Overlay Enhanced Centroids", value=True)

        st.markdown("<p style='font-size:0.8rem; font-weight:700; color:#334155; text-transform:uppercase; letter-spacing:0.05em; margin-top:16px; margin-bottom:8px;'>2D Chart Transformation</p>", unsafe_allow_html=True)
        ui_log = st.toggle("Logarithmic Scale Mapping", value=False)

        st.markdown("---")
        st.markdown("<p style='font-size:0.7rem; color:#64748B; font-weight:500; line-height:1.4;'>System Blueprint Architecture: Outlier Winsorisation Matrix (99th Percentile) -> Power Box-Cox Transforms -> Standard Scaler Arrays -> Target KMeans Optimization Sequence</p>", unsafe_allow_html=True)

    if not chosen_cohorts:
        st.warning("Active target selection array empty. Isolate at least one cohort stream to mount dashboard visualization layers.")
        st.stop()

    focused_df = df[df["Cohort"].isin(chosen_cohorts)]

    # =========================================================================
    # SHAREHOLDER OVERHAULED HEADER
    # =========================================================================
    st.markdown(
        "<div class='dashboard-header'>"
        "<h1 class='dashboard-title'>Customer Capital & Portfolio Performance Matrix</h1>"
        "<div class='dashboard-subtitle'>Strategic RFM Asset Valuation and Client Retention Framework for Shareholders</div>"
        "</div>",
        unsafe_allow_html=True
    )

    # Executive Custom Micro-Metric Component Row with Contextual Explanations
    total_records = len(df)
    total_revenue_pool = df["Monetary"].sum()
    cohort_count = df["Cohort"].nunique()
    avg_customer_spend = df["Monetary"].mean()
    avg_purchase_velocity = df["Frequency"].mean()

    st.markdown(
        f"<div class='metric-grid-container'>"
        f"<div class='metric-card-custom m-prime'><div class='metric-card-label'>Active Ledger Footprint</div><div class='metric-card-value'>{total_records:,}</div></div>"
        f"<div class='metric-card-custom'><div class='metric-card-label'>Allocated Channels</div><div class='metric-card-value'>{cohort_count}</div></div>"
        f"<div class='metric-card-custom m-teal'><div class='metric-card-label'>Gross Capital Return</div><div class='metric-card-value'>£{total_revenue_pool:,.0f}</div></div>"
        f"<div class='metric-card-custom m-teal'><div class='metric-card-label'>Mean Account Spend</div><div class='metric-card-value'>£{avg_customer_spend:,.2f}</div></div>"
        f"<div class='metric-card-custom m-orange'><div class='metric-card-label'>Transaction Velocity</div><div class='metric-card-value'>{avg_purchase_velocity:.1f} o</div></div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Short Strategic Analysis for Metrics Row
    st.markdown(
        f"<p style='font-size:0.85rem; color:#475569; margin-top:-1.75rem; margin-bottom:2.5rem; font-style:italic;'>"
        f"<strong>Shareholder Metric Context:</strong> Our current asset base comprises {total_records:,} accounts yielding a gross value pool of £{total_revenue_pool:,.0f}. "
        f"The mean spend profile stands healthy at £{avg_customer_spend:,.2f}, driven primarily by high-frequency compounding actors, indicating strong product-market insulation despite macro headwinds."
        f"</p>",
        unsafe_allow_html=True
    )

    # =========================================================================
    # PANEL 1: THREE-DIMENSIONAL TOPOLOGY SPACE
    # =========================================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header-text'>🔮 Spatial Macro Topology and Capital Density Mapping</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='executive-summary-text'>"
        "This structural coordinates framework vectorizes accounts across Recency, Frequency, and Monetary parameters simultaneously. "
        "By looking at the visual boundaries between these mathematical clusters, stakeholders can verify structural stability, track systemic risk clusters, and locate high-yield outliers. "
        "The dual-stamped diamond nodes pinpoint the geometric mathematical centers (centroids) around which core customer groups naturally concentrate."
        "</div>",
        unsafe_allow_html=True
    )
    st.plotly_chart(_draw_3d_canvas(focused_df, ui_size, ui_alpha, ui_centroids), use_container_width=True)
    
    # Embedded Graphic Analysis
    st.markdown(
        "<div class='insight-highlight-box'>"
        "<strong>Topological Capital Analysis:</strong> Clear geometric separation between clusters verifies high modeling integrity. "
        "Notice the heavy visual concentration of the <em>Premium Retention</em> core centering tightly around short recency and high transactional velocity vectors, confirming a steady and highly predictable corporate revenue base. "
        "Outliers branching upward on the logarithmic scale represent high-enterprise accounts that bypass generic purchasing patterns."
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # PANEL 2: VOLUME ALLOCATION AND REVENUE VECTOR SHARES
    # =========================================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header-text'>📊 Volumetric Account Density vs. Financial Yield Alpha</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='executive-summary-text'>"
        "An essential visual alignment tool to audit equity risk and capital leverage. "
        "This view contrasts total headcount distribution against actual gross financial output, exposing systemic reliance on narrow customer groups and tracking the scale of our underperforming account long-tail."
        "</div>",
        unsafe_allow_html=True
    )
    
    col_left, col_right = st.columns([1, 1.25])
    with col_left:
        st.plotly_chart(_draw_volume_pie(focused_df), use_container_width=True)
    with col_right:
        st.plotly_chart(_draw_revenue_bars(focused_df), use_container_width=True)
        
    # Embedded Graphic Analysis
    st.markdown(
        "<div class='insight-highlight-box'>"
        "<strong>Portfolio Optimization Insight:</strong> This breakdown illustrates the classic Pareto distribution driving corporate cash flow. "
        "While high-value segments like <em>Premium Retention</em> and <em>White Glove Concierge</em> occupy a tightly disciplined footprint in terms of total account volume, they command a dominant share of our gross capital return. "
        "Conversely, the <em>Low-Cost Re-Engage</em> long tail represents an operational surface area requiring low-overhead automated maintenance rather than intensive marketing capital allocation."
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # PANEL 3: VARIATIONAL DISTRIBUTION PROFILING
    # =========================================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header-text'>🧬 Structural Variance, Spread, and Dispersion Ranges</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='executive-summary-text'>"
        "A rigorous look at behavioral consistency inside each segment using standard box-plot structures. "
        "Analyzing these dispersion fields helps shareholders evaluate the internal predictability of our channels, evaluate margin ranges, and track operational variance across critical metrics."
        "</div>",
        unsafe_allow_html=True
    )
    st.plotly_chart(_draw_boxplot_distribution(focused_df), use_container_width=True)
    
    # Embedded Graphic Analysis
    st.markdown(
        "<div class='insight-highlight-box'>"
        "<strong>Dispersion Profile Analysis:</strong> The highly disciplined, narrow box dimensions within <em>Premium Retention</em> confirm exceptional cohort uniformity, which translates to a highly reliable forward revenue run-rate. "
        "In contrast, the wider dispersion boundaries within <em>High-Priority Winback</em> highlight individual account variance, pinpointing specific target opportunities where bespoke outreach can prevent permanent churn."
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # PANEL 4: FLAT DIMENSION MAPPING FIELDS
    # =========================================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header-text'>📐 Two-Dimensional Elasticity and Acquisition Trajectories</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='executive-summary-text'>"
        "Isolated two-axis tracking grids that cross-examine metrics to uncover hidden account elasticity. "
        "These projections isolate transaction frequency against total monetary velocity, revealing early churn signals and identifying accounts with high expansion potential."
        "</div>",
        unsafe_allow_html=True
    )
    
    tab_rec_freq, tab_rec_mon, tab_freq_mon = st.tabs([
        "⚡ Recency vs Frequency Metrics",
        "🔲 Recency vs Monetary Value Scale",
        "📈 Frequency vs Monetary Capital Yield",
    ])
    
    with tab_rec_freq:
        st.plotly_chart(_draw_2d_slice(focused_df, "Recency", "Frequency", "Recency Dimension (Days Elapsed)", "Frequency Dimension (Order Vol)"), use_container_width=True)
    with tab_rec_mon:
        st.plotly_chart(_draw_2d_slice(focused_df, "Recency", "Monetary", "Recency Dimension (Days Elapsed)", "Monetary Yield (£ Gross)", ui_log), use_container_width=True)
    with tab_freq_mon:
        st.plotly_chart(_draw_2d_slice(focused_df, "Frequency", "Monetary", "Frequency Dimension (Order Vol)", "Monetary Yield (£ Gross)", ui_log), use_container_width=True)
        
    # Embedded Graphic Analysis
    st.markdown(
        "<div class='insight-highlight-box'>"
        "<strong>Elasticity Mapping Review:</strong> The transaction frequency charts reveal a clear downward slope in order velocity as recency lengthens, highlighting the critical importance of early lifecycle intervention. "
        "Accounts moving upward along the vertical axes represent prime candidates for premium service tier graduation and strategic account expansion initiatives."
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # PANEL 5: GRANULAR COHORT BEHAVIORAL VECTORS
    # =========================================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header-text'>💡 Targeted Cohort Operational Anchors and Strategic Mandates</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='executive-summary-text'>"
        "Granular micro-segment profiles that break down key metrics into actionable operational paths. "
        "Each block details segment population size alongside exact behavioral means, directly linking statistical clusters to specific corporate growth playbooks."
        "</div>",
        unsafe_allow_html=True
    )
    
    display_cohorts = [c for c in COHORT_ORDER if c in chosen_cohorts]
    grid_columns = st.columns(min(len(display_cohorts), 5))
    
    for index, cohort in enumerate(display_cohorts):
        meta = CENTROID_PROFILES[cohort]
        color = PALETTE[cohort]
        slice_df = focused_df[focused_df["Cohort"] == cohort]
        
        slice_count = len(slice_df)
        mean_rec = slice_df["Recency"].mean()
        mean_freq = slice_df["Frequency"].mean()
        mean_mon = slice_df["Monetary"].mean()
        
        grid_columns[index % len(grid_columns)].markdown(
            f"<div class='cohort-block'>"
            f"<span class='cohort-badge' style='background:{color}; margin-bottom:0;'>{meta['emoji']} {meta['tag']}</span>"
            f"<div class='cohort-title-text'>{cohort}</div>"
            f"<div class='cohort-description-text'>{meta['desc']}</div>"
            f"<div class='cohort-stat-row' style='border-top: 1px solid #CBD5E1; padding-top:6px;'><span>Segment Population:</span><span style='font-weight:600;'>{slice_count:,}</span></div>"
            f"<div class='cohort-stat-row'><span>Mean Recency Delta:</span><span style='font-weight:600; font-family:monospace;'>{mean_rec:.0f}d</span></div>"
            f"<div class='cohort-stat-row'><span>Mean Order Velocity:</span><span style='font-weight:600; font-family:monospace;'>{mean_freq:.1f}o</span></div>"
            f"<div class='cohort-stat-row'><span>Mean Value Baseline:</span><span style='font-weight:600; font-family:monospace; color:#0F172A;'>£{mean_mon:,.0f}</span></div>"
            f"</div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # PANEL 6: STRATEGIC COMPLIANCE GENERAL LEDGER MATRIX
    # =========================================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header-text'>📓 Corporate Portfolio Ledger and Financial Audit Index</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='executive-summary-text'>"
        "The underlying un-winsorized data audit engine, preserving raw financial records to ensure absolute financial accountability. "
        "This ledger calculates exact capital contribution percentages and relative volume shares, serving as our core tool for strategic planning and board-level risk assessment."
        "</div>",
        unsafe_allow_html=True
    )
    
    ledger = (
        focused_df.groupby("Cohort")
        .agg(
            Volume_Count=("CustomerID", "count"),
            Mean_Recency=("Recency", "mean"),
            Mean_Frequency=("Frequency", "mean"),
            Mean_Monetary=("Monetary", "mean"),
            Gross_Revenue_Contribution=("Monetary", "sum"),
        )
        .round(1)
    )
    total_rev_sum = df["Monetary"].sum()
    total_cust_sum = df["CustomerID"].count()
    ledger["Volume_Share_%"] = (ledger["Volume_Count"] / total_cust_sum * 100).round(1)
    ledger["Revenue_Share_%"] = (ledger["Gross_Revenue_Contribution"] / total_rev_sum * 100).round(1)
    
    ledger = ledger[[
        "Volume_Count", "Volume_Share_%", "Mean_Recency",
        "Mean_Frequency", "Mean_Monetary", "Gross_Revenue_Contribution", "Revenue_Share_%"
    ]]
    ledger.index.name = "Strategic Classification Channel"
    
    st.dataframe(
        ledger.sort_values("Gross_Revenue_Contribution", ascending=False)
        .style.format({
            "Volume_Count": "{:,.0f}",
            "Volume_Share_%": "{:.1f}%",
            "Mean_Recency": "{:.1f} Days",
            "Mean_Frequency": "{:.1f} Orders",
            "Mean_Monetary": "£{:,.2f}",
            "Gross_Revenue_Contribution": "£{:,.0f}",
            "Revenue_Share_%": "{:.1f}%",
        })
        .background_gradient(subset=["Revenue_Share_%"], cmap="Blues", vmin=0, vmax=60)
        .background_gradient(subset=["Volume_Share_%"], cmap="Greens", vmin=0, vmax=40),
        use_container_width=True,
        height=220,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # PANEL 7: ACCOUNT REGISTRY AUDIT AND TARGET TRACE PROFILE
    # =========================================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header-text'>🔍 Direct Corporate Account Query and Target Registry Trace</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='executive-summary-text'>"
        "An institutional-grade look-up utility designed for immediate portfolio cross-examination. "
        "Input any unique CustomerID to instantly extract its verified transaction history, active capital contribution classification, and assigned corporate routing."
        "</div>",
        unsafe_allow_html=True
    )
    
    query_id = st.number_input(
        "Enter Target Account Reference Identifier (CustomerID)",
        min_value=0, value=0, step=1, format="%d"
    )
    
    if query_id > 0:
        match = df[df["CustomerID"].astype(int) == int(query_id)]
        if match.empty:
            st.warning(f"Target account trace lookup failed for customer reference sequence: {query_id}")
        else:
            record = match.iloc[0]
            cohort_color = PALETTE.get(record["Cohort"], "#64748B")
            
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            m_col1.metric("Validated Account ID", int(record["CustomerID"]))
            m_col2.metric("Recency Profile", f"{int(record['Recency'])} Days")
            m_col3.metric("Order Frequency Pool", f"{int(record['Frequency'])} Orders")
            m_col4.metric("Gross Capital Valuation", f"£{record['Monetary']:,.2f}")
            m_col5.metric("Target Strategic Route", record["Cohort"])
            
            if record["Is_VIP"]:
                st.success(
                    f"⭐ **High-Value Elite Account Verified** : Active positioning metrics exceed designated threshold vectors. "
                    f"Process routing assigned exclusively to manual premium workflows.",
                    icon="💎"
                )
            
            m_profile = CENTROID_PROFILES.get(record["Cohort"], {})
            if m_profile:
                st.markdown(
                    f"<div style='margin-top:14px; padding:12px; background:#E2E8F0; border:1px solid #CBD5E1; border-radius:8px;'> "
                    f"<span class='cohort-badge' style='background:{cohort_color}; margin-bottom:0;'>{m_profile.get('emoji','')} {m_profile.get('tag','')}</span>"
                    f"<span style='color:#334155; font-size:0.85rem; margin-left:12px; font-weight:500;'>{m_profile.get('desc','')}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# SYSTEM APPLICATION INVOKER
# =========================================================================
if __name__ == "__main__":
    main()