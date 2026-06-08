# =========================================================================
# src/app.py : Executive RFM Customer Segmentation Intelligence Hub
# =========================================================================
# Run Command: streamlit run src/app.py
#
# Code Architecture Specifications:
#   - Zero Em Dash Compliance (No structural em dash characters utilized)
#   - Premium Non-White Canvas Architecture (Seamless Executive Matte)
#   - Core Currency Anchoring: United Kingdom (GBP / £)
#   - Enhanced Geometric Centroid Mapping Layer
#   - Midnight Emerald & Copper Premium Palette Configuration
#   - Pure CSS Grid Layout Engine (Eliminates Streamlit Column Wrapping Faults)
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

# Streamlined Enterprise Profiles optimized for rapid scanning and operational cognition
CENTROID_PROFILES: dict[str, dict] = {
    "Premium Retention": {
        "emoji": "👑", 
        "tag": "Core Revenue Engine", 
        "desc": "High-frequency loyalists anchoring corporate cash flow with stable recurring yield."
    },
    "Low-Cost Re-Engage": {
        "emoji": "💤", 
        "tag": "Dormant Base Tail", 
        "desc": "Inactive, low-value client tail. Best targeted via automated, low-overhead workflows."
    },
    "Nurture & Upsell": {
        "emoji": "📈", 
        "tag": "Growth Pipeline", 
        "desc": "Recently active accounts with positive engagement signals. Prime cross-sell targets."
    },
    "High-Priority Winback": {
        "emoji": "🚨", 
        "tag": "Capital At Risk", 
        "desc": "Previously high-value customers showing severe inactivity risk. Urgent recovery required."
    },
    VIP_LABEL: {
        "emoji": "💎", 
        "tag": "High Net Worth Outliers", 
        "desc": "Elite high-net-worth accounts driving disproportionate revenue shares."
    },
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

        /* Collapse the Streamlit flex-gap that appears after an st.columns block.
           Targets the element-container wrapper (the actual flex child) rather
           than the inner div, which is the only level where negative margin can
           override a flex parent's gap property. */
        [data-testid="element-container"]:has(.executive-ledger-desc) {
            margin-top: -1rem !important;
        }

        /* Tighten the description-to-table gap. The base executive-summary-text
           carries 1.5rem margin-bottom; this override brings it down to 0.75rem
           specifically for the ledger description so it sits closer to the table. */
        .executive-ledger-desc {
            margin-bottom: 0.75rem !important;
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
            flex-wrap: wrap;
            gap: 16px;
            margin-bottom: 2.5rem;
            width: 100%;
        }
        .metric-card-custom {
            flex: 1;
            min-width: 200px;
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
        /* Left Accent Border Modifiers */
        .metric-card-custom.m-slate::before { background: #475569; }
        .metric-card-custom.m-prime::before { background: #0F766E; }
        .metric-card-custom.m-teal::before { background: #0284C7; }
        .metric-card-custom.m-orange::before { background: #D97706; }
        .metric-card-custom.m-indigo::before { background: #4F46E5; } /* Add this line */
        
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
            white-space: normal !important;
            word-break: break-word !important;
        }

        /* =========================================================================
           LOOKUP GRID CARD ARCHITECTURE (UNIFIED 3-COLUMN RHYTHM)
           ========================================================================= */
        .lookup-grid-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            width: 100%;
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        .lookup-span-2 {
            grid-column: span 2;
        }
        
        /* High-Contrast Accent Border Modifiers */
        .metric-card-custom.m-slate::before { 
            background: #475569; 
        }
        
        /* Whitespace Gap Reductions */
        .lookup-ledger-desc {
            margin-bottom: 0.5rem !important;
        }
        
        [data-testid="element-container"]:has(.lookup-grid-container) {
            margin-top: -0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* =========================================================================
           REFINED COHORT GRID CARD ARCHITECTURE (TRUE CSS GRID ENGINE)
           ========================================================================= */
        .cohort-grid-layout {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            width: 100%;
        }
        
        @media (max-width: 1400px) {
            .cohort-grid-layout {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        @media (max-width: 1024px) {
            .cohort-grid-layout {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        @media (max-width: 640px) {
            .cohort-grid-layout {
                grid-template-columns: 1fr;
            }
        }

        .cohort-block {
            background: #F8FAFC;
            border: 1px solid #CBD5E1;
            border-radius: 18px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            gap: 18px;
            height: 100%;
        }

        .cohort-badge {
            display: inline-flex;
            align-items: center;
            width: fit-content;
            padding: 8px 14px;
            border-radius: 10px;
            color: white;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.3px;
        }

        .cohort-title-text {
            font-size: 20px;
            font-weight: 700;
            line-height: 1.4;
            color: #0F172A;
            margin-top: 4px;
        }

        .cohort-description-text {
            font-size: 15px;
            line-height: 1.6;
            color: #475569;
            min-height: 120px;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 5;
            -webkit-box-orient: vertical;
        }

        .cohort-stat-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 18px;
            align-items: start;
            padding: 12px 0;
            border-top: 1px dashed #CBD5E1;
            font-size: 15px;
            color: #475569;
        }

        .cohort-stat-row span:last-child {
            text-align: right;
            white-space: nowrap;
            color: #334155;
            font-weight: 700;
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
        
        /* Sidebar Description Typography Overrides */
        .sidebar-caption-text {
            font-size: 0.78rem;
            color: #475569;
            line-height: 1.4;
            margin-top: -6px;
            margin-bottom: 14px;
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

    # Pre-compute explicit log coordinates to completely bypass Plotly's 3D log axis mirror bug
    df_plot = df.copy()
    df_plot["Plot_Frequency"] = np.log10(df_plot["Visual_Frequency"])
    df_plot["Plot_Monetary"] = np.log10(df_plot["Visual_Monetary"])

    for cohort in COHORT_ORDER:
        sub = df_plot[df_plot["Cohort"] == cohort]
        if sub.empty:
            continue
        color = PALETTE[cohort]

        is_vip = (cohort == VIP_LABEL)
        marker_symbol = "diamond" if is_vip else "circle"
        marker_size = size_ref + 2 if is_vip else size_ref
        node_opacity = min(opacity_ref + 0.1, 1.0) if is_vip else opacity_ref

        fig.add_trace(go.Scatter3d(
            x=sub["Recency"],
            y=sub["Plot_Frequency"],
            z=sub["Plot_Monetary"],
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
        # Compute spatial center vectors directly within the visual display coordinate metrics
        core_averages = (
            df_plot[df_plot["Cohort"] != VIP_LABEL]
            .groupby("Cohort")[["Recency", "Plot_Frequency", "Plot_Monetary"]]
            .mean()
        )
        for cohort, row in core_averages.iterrows():
            color = PALETTE.get(cohort, "#64748B")
            
            # High-contrast solid geometric anchors with a unified dark border to remove depth ambiguity
            fig.add_trace(go.Scatter3d(
                x=[row["Recency"]],
                y=[row["Plot_Frequency"]],
                z=[row["Plot_Monetary"]],
                mode="markers",
                name=f"Centroid Center: {cohort}",
                legendgroup=cohort,
                showlegend=False,
                marker=dict(
                    size=11,
                    color=color,
                    symbol="diamond",
                    line=dict(width=2, color="#0F172A"),
                ),
                customdata=np.array([[
                    10**row["Plot_Frequency"], 
                    10**row["Plot_Monetary"]
                ]]),
                hovertemplate=(
                    f"<b>Centroid Center Vector: {cohort}</b><br>"
                    "Mean Recency: %{x:.1f} Days<br>"
                    "Mean Frequency: %{customdata[0]:.1f} Orders<br>"
                    "Mean Value: £%{customdata[1]:,.1f}<extra></extra>"
                ),
            ))

    # Explicitly map order-of-magnitude bounds to align cleanly with the visual point clouds
    y_ticks = [0, 1, 2, 3]
    y_texts = ["1", "10", "100", "1k"]
    
    # Bottom bound begins at -1 to capture the 0.1 monetary clip boundary cleanly
    z_ticks = [-1, 0, 1, 2, 3, 4, 5, 6]
    z_texts = ["£0.1", "£1", "£10", "£100", "£1k", "£10k", "£100k", "£1M"]

    _axis_config = lambda title, tickvals=None, ticktext=None: dict(
        title=title,
        type="linear",
        showgrid=True,
        gridcolor="#CBD5E1",        
        showbackground=False,       
        showspikes=False,           
        tickvals=tickvals,
        ticktext=ticktext,
        title_font=dict(size=10, color="#475569", family="sans-serif", weight="bold"),
        tickfont=dict(size=8, color="#64748B", family="monospace"),
    )

    fig.update_layout(
        uirevision="constant",
        scene=dict(
            xaxis=_axis_config("Recency (Days)"),
            yaxis=_axis_config("Frequency (Orders Scale)", y_ticks, y_texts), 
            zaxis=_axis_config("Monetary Yield (Capital Value)", z_ticks, z_texts),  
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.1),
                up=dict(x=0, y=0, z=1)
            ),
            # Transitioned from cube to manual aspect layout to safely ground extreme monetary spikes
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.75),
        ),
        legend=dict(
            title="<b>Portfolio Segments</b>",
            bgcolor="rgba(248,250,252,0.85)",
            bordercolor="#CBD5E1",
            borderwidth=1,
            font=dict(size=10, color="#334155", family="sans-serif"),
            itemsizing="constant",
            orientation="h",        
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=30, b=0),
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
        
        is_vip = (cohort == VIP_LABEL)
        fig.add_trace(go.Scatter(
            x=sub[x_col], y=sub[y_col],
            mode="markers",
            name=cohort,
            showlegend=False,
            marker=dict(
                size=5.5 if is_vip else 3.5,
                color=PALETTE[cohort],
                opacity=0.45 if not is_vip else 0.75,
                symbol="circle",
                line=dict(width=0)
            ),
            hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: £%{{y:,.2f}}<extra>{cohort}</extra>" if "£" in y_label else f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra>{cohort}</extra>",
        ))
        
    # Isolate the base Y-axis configuration
    y_axis_config = dict(title=y_label, type="log" if log_mode else "linear", **_AX_SCHEMA)
    
    # Force clean integer formatting for log axes to prevent stripped zeros (like 20 becoming 2)
    if log_mode:
        y_axis_config["tickformat"] = "d"
        
    # Conditionally inject the currency prefix if the axis tracks capital/monetary metrics
    if "£" in y_label or "Monetary" in y_label:
        y_axis_config["tickprefix"] = "£"
        # Using pure currency shorthand formatting compatible with log scales
        y_axis_config["tickformat"] = "~s"
        
    fig.update_layout(
        xaxis=dict(title=x_label, **_AX_SCHEMA),
        yaxis=y_axis_config,
        height=340,
        margin=dict(l=55, r=15, t=15, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _draw_volume_pie(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby("Cohort")["CustomerID"].count().reindex(COHORT_ORDER).dropna()
    total_accounts = counts.sum()
    
    # Inject an explicit HTML line break for long corporate tier names
    # This prevents horizontal clipping by shifting text to a vertical layout
    display_labels = [str(label).replace(" Glove ", " Glove<br>") for label in counts.index]
    
    fig = go.Figure(go.Pie(
        labels=display_labels,
        values=counts.values,
        hole=0.60,
        marker=dict(colors=[PALETTE[c] for c in counts.index], line=dict(color="#F1F5F9", width=2)),
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=10, weight="bold", color="#0F172A", family="sans-serif"),
        hovertemplate="<b>%{label}</b><br>Volume Count: %{value:,}<br><extra></extra>",
    ))
    
    fig.add_annotation(
        text=f"<span style='font-size:9px;color:#475569;font-weight:bold;'>TOTAL BASE</span><br><span style='font-size:16px;color:#0F172A;font-weight:800;'>{total_accounts:,}</span>",
        showarrow=False,
        align="center"
    )
    
    fig.update_layout(
        showlegend=False,
        height=260,
        # Symmetrical margins optimized for line-broken text layouts
        margin=dict(l=70, r=70, t=25, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _draw_revenue_bars(df: pd.DataFrame) -> go.Figure:
    rev = df.groupby("Cohort")["Monetary"].sum()
    shares = (rev / rev.sum() * 100)
    # Sort descending so the highest revenue generator anchors the top bar
    shares = shares.sort_values(ascending=True) 
    
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


def _draw_boxplot_distribution(df: pd.DataFrame, log_mode: bool = False) -> go.Figure:
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
                    boxpoints="outliers",  # Keeps outliers clear across both linear and log views
                    boxmean="sd",          # Displays mean and standard deviation markers
                ),
                row=1, col=i,
            )
            
    fig.update_layout(
        height=360,
        margin=dict(l=55, r=15, t=40, b=85), # Left margin clears y-axis labels safely
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",   # Transparent to float over application canvas
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
        
        # Isolate a unique dictionary configuration per axis from global schema
        y_config = {**_AX_SCHEMA}
        
        # Apply conditional scaling properties based on the state of the control toggle
        if idx == 2:    # Frequency Distribution Axis
            y_config["type"] = "log" if log_mode else "linear"
            y_config["tickformat"] = "d"   # Clean integer formatting
            
        elif idx == 3:  # Monetary Distribution Axis
            y_config["type"] = "log" if log_mode else "linear"
            y_config["tickprefix"] = "£"   # Global currency identifier
            y_config["tickformat"] = "~s"  # Clean shorthand scaling (e.g., £10k, £600k)
            
        fig.update_yaxes(**y_config, row=1, col=idx)
        
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

    # =========================================================================
    # SHAREHOLDER CONTROL PANEL (SIDEBAR)
    # =========================================================================
    with st.sidebar:
        st.markdown(
            "<div style='padding-top:10px;'>"
            "<span style='font-size:1.35rem; font-weight:800; color:#0F172A; letter-spacing:-0.03em;'>Portfolio Controls</span>"
            "<p style='font-size:0.8rem; color:#475569; margin-top:2px;'>Governance State : Certified 4-Cohort Model Core</p>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        # 1. Cohort Channel Selection Controls
        st.markdown("<p style='font-size:0.8rem; font-weight:700; color:#334155; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;'>Channel Exposure Filters</p>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-caption-text'>Select or isolate specific customer capital streams to dynamically evaluate risk concentrations and portfolio exposure margins.</div>", unsafe_allow_html=True)
        
        valid_options = [c for c in COHORT_ORDER if c in df["Cohort"].unique()]
        chosen_cohorts = st.multiselect(
            "Strategic Target Channels",
            valid_options,
            default=valid_options,
            label_visibility="collapsed"
        )

        st.markdown("---")
        
        # 2. Macro Topology Space Resolution Configuration
        st.markdown("<p style='font-size:0.8rem; font-weight:700; color:#334155; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;'>Macro Space Resolution</p>", unsafe_allow_html=True)
        
        ui_size = st.slider("Account Volumetric Focus Scope", 2, 10, 4)
        st.markdown("<div class='sidebar-caption-text'>Alters point geometry diameters to distinguish dense macro transaction regions from micro outlier accounts.</div>", unsafe_allow_html=True)
        
        ui_alpha = st.slider("Interdependence Layer Contrast", 0.10, 1.0, 0.65, step=0.05)
        st.markdown("<div class='sidebar-caption-text'>Modulates node opacity to locate hidden volume overlapping inside high density cross sections.</div>", unsafe_allow_html=True)
        
        ui_centroids = st.toggle("Overlay Behavioral Performance Centers", value=True)
        st.markdown("<div class='sidebar-caption-text'>Projects double layered anchors tracing the mathematical averages around which core segments concentrate.</div>", unsafe_allow_html=True)

        st.markdown("---")
        
        # 3. Variance Scaling Transformation Options
        st.markdown("<p style='font-size:0.8rem; font-weight:700; color:#334155; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;'>Variance Normalization</p>", unsafe_allow_html=True)
        
        ui_log = st.toggle("Activate Logarithmic Value Normalization", value=False)
        st.markdown("<div class='sidebar-caption-text'>Applies logarithmic scaling to compress vertical revenue variance, ensuring lower spend categories remain visible alongside top tier anomalies.</div>", unsafe_allow_html=True)

        st.markdown("---")
        
        # 4. Institutional Pipeline Governance Blueprint Record
        st.markdown(
            "<p style='font-size:0.72rem; color:#475569; font-weight:600; line-height:1.4; background:#CBD5E1; padding:10px; border-radius:6px; border:1px solid #94A3B8;'>"
            "📊 <strong>Audit Processing Chain:</strong><br>"
            "Winsorisation (99th Percentile Outlier Cap) → Power Box-Cox Feature Correction → Multi-Axis Standard Scaler Vectors → Deterministic K-Means Core Optimisation."
            "</p>", 
            unsafe_allow_html=True
        )

    if not chosen_cohorts:
        st.warning("Active target selection array empty. Isolate at least one cohort stream to mount dashboard visualization layers.")
        st.stop()

    focused_df = df[df["Cohort"].isin(chosen_cohorts)]

    # =========================================================================
    # HEADER SPACE
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
        f"<div class='metric-card-custom m-orange'><div class='metric-card-label'>Transaction Velocity</div><div class='metric-card-value'>{avg_purchase_velocity:.1f} Orders</div></div>"
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
    
    # Balanced layout allows both data graphics to map cleanly at standard zoom scales
    col_left, col_right = st.columns([1.1, 1.1])
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
    st.plotly_chart(_draw_boxplot_distribution(focused_df, ui_log), use_container_width=True)
    
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
    
    tab_rec_freq, tab_rec_mon, tab_freq_mon = st.tabs([
        "⚡ Recency (Days Elapsed) vs Frequency (Order Volume)",
        "🔲 Recency (Days Elapsed) vs Monetary Yield (£ Gross)",
        "📈 Frequency (Order Volume) vs Monetary Yield (£ Gross)",
    ])
    
    with tab_rec_freq:
        st.markdown(
            "<div class='executive-summary-text'>"
            "Isolated two-axis tracking grid that cross-examines Recency (Days Elapsed) against Frequency (Order Volume) "
            "to evaluate account engagement velocity over time. This isolated layout identifies micro operational "
            "churn patterns before capital degradation occurs."
            "</div>",
            unsafe_allow_html=True
        )
        # Passed ui_log here to allow the layout to unfold compressed transactional volume layers
        st.plotly_chart(_draw_2d_slice(focused_df, "Recency", "Frequency", "Recency (Days Elapsed)", "Frequency (Order Volume)", ui_log), use_container_width=True)
        st.markdown(
            "<div class='insight-highlight-box'>"
            "<strong>Recency vs Frequency Review:</strong> The transaction frequency charts reveal a clear downward slope "
            "in order velocity as recency lengthens, highlighting the critical importance of early lifecycle intervention. "
            "Maintaining immediate operational contact prevents sudden decay in baseline client activity metrics."
            "</div>",
            unsafe_allow_html=True
        )
        
    with tab_rec_mon:
        st.markdown(
            "<div class='executive-summary-text'>"
            "Isolated two-axis tracking grid that maps Recency (Days Elapsed) against Monetary Yield (£ Gross) "
            "to monitor active corporate capital risk distributions. This isolated configuration detects high-value accounts "
            "experiencing momentum loss."
            "</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(_draw_2d_slice(focused_df, "Recency", "Monetary", "Recency (Days Elapsed)", "Monetary Yield (£ Gross)", ui_log), use_container_width=True)
        st.markdown(
            "<div class='insight-highlight-box'>"
            "<strong>Recency vs Monetary Yield Review:</strong> Highest gross asset positions maintain tight alignment with "
            "short temporal coordinates. Outliers expanding outward along the linear timeline indicate vulnerable enterprise accounts "
            "holding dormant financial capacity that requires manual corporate outreach."
            "</div>",
            unsafe_allow_html=True
        )
        
    with tab_freq_mon:
        st.markdown(
            "<div class='executive-summary-text'>"
            "Isolated two-axis tracking grid analyzing Frequency (Order Volume) against Monetary Yield (£ Gross) "
            "to gauge client monetization depth and pipeline scaling. This vector matrix isolates purchasing efficiency "
            "from pure historical transaction longevity."
            "</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(_draw_2d_slice(focused_df, "Frequency", "Monetary", "Frequency (Order Volume)", "Monetary Yield (£ Gross)", ui_log), use_container_width=True)
        st.markdown(
            "<div class='insight-highlight-box'>"
            "<strong>Frequency vs Monetary Yield Review:</strong> A definitive upward compounding path verifies strong scaling "
            "elasticity within active tiers. Accounts climbing vertically along this linear sequence represent prime targets "
            "for strategic account graduation programs and structured premium service expansions."
            "</div>",
            unsafe_allow_html=True
        )
        
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # PANEL 5: GRANULAR COHORT BEHAVIORAL VECTORS (TRUE CSS GRID ENGINE)
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
    
    # Render all selected cohort cards inside a true CSS grid layout container
    grid_html = "<div class='cohort-grid-layout'>"
    
    for cohort in display_cohorts:
        meta = CENTROID_PROFILES[cohort]
        color = PALETTE[cohort]
        slice_df = focused_df[focused_df["Cohort"] == cohort]
        
        slice_count = len(slice_df)
        mean_rec = slice_df["Recency"].mean() if not slice_df.empty else 0
        mean_freq = slice_df["Frequency"].mean() if not slice_df.empty else 0
        mean_mon = slice_df["Monetary"].mean() if not slice_df.empty else 0
        
        # Flattened string formatting completely eliminates accidental markdown code blocks
        grid_html += "<div class='cohort-block'>"
        grid_html += f"<span class='cohort-badge' style='background:{color};'>{meta['emoji']} {meta['tag']}</span>"
        grid_html += f"<div class='cohort-title-text'>{cohort}</div>"
        grid_html += f"<div class='cohort-description-text'>{meta['desc']}</div>"
        grid_html += f"<div class='cohort-stat-row'><span>Population:</span><span>{slice_count:,}</span></div>"
        grid_html += f"<div class='cohort-stat-row'><span>Recency:</span><span>{mean_rec:.0f} Days</span></div>"
        grid_html += f"<div class='cohort-stat-row'><span>Order Velocity:</span><span>{mean_freq:.1f} Orders</span></div>"
        grid_html += f"<div class='cohort-stat-row'><span>Value Baseline:</span><span>£{mean_mon:,.0f}</span></div>"
        grid_html += "</div>"
        
    grid_html += "</div>"
    st.markdown(grid_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # PANEL 6: STRATEGIC COMPLIANCE GENERAL LEDGER MATRIX (PREMIUM REDESIGN)
    # =========================================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    
    # Dual-column header architecture to split the corporate title and the sorting controller
    col_ledger_title, col_ledger_sort = st.columns([2.2, 1])
    with col_ledger_title:
        st.markdown("<div class='section-header-text'>📓 Corporate Portfolio Ledger and Financial Audit Index</div>", unsafe_allow_html=True)
    with col_ledger_sort:
        sort_by = st.selectbox(
            "Sort Active Ledger Rows By",
            options=[
                "Gross Revenue Contribution",
                "Volume Count",
                "Volume Share %",
                "Mean Recency",
                "Mean Frequency",
                "Mean Monetary",
                "Revenue Share %",
                "Strategic Channel Name"
            ],
            index=0,
            label_visibility="collapsed"
        )

    # Full-width description rendered outside the columns block so it matches
    # the table width. The executive-ledger-desc class hooks the CSS rule that
    # collapses the Streamlit flex-gap injected after st.columns.
    st.markdown(
        "<div class='executive-summary-text executive-ledger-desc'>"
        "Preserves raw, unadjusted transaction data to ensure absolute audit integrity. "
        "This ledger calculates precise capital contributions and volume shares to guide board-level risk management. "
        "Use the dropdown to sort active cohorts without displacing the anchored summary metrics."
        "</div>",
        unsafe_allow_html=True
    )
    
    # 1. Compute Base Cohort Metrics
    ledger_core = (
        focused_df.groupby("Cohort")
        .agg(
            Volume_Count=("CustomerID", "count"),
            Mean_Recency=("Recency", "mean"),
            Mean_Frequency=("Frequency", "mean"),
            Mean_Monetary=("Monetary", "mean"),
            Gross_Revenue_Contribution=("Monetary", "sum"),
        )
    )
    total_rev_sum = df["Monetary"].sum()
    total_cust_sum = df["CustomerID"].count()
    
    ledger_core["Volume_Share_%"] = (ledger_core["Volume_Count"] / total_cust_sum * 100)
    ledger_core["Revenue_Share_%"] = (ledger_core["Gross_Revenue_Contribution"] / total_rev_sum * 100)
    
    # 2. Execute Streamlit-Driven Sorting to Safeguard Summary Anchoring
    sort_mapping = {
        "Gross Revenue Contribution": "Gross_Revenue_Contribution",
        "Volume Count": "Volume_Count",
        "Volume Share %": "Volume_Share_%",
        "Mean Recency": "Mean_Recency",
        "Mean Frequency": "Mean_Frequency",
        "Mean Monetary": "Mean_Monetary",
        "Revenue Share %": "Revenue_Share_%",
        "Strategic Channel Name": "index"
    }
    
    sort_target = sort_mapping.get(sort_by, "Gross_Revenue_Contribution")
    if sort_target == "index":
        ledger_core = ledger_core.sort_index()
    else:
        # Sort lower values first for elapsed time vectors (Recency); higher values first for productivity
        ascending_flag = True if sort_target == "Mean_Recency" else False
        ledger_core = ledger_core.sort_values(sort_target, ascending=ascending_flag)
        
    # 3. Compute Consolidated Summary Metrics (Combining Averages and Totals)
    summary_volume = focused_df["CustomerID"].count()
    summary_vol_share = (summary_volume / total_cust_sum * 100)
    summary_recency = focused_df["Recency"].mean()
    summary_frequency = focused_df["Frequency"].mean()
    summary_monetary = focused_df["Monetary"].mean()
    summary_revenue = focused_df["Monetary"].sum()
    summary_rev_share = (summary_revenue / total_rev_sum * 100)
    
    # 4. Inject Premium Inline CSS Styling Specifications
    st.markdown(
        """
        <style>
        .executive-ledger-frame {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: #F8FAFC;
            border: 1px solid #CBD5E1;
            border-radius: 14px;
            overflow: hidden;
            margin-top: 8px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        }
        .executive-ledger-frame th {
            background-color: #0F172A;
            color: #F8FAFC;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 10px 10px;
            text-align: left;
            border-bottom: 2px solid #CBD5E1;
        }
        .executive-ledger-frame td {
            padding: 10px 10px;
            font-size: 0.86rem;
            color: #334155;
            border-bottom: 1px solid #E2E8F0;
            vertical-align: middle;
            white-space: nowrap;
        }
        .executive-ledger-frame tbody tr:last-child td {
            border-bottom: none;
        }
        .executive-ledger-frame tbody tr:hover {
            background-color: #F1F5F9;
            transition: background-color 0.15s ease;
        }
        .ledger-summary-row-mix {
            background-color: #E2E8F0 !important;
            font-weight: 700;
        }
        .ledger-summary-row-mix td {
            color: #0F172A !important;
            border-top: 2px solid #CBD5E1;
            border-bottom: none !important;
        }
        .ledger-num-cell {
            text-align: right !important;
            font-family: monospace;
            font-size: 0.86rem !important;
        }
        .cohort-indicator-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
            vertical-align: middle;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # 5. Construct the Structured Presentation Canvas
    table_html = """
    <table class='executive-ledger-frame'>
        <thead>
            <tr>
                <th>Strategic Channel</th>
                <th class='ledger-num-cell'>Volume Count</th>
                <th class='ledger-num-cell'>Volume Share</th>
                <th class='ledger-num-cell'>Mean Recency</th>
                <th class='ledger-num-cell'>Mean Frequency</th>
                <th class='ledger-num-cell'>Mean Monetary</th>
                <th class='ledger-num-cell'>Gross Revenue</th>
                <th class='ledger-num-cell'>Revenue Share</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # Generate core variable row items
    for cohort_name, row in ledger_core.iterrows():
        badge_color = PALETTE.get(cohort_name, "#64748B")
        text_style = "color: #B91C1C; font-weight: 600;" if cohort_name == "High-Priority Winback" else ""
        
        table_html += "<tr>"
        table_html += f"<td><span class='cohort-indicator-dot' style='background-color:{badge_color};'></span><span style='{text_style}'>{cohort_name}</span></td>"
        table_html += f"<td class='ledger-num-cell'>{row['Volume_Count']:,.0f}</td>"
        table_html += f"<td class='ledger-num-cell'>{row['Volume_Share_%']:.1f}%</td>"
        table_html += f"<td class='ledger-num-cell'>{row['Mean_Recency']:.1f} Days</td>"
        table_html += f"<td class='ledger-num-cell'>{row['Mean_Frequency']:.1f} Orders</td>"
        table_html += f"<td class='ledger-num-cell'>£{row['Mean_Monetary']:,.2f}</td>"
        table_html += f"<td class='ledger-num-cell'>£{row['Gross_Revenue_Contribution']:,.0f}</td>"
        table_html += f"<td class='ledger-num-cell' style='font-weight:600; background-color: rgba(2, 132, 199, 0.04);'>{row['Revenue_Share_%']:.1f}%</td>"
        table_html += "</tr>"
        
    # Append the combined consolidated row fixed firmly at the footer boundary
    table_html += f"""
            <tr class='ledger-summary-row-mix'>
                <td>Portfolio Summary</td>
                <td class='ledger-num-cell'>{summary_volume:,.0f}</td>
                <td class='ledger-num-cell'>{summary_vol_share:.1f}%</td>
                <td class='ledger-num-cell'>{summary_recency:.1f} Days</td>
                <td class='ledger-num-cell'>{summary_frequency:.1f} Orders</td>
                <td class='ledger-num-cell'>£{summary_monetary:,.2f}</td>
                <td class='ledger-num-cell'>£{summary_revenue:,.0f}</td>
                <td class='ledger-num-cell'>{summary_rev_share:.1f}%</td>
            </tr>
        </tbody>
    </table>
    """
    
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # PANEL 7: ACCOUNT REGISTRY AUDIT AND TARGET TRACE PROFILE
    # =========================================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header-text'>🔍 Direct Corporate Account Query and Target Registry Trace</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='executive-summary-text lookup-ledger-desc'>"
        "An institutional-grade look-up utility designed for immediate portfolio cross-examination. "
        "Input any unique CustomerID to instantly extract its verified transaction history, active capital contribution classification, and assigned corporate routing."
        "</div>",
        unsafe_allow_html=True
    )
    
    # Premium text input replacement eliminating clunky native increment steppers
    query_id = st.text_input(
        "Enter Target Account Reference Identifier (CustomerID)",
        value="",
        placeholder="Type Customer ID (e.g., 13474)...",
        label_visibility="collapsed",
        key="portfolio_registry_lookup_id" # Guarantees state stability during rerun sequences
    )
    
    if query_id.strip():
        cleaned_id = query_id.strip()
        if cleaned_id.isdigit():
            # Bulletproof type-safe numeric lookup avoiding unhandled exceptions from NaNs or mixed types
            try:
                target_num = int(cleaned_id)
                match = df[pd.to_numeric(df["CustomerID"], errors="coerce") == target_num]
            except (ValueError, TypeError):
                match = pd.DataFrame()
                
            if match.empty:
                st.warning(f"Target account trace lookup failed for customer reference sequence: {cleaned_id}")
            else:
                record = match.iloc[0]
                cohort_name = record["Cohort"]
                
                # Dynamic CSS modifier mapping based directly on the tracked cohort identity
                cohort_style_map = {
                    "Premium Retention": "m-prime",
                    "Low-Cost Re-Engage": "m-slate",
                    "Nurture & Upsell": "m-teal",
                    "High-Priority Winback": "m-orange",
                    VIP_LABEL: "m-indigo"
                }
                route_accent_class = cohort_style_map.get(cohort_name, "m-slate")
                
                # Balanced 3-Column Grid Layout with dynamic left-accent variations
                st.markdown(
                    f"<div class='lookup-grid-container'>"
                    f"<div class='metric-card-custom m-slate'><div class='metric-card-label'>Validated Account ID</div><div class='metric-card-value'>{int(record['CustomerID'])}</div></div>"
                    f"<div class='metric-card-custom m-slate'><div class='metric-card-label'>Recency Profile</div><div class='metric-card-value'>{int(record['Recency'])} Days</div></div>"
                    f"<div class='metric-card-custom m-slate'><div class='metric-card-label'>Order Frequency Pool</div><div class='metric-card-value'>{int(record['Frequency'])} Orders</div></div>"
                    f"<div class='metric-card-custom m-teal'><div class='metric-card-label'>Gross Capital Valuation</div><div class='metric-card-value'>£{record['Monetary']:,.2f}</div></div>"
                    f"<div class='metric-card-custom {route_accent_class} lookup-span-2'><div class='metric-card-label'>Target Strategic Route</div><div class='metric-card-value' style='font-size:1.35rem;'>{cohort_name}</div></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                # Omnichannel Unique Strategic Status Routing Framework
                if cohort_name == VIP_LABEL:
                    st.success(
                        "**High-Value Elite Account Verified** : Active positioning metrics exceed designated threshold vectors. "
                        "Process routing assigned exclusively to manual premium workflows.",
                        icon="💎"
                    )
                elif cohort_name == "Premium Retention":
                    st.success(
                        "**Core Revenue Asset Confirmed** : Account maintains optimal purchasing velocity and high margin stability. "
                        "Process routing assigned to high-touch client health retention cycles.",
                        icon="👑"
                    )
                elif cohort_name == "Nurture & Upsell":
                    st.info(
                        "**High-Potential Growth Account Identified** : Strong frequency signals present with room for margin expansion. "
                        "Process routing assigned to strategic account development and cross-sell tracks.",
                        icon="📈"
                    )
                elif cohort_name == "Low-Cost Re-Engage":
                    st.info(
                        "**Dormant Long-Tail Account Tracked** : Revenue contribution is minimal with extended temporal gaps. "
                        "Process routing assigned to low-overhead automated programmatic marketing sequences.",
                        icon="💤"
                    )
                elif cohort_name == "High-Priority Winback":
                    st.warning(
                        "**At-Risk Historical Account Flags Triggered** : High lifetime valuation paired with critical temporal decay. "
                        "Process routing assigned to urgent proactive recovery protocols and winback squads.",
                        icon="🚨"
                    )
        else:
            st.error("Invalid entry string format. Customer identifiers must strictly consist of numeric sequences.")
            
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# SYSTEM APPLICATION INVOKER
# =========================================================================
if __name__ == "__main__":
    main()