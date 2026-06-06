"""
Visualization Utility Module
========================================

Deploys standardized visualization wrappers for evaluating feature engineering 
pipelines. The primary focus of this module is to deliver strict structural 
comparisons between highly skewed raw input matrices and stabilized, normalized 
data spaces.

Dependencies:
    - logging: Pipeline tracing and execution tracking.
    - typing: Static type analysis and contract enforcement.
    - matplotlib.pyplot: Subplot grid canvas instantiation.
    - pandas: Tabular vector extractions.
    - seaborn: High-level statistical plotting execution.
"""

import logging
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Module-level logging configuration for containerized execution tracking
logger = logging.getLogger(__name__)


def plot_rfm_distribution_comparison(
    df_raw: pd.DataFrame,
    df_transformed: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 12)
) -> plt.Figure:
    """
    Generates a 3x2 diagnostic visualization matrix evaluating RFM stabilization.

    Architectural Rationale:
        Distance-based clustering methodologies (e.g., K-Means) require geometric 
        symmetry and variance stabilization across dimensions. This function acts 
        as a visual test stage in data science pipelines to confirm that right-skewed 
        power-law distributions have successfully converged toward Gaussian targets 
        prior to model training.

    Design Patterns Implemented:
        1. Fail-Fast Validation: Immediate execution rejection upon contract breach.
        2. Declarative Schema Mapping: Separates layout metadata from execution logic.
        3. Aesthetic Isolation: Thread-safe canvas rendering via localized context managers.
        4. DRY Execution Matrix: Consolidates duplicate charting logic into a unified loop.

    Parameters:
        df_raw (pd.DataFrame):
            The un-transformed baseline dataframe containing raw 'Recency', 
            'Frequency', and 'Monetary' dimensions.
        df_transformed (pd.DataFrame):
            The transformed target dataframe containing the stabilized mathematical 
            counterparts ('Recency_sqrt', 'Frequency_boxcox', 'Monetary_log').
        figsize (Tuple[int, int], optional):
            The dimensions (width, height) of the overall plotting canvas. 
            Defaults to (15, 12).

    Returns:
        plt.Figure:
            The completed Matplotlib Figure instance, decoupled from active drawing 
            states and cleared for serial archiving or cell display.

    Raises:
        ValueError: 
            Triggered if either input dataframe lacks structural observations.
        KeyError: 
            Triggered if column mapping protocols fail due to missing keys in 
            the input schemas.
    """

    # -------------------------------------------------------------------------
    # SECTION 1: PRE-FLIGHT DATA INTEGRITY VALIDATION
    # -------------------------------------------------------------------------
    # Ensure that dataframes contain processable records before creating the canvas
    if df_raw.empty or df_transformed.empty:
        logger.error("Data integrity failure: Input dataframes cannot be empty.")
        raise ValueError("Provided dataframes contain zero observations.")

    # Declarative schema registry isolating operational metadata parameters.
    # To introduce new metrics or alter transformation targets, update this map.
    feature_schema: Dict[str, Dict[str, Any]] = {
        "Recency": {
            "trans_col": "Recency_sqrt",
            "raw_label": "Recency (Days)",
            "trans_label": "Square Root of Recency",
            "raw_fmt": ",.1f",
            "trans_fmt": ".2f"
        },
        "Frequency": {
            "trans_col": "Frequency_boxcox",
            "raw_label": "Frequency (Orders)",
            "trans_label": "Box-Cox Transformed",
            "raw_fmt": ",.1f",
            "trans_fmt": ".2f"
        },
        "Monetary": {
            "trans_col": "Monetary_log",
            "raw_label": "Monetary (Value)",
            "trans_label": "Logarithmic [ln(x+1)]",
            "raw_fmt": ",.1f",
            "trans_fmt": ".2f"
        }
    }

    # Verify column existence across inputs to guarantee execution contract safety
    for raw_col, config in feature_schema.items():
        if raw_col not in df_raw.columns:
            logger.error(f"Target raw feature missing from input schema: '{raw_col}'")
            raise KeyError(f"Column '{raw_col}' not found in raw dataframe.")
        if config["trans_col"] not in df_transformed.columns:
            logger.error(f"Target transformed feature missing from input schema: '{config['trans_col']}'")
            raise KeyError(f"Column '{config['trans_col']}' not found in transformed dataframe.")

    logger.info("Pre-flight schemas validated successfully. Commencing canvas compilation...")

    # -------------------------------------------------------------------------
    # SECTION 2: CANVAS INITIALIZATION & AESTHETIC ISOLATION
    # -------------------------------------------------------------------------
    # Enforcing 'with' context scoping prevents global plot style leaks.
    # This keeps style changes local to this function call.
    with sns.axes_style("white"):
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize, sharey=False)
        
        # Configure global canvas title hierarchies
        fig.suptitle(
            "RFM Distribution Engine: Statistical Transformation Analysis", 
            fontsize=18, 
            fontweight="bold",
            color="#2C3E50",
            y=0.98
        )

        # Standard typography tone across all text components
        text_color = "#2C3E50"

        # -------------------------------------------------------------------------
        # SECTION 3: PROGRAMMATIC GRID GENERATION (DRY PATTERN)
        # -------------------------------------------------------------------------
        # Loop over rows to dynamically build subplots instead of copying code block variations
        for row_idx, (raw_col, config) in enumerate(feature_schema.items()):
            
            # Map structural coordinates for the raw vs transformed data tracks
            column_matrix = [
                {
                    "ax": axes[row_idx, 0],
                    "series": df_raw[raw_col].dropna(), # Dropping nulls prevents downstream kde generation anomalies
                    "color": "#B84A62",                 # Muted Crimson indicates an un-normalized skew space
                    "title": f"Raw {raw_col} Geometry",
                    "label": config["raw_label"],
                    "fmt": config["raw_fmt"],
                    "accent_color": "#7F8C8D"
                },
                {
                    "ax": axes[row_idx, 1],
                    "series": df_transformed[config["trans_col"]].dropna(),
                    "color": "#2A9D8F",                 # Deep Sea Teal indicates a stabilized target workspace
                    "title": f"Stabilized {raw_col} Vector Workspace",
                    "label": config["trans_label"],
                    "fmt": config["trans_fmt"],
                    "accent_color": "#2A9D8F"
                }
            ]

            # Execute sequential column matrix array injections
            for col_cfg in column_matrix:
                ax = col_cfg["ax"]
                data_vector = col_cfg["series"]
                
                # Extract 3rd standardized moment and arithmetic averages
                skew_val = data_vector.skew()
                mean_val = data_vector.mean()
                median_val = data_vector.median()

                # Optimization Technique: Array extraction via data_vector bypasses the multi-index 
                # filtering overhead typical of standard 'data=df, x=string' executions inside iterations.
                sns.histplot(
                    x=data_vector, 
                    kde=True, 
                    ax=ax, 
                    color=col_cfg["color"], 
                    edgecolor="white", 
                    alpha=0.6
                )
                
                # Render Central Tendency Overlays.
                # The spatial separation between these indicators displays the presence or absence of skewness.
                ax.axvline(
                    mean_val, 
                    color="#7B1FA2", 
                    linestyle="--", 
                    linewidth=1.5, 
                    label=f"Mean: {mean_val:{col_cfg['fmt']}}"
                )
                ax.axvline(
                    median_val, 
                    color="#E67E22", 
                    linestyle=":", 
                    linewidth=2, 
                    label=f"Median: {median_val:{col_cfg['fmt']}}"
                )
                
                # Set axes layouts and typographical sizing contracts
                ax.set_title(col_cfg["title"], fontsize=13, fontweight="bold", color=text_color, pad=10)
                ax.set_xlabel(col_cfg["label"], fontsize=11, color=text_color)
                ax.set_ylabel("Observation Count", fontsize=11, color=text_color)
                ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9, loc="upper right")
                
                # Render the Skewness Text Card overlay onto the viewport grid coordinate space
                ax.text(
                    0.05, 0.85, 
                    f"Skewness: {skew_val:.2f}", 
                    transform=ax.transAxes, 
                    fontsize=10, 
                    fontweight="bold", 
                    color=col_cfg["accent_color"],
                    bbox=dict(facecolor="#FAFAFA", alpha=0.85, boxstyle="round,pad=0.5", edgecolor="#E0E0E0")
                )

        # -------------------------------------------------------------------------
        # SECTION 4: POST-PROCESSING CANVAS OPTIMIZATIONS
        # -------------------------------------------------------------------------
        # Strip structural spine noise from axes blocks to improve data ink ratios
        sns.despine(left=False, bottom=False, trim=True)
        
        # Calculate bounding boxes to fix overlapping labels across rows
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        logger.info("Diagnostic matrix generation completed successfully.")
        return fig