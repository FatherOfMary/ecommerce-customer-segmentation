"""
Cluster Diagnostic Visualization Module
=======================================

Renders high-fidelity dual-axis diagnostic plots to validate cluster tuning 
metrics, specifically tracking elbow point variance and silhouette max peaks.

Dependencies:
    - datetime: Standard calendar clock extraction libraries.
    - logging: Standard internal tracking streams.
    - typing: Static type definitions and structural contracts.
    - pandas: High-performance tabular data frame vectors.
    - matplotlib.pyplot: Vector graphics rendering engine.
"""

import datetime
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

# Initialize module-level logger bound to the parent execution namespace
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# CONSTANT DEFINITIONS & SYSTEM DEFAULT LAYOUT PROFILES
# -------------------------------------------------------------------------
# UI/UX Style Profile: Encapsulates WCAG-compliant high-contrast colors 
# and structural canvas properties to decouple styling from core rendering logic.
DEFAULT_STYLE: Dict[str, str] = {
    "color_wcss": "#1E3A8A",    # Deep Tailored Blue (Primary Metric)
    "color_sil": "#0F766E",     # Muted Emerald Teal (Secondary Metric)
    "color_opt": "#DF1243",     # Production Crimson Focus (Target Intersects)
    "color_grid": "#E2E8F0",    # Soft Slate Gray (Structural Gridlines)
    "color_text": "#1E293B",    # Dark Slate for Typography (High Legibility)
    "canvas_bg": "#FAFAFA",     # Premium off-white outer canvas
    "container_bg": "#FFFFFF",  # Pure white internal plotting container
    "border_color": "#CBD5E1"   # Clean muted gray bounding boxes
}


def plot_cluster_metrics(
    df_metrics: pd.DataFrame,
    optimal_k: Optional[int] = 4,
    save_path: Optional[str] = None,
    append_timestamp: bool = True,
    theme_config: Optional[Dict[str, str]] = None
) -> None:
    """
    Generates a high-fidelity dual-axis line chart for WCSS and Silhouette scores.

    Architectural Rationale:
        Decouples plotting frameworks from active notebook states. Accepts a 
        structured metrics DataFrame to map out spatial parameters dynamically. 
        Supports dynamic chronological versioning to preserve run histories.
        Guarantees engine memory cleanup using structured execution management.

    Parameters:
        df_metrics (pd.DataFrame): Tracking log containing K, WCSS, and Silhouette keys.
        optimal_k (Optional[int]): Target threshold to highlight with localized markers.
        save_path (Optional[str]): Target export path string for disk-level archival.
        append_timestamp (bool): Toggles inclusion of a runtime stamp on the saved file.
        theme_config (Optional[Dict[str, str]]): Overriding color mapping schema.

    Returns:
        None

    Raises:
        ValueError: If essential tracking elements or data points are missing.
    """
    # -------------------------------------------------------------------------
    # 1. STRUCTURAL CONTRACT GUARDRAILS & INPUT VALIDATION
    # -------------------------------------------------------------------------
    # Defensive Boundary: Block downstream logic from processing null or zero-row matrices
    if df_metrics is None or df_metrics.empty:
        logger.error("Plot generation rejected: The incoming metric dataset is null or empty.")
        raise ValueError("The provided metrics dataframe contains no data records to visualize.")

    # Target Structural Check: Ensure schema completeness before array slicing
    required_cols = ['k_clusters', 'wcss', 'silhouette_avg']
    missing_cols = [col for col in required_cols if col not in df_metrics.columns]
    
    if missing_cols:
        logger.error(f"Plot generation aborted. Expected tracking parameters missing: {missing_cols}")
        raise ValueError(f"Input dataframe must contain all required tracking keys: {required_cols}")

    # Configuration Fallback: Apply custom workspace styles if injected, else drop back to defaults
    style = theme_config if theme_config else DEFAULT_STYLE

    # Type Standardization: Cast vectors explicitly to vanilla Python structures 
    # to avoid unexpected downstream serialization or matrix rendering conflicts
    k_range = df_metrics['k_clusters'].astype(int).tolist()
    wcss = df_metrics['wcss'].tolist()
    silhouette_scores = df_metrics['silhouette_avg'].tolist()

    # Pre-allocate container instances to ensure global scope safety for resource cleanup
    fig, ax1 = None, None

    # -------------------------------------------------------------------------
    # 2. GRAPHICAL RENDERING ENGINE COHORT
    # -------------------------------------------------------------------------
    try:
        logger.info("Initializing modular dual-axis validation visualization sequence.")
        
        # Canvas Construction: Establish asset resolution bounds (120 DPI for document print clarity)
        fig, ax1 = plt.subplots(figsize=(12, 6.5), dpi=120)
        fig.patch.set_facecolor(style.get("canvas_bg", "#FAFAFA"))
        ax1.set_facecolor(style.get("container_bg", "#FFFFFF"))

        # Primary Axis Orchestration: Within-Cluster Sum of Squares (Inertia Curve)
        ax1.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='600', labelpad=12, color=style["color_text"])
        ax1.set_ylabel('WCSS / Inertia (Lower is Better)', color=style["color_wcss"], fontsize=12, fontweight='600', labelpad=12)
        
        # Primary Line Blueprint: Hollow circle anchors to emphasize coordinate values cleanly
        line1 = ax1.plot(
            k_range, wcss, 
            marker='o', color=style["color_wcss"], linewidth=3, markersize=8,
            markerfacecolor='#FFFFFF', markeredgewidth=2.5,
            label='WCSS (Elbow Method)'
        )
        ax1.tick_params(axis='y', labelcolor=style["color_wcss"], labelsize=10)
        ax1.tick_params(axis='x', colors=style["color_text"], labelsize=10)
        
        # Grid Architecture: Restrict gridlines to Axis 1 to prevent secondary mesh collisions
        ax1.grid(True, linestyle='-', color=style["color_grid"], linewidth=1, alpha=0.7)

        # Secondary Axis Orchestration: Shared Domain Twin Mapping
        ax2 = ax1.twinx()  
        ax2.set_ylabel('Average Silhouette Score (Higher is Better)', color=style["color_sil"], fontsize=12, fontweight='600', labelpad=12)
        
        # Secondary Line Blueprint: Differentiated dash style and square node anchors
        line2 = ax2.plot(
            k_range, silhouette_scores, 
            marker='s', color=style["color_sil"], linewidth=3, markersize=8,
            linestyle='--', markerfacecolor='#FFFFFF', markeredgewidth=2.5,
            label='Silhouette Score'
        )
        ax2.tick_params(axis='y', labelcolor=style["color_sil"], labelsize=10)
        ax2.grid(False)  # Explicitly disabled to preserve background grid clarity

        # Layout Minimalization: Strip standard bounding borders for a clean aesthetic look
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_color(style.get("border_color", "#CBD5E1"))
            ax.spines['left'].set_color(style.get("border_color", "#CBD5E1"))
            ax.spines['right'].set_color(style.get("border_color", "#CBD5E1"))

        # -------------------------------------------------------------------------
        # 3. OPTIMAL SELECTION HIGHLIGHT ZONE & ANNOTATIONS
        # -------------------------------------------------------------------------
        # Unify structural object tracking lists to build a coordinated plot legend
        lines = line1 + line2
        
        if optimal_k is not None and optimal_k in k_range:
            # Highlight Pillar: Frame the selected cluster model context with soft transparency
            ax1.axvspan(optimal_k - 0.15, optimal_k + 0.15, color=style["color_opt"], alpha=0.07, label=f'Selected Optimal (K={optimal_k})')
            ax1.axvline(x=optimal_k, color=style["color_opt"], linestyle=':', linewidth=2, alpha=0.6)
            
            # Extract the generated background rectangle patch to map it into the legend array
            lines += [ax1.patches[-1]]
            
            # Locate the numerical matrix index corresponding to the target configuration
            idx = k_range.index(optimal_k)
            
            # Targeting Overlay: Concentric target rings to highlight exact metric intersections
            ax1.plot(optimal_k, wcss[idx], marker='o', color=style["color_opt"], markersize=14, fillstyle='none', markeredgewidth=2.5)
            ax2.plot(optimal_k, silhouette_scores[idx], marker='s', color=style["color_opt"], markersize=14, fillstyle='none', markeredgewidth=2.5)
            
            # Callout 1 Processing: WCSS Elbow Point Annotation
            ax1.annotate(
                f'Elbow Inflection\nWCSS: {wcss[idx]:,.1f}',
                xy=(optimal_k, wcss[idx]),
                xytext=(optimal_k + 0.3, wcss[idx] + (max(wcss) - min(wcss)) * 0.05),
                arrowprops=dict(arrowstyle="->", color='#64748B', connectionstyle="arc3,rad=-0.1", linewidth=1.2),
                fontsize=10, color='#334155', fontweight='500'
            )
            
            # Callout 2 Processing: Silhouette Maximum Score Annotation
            ax2.annotate(
                f'Local Silhouette Peak\nScore: {silhouette_scores[idx]:.4f}',
                xy=(optimal_k, silhouette_scores[idx]),
                xytext=(optimal_k + 0.3, silhouette_scores[idx] - (max(silhouette_scores) - min(silhouette_scores)) * 0.15),
                arrowprops=dict(arrowstyle="->", color='#64748B', connectionstyle="arc3,rad=0.1", linewidth=1.2),
                fontsize=10, color='#334155', fontweight='500'
            )

        # Legend Consolidation: Merges elements from both axes into a single legend container
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=10, framealpha=0.95, facecolor='white', edgecolor='#E2E8F0')

        # Domain Locking: Constrain horizontal increments strictly to discrete valid integers
        ax1.set_xticks(k_range)
        
        plt.title('Hyperparameter Optimization Framework: Dual-Axis Geometric Validation', fontsize=14, fontweight='bold', pad=22, color=style["color_text"])
        plt.tight_layout()

        # -------------------------------------------------------------------------
        # 4. REPORT ARCHIVAL & PIPELINE RUNTIME LOGISTICS (WITH VERSIONING)
        # -------------------------------------------------------------------------
        if save_path:
            final_output_path = Path(save_path)
            
            # Dynamic Versioning Scan: Safely injects a sortable timestamp between file name and suffix
            if append_timestamp:
                runtime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                versioned_filename = f"{final_output_path.stem}_{runtime_str}{final_output_path.suffix}"
                final_output_path = final_output_path.with_name(versioned_filename)
            
            # Ensure target directories exist before writing file payload to disk
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(final_output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Diagnostic evaluation map archived to disk destination: {final_output_path.as_posix()}")

        # Headless Execution Safe Check: Prevents interactive shell blockages 
        # when running in automated cloud environments like Docker or Airflow
        if plt.get_backend().lower() != 'agg':
            plt.show()
        else:
            logger.info("Agg backend active. Graphic rendering saved without opening GUI window elements.")

    except Exception as visualization_error:
        logger.error(f"Visualization rendering crash in core module function: {visualization_error}", exc_info=True)
        raise RuntimeError(f"Failed to compile diagnostic charts: {visualization_error}") from visualization_error

    finally:
        # -------------------------------------------------------------------------
        # 5. HARDENED RESOURCE ISOLATION GUARD
        # -------------------------------------------------------------------------
        # Core Lifecycle Guard: Moving the close operation to a finally block 
        # guarantees that unmanaged memory context items are swept immediately 
        # even if a runtime failure occurs during execution.
        if fig is not None:
            plt.close(fig)
            logger.debug("Closed active figure context safely to preserve system memory space.")