"""
Strategic Profile Synthesis & Hybrid Cohort Aggregation Module
==============================================================

Engineered to blend unsupervised clustering outputs with raw physical metrics 
and heuristic shadow flags. This module produces action-oriented customer 
cohort profiles designed for direct integration with CRM routing engines.

Design Patterns:
    * Defensive Programming: Rigid structural schema validation and index matching.
    * Fault Tolerance: Comprehensive handling of unmapped cluster IDs and cold-starts.
    * Vectorized Execution: Minimizes loop overhead via native Pandas aggregations.

Dependencies:
    * logging: Standard tracking streams for pipeline auditing.
    * typing: Static type definitions and structural interfaces.
    * pandas: Tabular data frame data structures for structural reporting.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd

# Enforce explicit logging trace inside the parent notebook namespace
logger = logging.getLogger(__name__)


def synthesize_strategic_profiles(
    df_rfm: pd.DataFrame,
    df_rfm_transformed: pd.DataFrame,
    cluster_labels: Any,
    cluster_mapping: Optional[Dict[int, str]] = None,
    shadow_flag_col: str = "Is_Systemic_VIP"
) -> pd.DataFrame:
    """
    Combines machine learning cluster labels with raw customer metrics, applies 
    a deterministic business shadow flag override, and generates macro business metrics.

    Architectural Rationale:
        Decouples reporting visualization from the core model execution path. 
        By mapping cluster assignments back to the unscaled, un-Winsorized raw 
        dataset, executive dashboards reflect true financial realities while 
        safely isolating the extreme variance of systemic outliers.

    Parameters:
        df_rfm (pd.DataFrame): Raw, un-Winsorized data containing original physical units.
        df_rfm_transformed (pd.DataFrame): Preprocessed data containing the shadow flag column.
        cluster_labels (Any): Array-like vector of cluster IDs assigned by the model.
        cluster_mapping (Optional[Dict[int, str]]): Map translating numeric clusters to strings.
                                                    Defaults to Action Stream taxonomy mapping.
        shadow_flag_col (str): Column name identifying the systemic ultra-whale records.
                               Defaults to 'Is_Systemic_VIP'.

    Returns:
        pd.DataFrame: A structured summary profile dataframe aggregated by strategic cohort.

    Raises:
        TypeError: If incoming parameters violate structural type constraints.
        KeyError: If mandatory columns or flags are missing from the input spaces.
        ValueError: If array row dimensions or indices are misaligned.
    """
    
    # -------------------------------------------------------------------------
    # 0. SCHEMA INITIALIZATION & FALLBACK MANAGEMENT
    # -------------------------------------------------------------------------
    # Fallback to Option 2 Action Stream taxonomy if no custom dictionary is injected
    if cluster_mapping is None:
        cluster_mapping = {
            0: 'Action_Premium_Retention',
            1: 'Action_Low_Cost_Reengage',
            2: 'Action_Nurture_Upsell',
            3: 'Action_High_Priority_Winback'
        }

    # -------------------------------------------------------------------------
    # 1. DEFENSIVE DATA INTEGRITY GUARDRAILS
    # -------------------------------------------------------------------------
    # Type Validation: Reject raw primitive types or uninstantiated arrays
    if not isinstance(df_rfm, pd.DataFrame) or not isinstance(df_rfm_transformed, pd.DataFrame):
        logger.error("Profile synthesis aborted: Core inputs must be valid Pandas DataFrames.")
        raise TypeError("Inputs df_rfm and df_rfm_transformed must be instantiated DataFrames.")

    # Schema Verification: Ensure mandatory input features exist in the raw dataframe space
    required_rfm_cols: List[str] = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    missing_rfm_cols: List[str] = [col for col in required_rfm_cols if col not in df_rfm.columns]
    if missing_rfm_cols:
        logger.error("Profile synthesis aborted: Missing columns in df_rfm: %s", missing_rfm_cols)
        raise KeyError(f"Dataframe df_rfm is missing required structural features: {missing_rfm_cols}")

    # Shadow Flag Visibility: Validate presence of the tracking flag before proceeding to modifications
    if shadow_flag_col not in df_rfm_transformed.columns:
        logger.error("Profile synthesis aborted: Shadow flag '%s' missing from transformed matrix.", shadow_flag_col)
        raise KeyError(f"The required shadow flag column '{shadow_flag_col}' was not found in df_rfm_transformed.")

    # Shape Matrix Validation: Verify parallel row dimensions across all incoming arrays
    if len(df_rfm) != len(df_rfm_transformed) or len(df_rfm) != len(cluster_labels):
        logger.error("Dimension Mismatch: Array row shapes do not match across data vectors. "
                     "Lengths -> df_rfm: %d, df_rfm_transformed: %d, cluster_labels: %d",
                     len(df_rfm), len(df_rfm_transformed), len(cluster_labels))
        raise ValueError("Pipeline Alignment Error: Mismatched row counts between raw frames, transformed frames, and labels.")

    # Coordinate Axis Verification: Ensure perfect index continuity to guarantee alignment safety
    if not df_rfm.index.equals(df_rfm_transformed.index):
        logger.error("Index Structure Contamination: df_rfm and df_rfm_transformed indices are misaligned.")
        raise ValueError("Pipeline Alignment Error: Indices of raw and transformed dataframes must be perfectly identical.")

    # -------------------------------------------------------------------------
    # 2. DATA MERGING & HEURISTIC OVERRIDE CORRECTION
    # -------------------------------------------------------------------------
    try:
        logger.info("Initializing baseline cohort assignments on raw metric space.")
        
        # Deep copy to decouple core processing steps from the source notebook dataframe
        df_working = df_rfm.copy()
        
        # Force label series alignment using the raw dataframe index to avoid positional drift
        series_labels = pd.Series(cluster_labels, index=df_working.index)
        
        # Apply dictionary mapping values to map text designations onto numeric keys
        df_working['Cohort_Label'] = series_labels.map(cluster_mapping)
        
        # Anomaly Isolation: Gracefully capture unmapped cluster IDs instead of generating null rows
        if df_working['Cohort_Label'].isna().any():
            unmapped_count = df_working['Cohort_Label'].isna().sum()
            logger.warning("Unmapped cluster IDs detected: %d records fell outside target definitions.", unmapped_count)
            df_working['Cohort_Label'] = df_working['Cohort_Label'].fillna("Unassigned_Cluster_Anomaly")
        
        # Isolate the boolean mask for ultra-whale accounts via the transformed shadow flag
        vip_mask = df_rfm_transformed[shadow_flag_col] == 1
        systemic_count = vip_mask.sum()
        
        # Apply the deterministic hard strategic override to route ultra-whales out of regular cohorts
        logger.info("Applying heuristic override: Routing %d systemic ultra-whale profiles.", systemic_count)
        df_working.loc[vip_mask, 'Cohort_Label'] = 'Action_White_Glove_Concierge'

        # -------------------------------------------------------------------------
        # 3. MULTI-INDEX COHORT AGGREGATION
        # -------------------------------------------------------------------------
        logger.info("Executing macro-level group summaries on physical units.")
        
        # Vectorized groupby pass calculating true baseline physical volumes
        cluster_profile = df_working.groupby('Cohort_Label').agg({
            'CustomerID': 'count',
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum']
        })

        # Flatten the structural multi-index hierarchy to enforce consistent schemas downstream
        cluster_profile.columns = [
            'Customer_Count', 
            'Avg_Recency_Days', 
            'Avg_Frequency_Orders', 
            'Avg_Monetary_Value', 
            'Total_Revenue_Contribution'
        ]

        # -------------------------------------------------------------------------
        # 4. RATIO COMPUTATION & ZERO-DIVISION PROTECTION
        # -------------------------------------------------------------------------
        global_revenue = cluster_profile['Total_Revenue_Contribution'].sum()
        global_customers = len(df_working)

        # Zero-Division Guardrail: Handle empty platforms or complete zero revenue states cleanly
        if global_revenue > 0:
            cluster_profile['Revenue_Share_%'] = (cluster_profile['Total_Revenue_Contribution'] / global_revenue) * 100
        else:
            logger.warning("Zero Financial Activity Baseline: Global revenue evaluates exactly to zero.")
            cluster_profile['Revenue_Share_%'] = 0.0

        # Zero-Division Guardrail: Prevent runtime calculation crashes during testing or cold starts
        if global_customers > 0:
            cluster_profile['Customer_Share_%'] = (cluster_profile['Customer_Count'] / global_customers) * 100
        else:
            logger.error("Zero Record Ingestion: Active working dataset contains no rows.")
            cluster_profile['Customer_Share_%'] = 0.0

        # Re-index to enforce clean structural layouts and truncate floating-point noise
        final_profile = cluster_profile[[
            'Customer_Count', 
            'Customer_Share_%', 
            'Avg_Recency_Days', 
            'Avg_Frequency_Orders', 
            'Avg_Monetary_Value', 
            'Total_Revenue_Contribution', 
            'Revenue_Share_%'
        ]].round(1)

        # Name the index explicitly to ensure clean visualization headers
        final_profile.index.name = 'Strategic_Operational_Action_Stream'
        return final_profile

    except Exception as execution_error:
        # Traceback Isolation Pattern: Ensure the original trace details are preserved for debugging
        logger.error("Critical tracking error inside aggregation engine: %s", str(execution_error), exc_info=True)
        raise RuntimeError(f"Failed to compile strategic customer profile matrices: {execution_error}") from execution_error