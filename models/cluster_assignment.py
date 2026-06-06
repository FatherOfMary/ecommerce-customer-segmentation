"""
Cluster Assignment Synchronization Module
==========================================

Extracts pre-computed, fully converged clustering estimators from active 
registry memory and propagates the partition vectors across active dataframes.

Dependencies:
    - logging: Standard internal tracking streams.
    - typing: Static type definitions and structural contracts.
    - pandas: High-performance tabular data frame vectors.
"""

import logging
from typing import Dict, Any
import pandas as pd

# Initialize module-level logger bound to the parent execution namespace.
# This ensures log traceability back to the main notebook or orchestration framework.
logger = logging.getLogger(__name__)


def map_prefitted_cluster_labels(
    fitted_kmeans_registry: Dict[int, Any],
    optimal_k: int,
    df_rfm: pd.DataFrame,
    df_rfm_transformed: pd.DataFrame,
    df_scaled: pd.DataFrame
) -> None:
    """
    Extracts a pre-fitted KMeans model from memory and synchronizes its partition
    labels across all core analytical registries.

    Architectural Rationale:
        Bypasses redundant CPU training passes entirely by accessing the pre-computed
        state elements of the estimator tracking dictionary. Enforces strict index
        identity validations to ensure safe, deterministic matrix alignments across 
        all targeted datasets. Protects operations from SettingWithCopy warnings.

    Parameters:
        fitted_kmeans_registry (Dict[int, Any]): Active dictionary holding pre-fitted estimators.
        optimal_k (int): Target configuration cluster count key.
        df_rfm (pd.DataFrame): Core master customer profile dataframe.
        df_rfm_transformed (pd.DataFrame): Outlier-treated analytic workspace dataframe.
        df_scaled (pd.DataFrame): Scaled/transformed feature space dataframe.

    Returns:
        None (Modifies target dataframes in-place)

    Raises:
        TypeError: If the incoming tracking registry is not a valid dictionary.
        KeyError: If the target optimal_k model does not exist in the registry.
        ValueError: If a structural or index alignment mismatch is detected.
        AttributeError: If the stored estimator instance lacks a trained state.
    """
    # -------------------------------------------------------------------------
    # 1. REGISTRY INTEGRITY & HYPERPARAMETER VALIDATION
    # -------------------------------------------------------------------------
    # Defensive Boundary: Ensure the registry object is an initialized dictionary
    # to protect the subsequent subscript lookups from fatal type errors.
    if not isinstance(fitted_kmeans_registry, dict):
        logger.error("Registry mapping aborted: Provided fitted_kmeans_registry is not a dictionary.")
        raise TypeError("The tracking registry must be an instantiated dictionary object.")

    # Validation Guardrail: Confirm the selected model configuration exists in 
    # memory. This catches upstream grid search crashes or parameter mismatches early.
    if optimal_k not in fitted_kmeans_registry:
        # Utilizing lazy parameter formatting (%d) to prevent premature string 
        # evaluation overhead within inactive logging states.
        logger.error("Registry mapping aborted: Model instance for K=%d is missing from active memory.", optimal_k)
        raise KeyError(f"The model registry does not contain a pre-fitted estimator for hyperparameter K={optimal_k}.")

    # -------------------------------------------------------------------------
    # 2. CORE REGISTRY ACCESSIBILITY & SHAPE BOUNDARY CHECKS
    # -------------------------------------------------------------------------
    # Integrity Check: Ensure the baseline target dataframe contains valid data 
    # rows before establishing it as the index alignment anchor.
    if df_rfm is None or df_rfm.empty:
        logger.error("Registry mapping aborted: Master dataframe 'df_rfm' is null or empty.")
        raise ValueError("Master dataframe 'df_rfm' must be a valid, populated pandas DataFrame.")

    # Construct an execution manifest to batch process independent analytical dataframes.
    target_frames = {
        "df_rfm_transformed": df_rfm_transformed,
        "df_scaled": df_scaled
    }

    for name, df in target_frames.items():
        # Defensive Check: Block null matrices or unpopulated data spaces from 
        # reaching the array synchronization step.
        if df is None or df.empty:
            logger.error("Registry mapping aborted: Target dataframe '%s' is null or empty.", name)
            raise ValueError(f"Target data workspace '{name}' contains no valid records.")
        
        # Production Edge Case Fix: Validate exact index configurations rather 
        # than generic row counts. This guarantees rows match record-for-record 
        # even if an upstream step reordered or shuffled specific frames.
        if not df.index.equals(df_rfm.index):
            logger.error("Registry mapping aborted: Index alignment mismatch between df_rfm and %s.", name)
            raise ValueError(f"Structural alignment mismatch: '{name}' index configuration must match df_rfm exactly.")

    # -------------------------------------------------------------------------
    # 3. ZERO-WASTE EXTRACTION AND STATE PROPAGATION
    # -------------------------------------------------------------------------
    try:
        logger.info("Extracting converged estimator labels for K=%d from execution registry.", optimal_k)
        
        # Memory Reference Access: Retrieve the specific pre-fitted model object 
        # pointer directly from our active execution dictionary.
        final_kmeans = fitted_kmeans_registry[optimal_k]
        
        # State Verification: Confirm that the extracted model is fully materialized 
        # and has successfully undergone numerical convergence before parsing attributes.
        if not hasattr(final_kmeans, "labels_"):
            logger.error("Registry mapping aborted: Estimator for K=%d is present but has not been fitted.", optimal_k)
            raise AttributeError(f"The retrieved KMeans estimator for K={optimal_k} lacks a valid 'labels_' state attribute.")
        
        # O(1) Data Extraction: Read the finalized cluster token array straight 
        # from the model state, discarding any need for a costly .predict() pass.
        cluster_labels = final_kmeans.labels_

        # Final Verification: Validate that the array shape matches our target 
        # dataframe lengths exactly, preventing partial data truncation or stretching.
        base_len = len(df_rfm)
        if len(cluster_labels) != base_len:
            logger.error("Partition array length (%d) does not match destination row count (%d).", len(cluster_labels), base_len)
            raise ValueError("Extracted partition array length fails to match destination data tracking dimensions.")

        # Hardened Modification Layer: Use explicit row-and-column slice syntax (.loc) 
        # to block downstream SettingWithCopyWarnings. This guarantees modifications 
        # occur safely in-place, even if the dataframes are structural views of memory.
        df_rfm.loc[:, 'Cluster'] = cluster_labels
        df_rfm_transformed.loc[:, 'Cluster'] = cluster_labels
        df_scaled.loc[:, 'Cluster'] = cluster_labels

        logger.info("Successfully propagated cluster vector across %d core data registries.", len(target_frames) + 1)

    except Exception as execution_error:
        # Fail-Safe Logging Architecture: Capture the entire execution traceback block 
        # via exc_info=True to decouple environmental memory leaks from syntax anomalies.
        logger.error("Structural assignment failure in core module engine: %s", str(execution_error), exc_info=True)
        raise RuntimeError(f"Failed to synchronize model partitions: {execution_error}") from execution_error