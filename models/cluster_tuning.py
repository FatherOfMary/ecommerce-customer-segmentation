"""
Cluster Optimization Search Module
==================================

Executes optimization grid searches to locate structural inflection thresholds 
(the Elbow Method) and maximize cohesive boundary matching (Silhouette Analysis).

Dependencies:
    - logging: Standard internal tracking streams.
    - typing: Static type definitions and structural contracts.
    - time: Precise execution tracking benchmarks.
    - numpy: Array-level validation metrics.
    - pandas: High-performance tabular data frame vectors.
    - sklearn.cluster: Centroid optimization utilities.
    - sklearn.metrics: Cluster distance silhouette evaluation metrics.
"""

import logging
import time
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def evaluate_kmeans_grid(
    df: pd.DataFrame,
    feature_cols: List[str],
    k_min: int = 2,
    k_max: int = 8,
    random_seed: int = 42,
    silhouette_sample_size: int = 20000
) -> Tuple[pd.DataFrame, Dict[int, KMeans]]:
    """
    Executes a structured hyperparameter grid search across a range of clusters.

    Architectural Rationale:
        Isolates mathematical feature slices from incoming data frames to protect
        vector calculations from identity key contamination. Implements statistical
        sub-sampling for high-dimensional silhouette tracking to manage volatile 
        memory allocations. Returns both tracking metrics and fitted models to
        eliminate downstream retraining cycles.

    Parameters:
        df (pd.DataFrame): Target coordinate data frame containing modeling metrics.
        feature_cols (List[str]): Strictly numeric column handles for distance tracking.
        k_min (int): Lower bound limit of clusters to test (inclusive).
        k_max (int): Upper bound limit of clusters to test (inclusive).
        random_seed (int): State seed to lock deterministic centroid initializations.
        silhouette_sample_size (int): Threshold ceiling for cluster row sampling.

    Returns:
        Tuple[pd.DataFrame, Dict[int, KMeans]]:
            - pd.DataFrame: Metric tracking log containing K, WCSS, and Silhouette scores.
            - Dict[int, KMeans]: Registry of fitted model objects keyed by their K value.

    Raises:
        KeyError: If targeted feature columns are absent from the data frame schema.
        ValueError: If configuration limits are invalid, data is empty, or constraints fail.
        TypeError: If targeted feature columns contain non-numeric data types.
    """
    # -------------------------------------------------------------------------
    # 1. STRUCTURAL CONTRACT GUARDRAILS (FAIL-FAST PATTERN)
    # -------------------------------------------------------------------------
    if df.empty:
        logger.error("Grid search aborted: Input dataframe contains zero observations.")
        raise ValueError("Cannot execute cluster hyperparameter tuning on an empty dataframe.")

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Grid search aborted. Feature columns missing from schema: {missing_cols}")
        raise KeyError(f"Required feature columns absent from matrix: {missing_cols}")

    if k_min < 2:
        logger.error(f"Invalid configuration boundary: k_min={k_min}. Must be >= 2.")
        raise ValueError("Silhouette score optimizations require a minimum boundary of 2 clusters.")

    if k_max < k_min:
        logger.error(f"Invalid configuration boundaries: k_max={k_max} is less than k_min={k_min}.")
        raise ValueError("The maximum cluster boundary cannot be smaller than the minimum boundary.")

    # Isolate mathematical target coordinates from identity parameters
    X_input = df[feature_cols].copy()
    total_observations = len(X_input)
    
    if total_observations < k_max:
        logger.error(f"Algorithmic constraint violation: Row count ({total_observations}) < k_max ({k_max}).")
        raise ValueError("Sample size must be greater than or equal to the maximum number of target clusters.")

    # Data Contamination Verification: Safeguards scikit-learn backend from math explosions
    if X_input.isnull().any().any():
        logger.error("Data integrity failure: Null values detected within optimization features.")
        raise ValueError("Input feature space contains missing values. Clean features prior to tuning.")

    if np.isinf(X_input.values).any():
        logger.error("Data integrity failure: Infinite values detected within optimization features.")
        raise ValueError("Input feature space contains infinite values. Clean features prior to tuning.")

    non_numeric = [col for col in feature_cols if not np.issubdtype(X_input[col].dtype, np.number)]
    if non_numeric:
        logger.error(f"Type contract breach: Non-numeric vectors found: {non_numeric}")
        raise TypeError(f"All target clustering features must be strictly numeric: {non_numeric}")

    logger.info(f"Grid search validated. Array scale: {total_observations:,} rows across {len(feature_cols)} dimensions.")
    
    # Storage structures for pipeline assets
    performance_metrics: List[Dict[str, Any]] = []
    fitted_models: Dict[int, KMeans] = {}

    # -------------------------------------------------------------------------
    # 2. HYPERPARAMETER LOOP SEARCH
    # -------------------------------------------------------------------------
    for k in range(k_min, k_max + 1):
        iteration_start = time.perf_counter()
        logger.info(f"Running iteration profile for K={k}")

        # Initialize estimator: k-means++ optimizes baseline seed positions 
        # to prevent model variations across different execution cycles
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=random_seed
        )
        
        try:
            kmeans.fit(X_input)
        except Exception as alg_error:
            logger.critical(f"Mathematical engine crash during KMeans execution at K={k}: {alg_error}", exc_info=True)
            raise RuntimeError(f"Centroid optimization failure at cluster size {k}: {alg_error}")

        wcss_score = float(kmeans.inertia_)
        
        # Cache the fitted model instance directly into the memory registry
        fitted_models[k] = kmeans

        # Memory Guard Silhouette Processing: Sub-samples tracking spaces 
        # for high-volume datasets to prevent memory overflow errors
        if total_observations > silhouette_sample_size:
            logger.debug(f"Row count exceeds threshold. Sampling {silhouette_sample_size:,} observations for silhouette tracking.")
            sil_score = float(
                silhouette_score(
                    X_input,
                    kmeans.labels_,
                    sample_size=silhouette_sample_size,
                    random_state=random_seed
                )
            )
        else:
            sil_score = float(silhouette_score(X_input, kmeans.labels_))

        iteration_duration = time.perf_counter() - iteration_start
        logger.info(f"Iteration completed for K={k} in {iteration_duration:.2f} seconds.")

        performance_metrics.append({
            "k_clusters": k,
            "wcss": wcss_score,
            "silhouette_avg": sil_score,
            "compute_time_sec": iteration_duration
        })

    # Pack results into structured tracking dataframe
    metrics_df = pd.DataFrame(performance_metrics)
    
    return metrics_df, fitted_models