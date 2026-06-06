"""
Centroid De-Normalization & Decoding Module
===========================================

Reverses multi-stage geometric and variance-stabilizing transformations 
to map abstract cluster coordinates back to physical business KPIs.

Dependencies:
    - logging: Standard internal tracking streams for pipeline auditing.
    - typing: Static type definitions and structural interfaces.
    - numpy: High-performance vectorized numerical arrays for linear operations.
    - pandas: Tabular data frame data structures for structural reporting.
    - scipy: Advanced mathematical special functions for non-linear inversion.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.special import inv_boxcox

# Initialize module-level logger bound to the parent execution namespace.
# This guarantees complete log traceability back to the main orchestrator.
logger = logging.getLogger(__name__)


def decode_cluster_centroids(
    fitted_kmeans_registry: Dict[int, Any],
    optimal_k: int,
    scaler: Any,
    lambda_opt: float,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extracts high-dimensional centroids from a prefitted estimator and reverses 
    standardization and mathematical scaling to recover original physical units.

    Architectural Rationale:
        Decouples geometric cluster analysis from business interpretation layers. 
        Maintains an exact mathematical inversion pipeline (Z-Score -> Linearized 
        Continuous -> Raw Skewed Physical Units) while running defensive checks 
        on transformation constants to eliminate silent numerical distortion.

    Parameters:
        fitted_kmeans_registry (Dict[int, Any]): Active execution dictionary holding 
                                                 pre-fitted estimators indexed by K.
        optimal_k (int): Target hyperparameter cluster count used as the lookup key.
        scaler (Any): Fitted StandardScaler instance containing empirical mean/variance vectors.
        lambda_opt (float): Strictly optimized power transformation scalar used for Frequency.
        feature_names (Optional[List[str]]): Target string labels for output columns. 
                                              Defaults to standard RFM schema naming.

    Returns:
        pd.DataFrame: A structured, index-aligned dataframe holding the de-normalized 
                      cluster centroids in real-world human-readable units.

    Raises:
        TypeError: If incoming parameters violate structural type constraints.
        KeyError: If the requested optimal_k model is missing from the active registry.
        AttributeError: If the estimator or scaler objects lack fully trained internal states.
        ValueError: If dimensional bounds are mismatched or numerical boundaries are breached.
    """
    
    # -------------------------------------------------------------------------
    # 0. SCHEMA INITIALIZATION & FALLBACK MANAGEMENT
    # -------------------------------------------------------------------------
    # Fallback Assignment: Enforce a default naming contract if no explicit labels 
    # are provided. This ensures out-of-the-box structural compatibility with RFM models.
    if feature_names is None:
        feature_names = ['Recency_Days', 'Frequency_Orders', 'Monetary_Value']

    # -------------------------------------------------------------------------
    # 1. DEPENDENCY INTEGRITY & HYPERPARAMETER VALIDATION
    # -------------------------------------------------------------------------
    # Defensive Boundary: Verify the tracking registry is a valid dictionary to protect 
    # subsequent subscript lookups from throwing fatal type errors.
    if not isinstance(fitted_kmeans_registry, dict):
        logger.error("Centroid decoding aborted: Provided registry is not a dictionary.")
        raise TypeError("The tracking registry must be an instantiated dictionary object.")

    # Validation Guardrail: Confirm the selected model configuration exists in memory.
    # This prevents failures caused by missing grid search paths or configuration mismatches.
    if optimal_k not in fitted_kmeans_registry:
        logger.error("Centroid decoding aborted: Model instance for K=%d is missing from active memory.", optimal_k)
        raise KeyError(f"The model registry does not contain a pre-fitted estimator for key K={optimal_k}.")

    # Pipeline Guardrail: Ensure the scaling instance is fully materialized and contains
    # the proper transformation interfaces before exposing data arrays to execution loops.
    if scaler is None or not hasattr(scaler, "inverse_transform"):
        logger.error("Centroid decoding aborted: Scaler instance is uninitialized or missing inverse methods.")
        raise AttributeError("Provided scaler must be an active instance capable of running inverse transformations.")
        
    # State Verification: Inspect internal attributes to confirm the scaler was actually 
    # fit on training arrays. Checking for 'mean_' and 'scale_' eliminates silent failures.
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        logger.error("Centroid decoding aborted: Provided scaler is uninstantiated with fitted parameters.")
        raise AttributeError("The scaler object has not been fit to training data; inversion cannot be computed.")

    # Mathematical Boundary: Block execution if the Box-Cox parameter is unassigned 
    # or non-numeric. A null parameter makes inverse power transformations mathematically undefined.
    if lambda_opt is None or not isinstance(lambda_opt, (int, float, np.number)):
        logger.error("Centroid decoding aborted: Transformation parameter 'lambda_opt' is undefined or non-numeric.")
        raise ValueError("A valid scalar float 'lambda_opt' is required to execute inverse Box-Cox mapping.")

    # -------------------------------------------------------------------------
    # 2. STEP-WISE MATHEMATICAL TRANSFORMATION INVERSION
    # -------------------------------------------------------------------------
    try:
        # Memory Reference Access: Retrieve the specific pre-fitted model object 
        # pointer directly from our active execution dictionary.
        logger.info("Extracting coordinate matrix for K=%d from execution registry.", optimal_k)
        final_kmeans = fitted_kmeans_registry[optimal_k]
        
        # State Verification: Confirm that the extracted model is fully materialized 
        # and contains trained coordinate arrays before running matrix transformations.
        if not hasattr(final_kmeans, "cluster_centers_"):
            logger.error("Centroid decoding aborted: Estimator for K=%d lacks a trained coordinate state.", optimal_k)
            raise AttributeError(f"The retrieved estimator for K={optimal_k} does not contain 'cluster_centers_'.")
        
        # Step A: Extract raw coordinates from isotropic space (Z-Scores).
        # Matrix shape corresponds directly to (K, Dimensions).
        scaled_centroids = final_kmeans.cluster_centers_
        
        # Structural Verification: Assert feature dimensionality matches column names length
        # exactly to block data fragmentation or partial slicing errors.
        if scaled_centroids.shape[1] != len(feature_names):
            logger.error("Dimensionality Mismatch: Centroid columns (%d) do not match feature names length (%d).", 
                         scaled_centroids.shape[1], len(feature_names))
            raise ValueError("Feature dimensional length must match provided output column matrix names precisely.")
        
        # Step B: Reverse standard scaling to restore variance-stabilized space.
        # This shifts and scales coordinates back to the continuous pre-normalized space.
        # Explicit Mathematical Formula: x_transformed = (z * sigma) + mu
        logger.info("Executing linear reversion of standard scaling normalization vector.")
        transformed_centroids = scaler.inverse_transform(scaled_centroids)
        
        # Step C: Initialize destination array using a clean copy configuration.
        # Specifying float64 explicitly prevents precision truncation during exponential scaling.
        real_centroids = np.zeros_like(transformed_centroids, dtype=np.float64)
        
        # Column 0: Reverse Square Root Transformation for Recency.
        # Direct inverse mapping requires squaring the elements.
        # Explicit Mathematical Formula: f(x) = x^2
        real_centroids[:, 0] = np.square(transformed_centroids[:, 0])
        
        # Column 1: Reverse Box-Cox Transformation for Frequency.
        # Utilizes SciPy specialized vector functions to compute the non-linear power inversion.
        # Explicit Mathematical Formula: f(x, lambda) = (x * lambda + 1) ^ (1 / lambda)
        real_centroids[:, 1] = inv_boxcox(transformed_centroids[:, 1], lambda_opt)
        
        # Column 2: Reverse Log1p Transformation for Monetary Value.
        # Uses expm1 to handle precision scaling gracefully. This computes exp(x) - 1, 
        # which preserves exact precision for fractional currency entries near zero.
        # Explicit Mathematical Formula: f(x) = exp(x) - 1
        real_centroids[:, 2] = np.expm1(transformed_centroids[:, 2])
        
        # -------------------------------------------------------------------------
        # 3. NUMERICAL STABILITY BOUNDARY CHECK
        # -------------------------------------------------------------------------
        # Defensive Check: Verify that no exponential or power transformations introduced 
        # infinite values or NaN limits due to floating-point overflows or domain restrictions.
        if not np.all(np.isfinite(real_centroids)):
            logger.error("Numerical Instability: De-normalization generated infinite or NaN float spaces.")
            raise ValueError("Mathematical inversion encountered overflow or domain restrictions during calculation.")
            
        # -------------------------------------------------------------------------
        # 4. STRUCTURED DATA REGISTRY CONSTRUCTION
        # -------------------------------------------------------------------------
        # Hardened DataFrame Construction: Wrap the final physical matrix into a 
        # structured DataFrame using the validated feature labels contract.
        logger.info("Structuring physical unit matrix into a standardized Dataframe registry.")
        df_centroids = pd.DataFrame(
            real_centroids, 
            columns=feature_names
        )
        df_centroids.index.name = 'Cluster_ID'
        
        return df_centroids

    except Exception as execution_error:
        # Traceback Encapsulation: Route the complete system stack trace block via 
        # exc_info=True to decouple environmental anomalies from code logic flaws.
        logger.error("Critical mapping failure inside centroid inversion engine: %s", str(execution_error), exc_info=True)
        raise RuntimeError(f"Failed to decode mathematical cluster centroids: {execution_error}") from execution_error