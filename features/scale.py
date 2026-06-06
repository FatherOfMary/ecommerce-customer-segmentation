"""
Feature Engineering Scaling Module
==================================

Author: Curriculum & Analytics Pipeline Architecture
Deploys standardized scaling workflows to project feature vectors onto a shared
zero-mean, unit-variance coordinate plane. Enforces dimensional equity across
Euclidean distance operations.

Dependencies:
    - logging: Pipeline tracing and execution tracking.
    - typing: Static type analysis and contract enforcement.
    - numpy: Matrix verification and calculation.
    - pandas: Tabular vector manipulation.
    - sklearn.preprocessing: Scale normalization tools.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def standardize_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    metadata_cols: List[str],
    scaler: Optional[StandardScaler] = None,
    mean_tolerance: float = 1e-6,
    std_tolerance: float = 1e-6
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Validates, scales, and hardens a feature matrix for distance-based estimators.

    Architectural Rationale:
        This implementation incorporates a dual-modality execution pattern. If no 
        pre-fitted scaler is provided, it operates in training mode (fit_transform). 
        If an existing scaler state is provided, it operates in inference mode 
        (transform only). This architecture eliminates training-serving data skew.

    Parameters:
        df (pd.DataFrame): The incoming unscaled transformation matrix.
        feature_cols (List[str]): Columns designated for standardization.
        metadata_cols (List[str]): Identity/context metrics to keep unscaled.
        scaler (Optional[StandardScaler]): Pre-fitted operator state for inference execution.
        mean_tolerance (float): Numeric boundary for post-scaling mean verification.
        std_tolerance (float): Numeric boundary for post-scaling variance verification.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]:
            - pd.DataFrame: Structurally hardened, float32 downcasted matrix.
            - StandardScaler: The active scikit-learn operator instance.

    Raises:
        KeyError: Triggered if specified columns do not exist in the dataframe schema.
        ValueError: Triggered if dataframes are empty, or contain nulls/infinite values.
        TypeError: Triggered if targeted feature columns contain non-numeric types.
        AssertionError: Triggered if training scaling results violate mathematical tolerances.
    """
    # -------------------------------------------------------------------------
    # 1. VALIDATION GATES (FAIL-FAST PATTERN)
    # -------------------------------------------------------------------------
    
    # Gate 1: Column Presence Validation
    # Evaluates the structural integrity of the incoming schema contract. 
    # Catches breaking changes or upstream pipeline drift before wasting compute resources.
    missing_cols = [c for c in feature_cols + metadata_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Schema contract breach: Required columns absent from input: {missing_cols}")
        raise KeyError(f"Missing columns in input dataframe: {missing_cols}")

    # Gate 2: Empty Dataframe Guard
    # Zero-observation matrices produce degenerate, volatile scaling parameters (e.g., NaN means,
    # zero variance), leading to catastrophic failures in down-stream distance models.
    if df.empty:
        logger.error("Validation failure: Input dataframe contains zero observations.")
        raise ValueError("Cannot fit StandardScaler on an empty feature matrix.")

    # Gate 3: Numeric Type Compliance Enforcement
    # Scikit-learn algorithms implicitly attempt object conversions or fail with cryptic errors.
    # Explicit type verification guarantees that only mathematical arrays enter the matrix stream.
    non_numeric_cols = [c for c in feature_cols if not np.issubdtype(df[c].dtype, np.number)]
    if non_numeric_cols:
        logger.error(f"Data type violation: Feature columns must be strictly numeric: {non_numeric_cols}")
        raise TypeError(f"Non-numeric columns found in feature space: {non_numeric_cols}")

    # Gate 4: Null Contamination Guard
    # Missing values propagate through matrix operations, corrupting calculating parameters.
    # This check ensures strict array density requirements are met prior to calculation.
    null_counts = df[feature_cols].isnull().sum()
    if null_counts.any():
        offending = null_counts[null_counts > 0].to_dict()
        logger.error(f"Pre-scaling null contamination detected: {offending}")
        raise ValueError(f"Null values present in feature columns: {offending}")

    # Gate 5: Infinite Value Guard
    # Overflowing or infinite values passed into the scikit-learn backend generate NaN tracking parameters.
    # This prevents un-trackable numeric explosions during standard deviation calculation.
    inf_counts = np.isinf(df[feature_cols].values).sum(axis=0)
    if inf_counts.any():
        offending = dict(zip(feature_cols, inf_counts.tolist()))
        logger.error(f"Pre-scaling infinite value contamination detected: {offending}")
        raise ValueError(f"Infinite values present in feature columns: {offending}")

    logger.info(f"Pre-scaling validation gates cleared. Row count: {df.shape[0]:,}")

    # -------------------------------------------------------------------------
    # 2. STANDARD SCALING EXECUTION (DUAL-MODALITY TRACK)
    # -------------------------------------------------------------------------
    # Isolate a clean pointer slice of the mathematical target space.
    # Operating on an explicit copy eliminates SettingWithCopyWarning exceptions and
    # guarantees that the upstream workspace dataframe remains unmutated.
    X_input = df[feature_cols].copy()
    
    if scaler is None:
        logger.info("Executing pipeline in TRAINING mode (fit_transform).")
        scaler = StandardScaler()
        X_scaled_array = scaler.fit_transform(X_input)
        
        # Log training parameters as an immutable audit footprint.
        # These locked configurations serve as validation fingerprints during production deployment.
        logger.info("StandardScaler fit parameters locked:")
        for col, mu, sigma in zip(feature_cols, scaler.mean_, scaler.scale_):
            logger.info(f"  {col:<28} | μ = {mu:>9.5f} | σ = {sigma:>9.5f}")
            
        # ---------------------------------------------------------------------
        # 3. POST-SCALING CONTRACT ASSERTIONS (TRAINING MODALITY ONLY)
        # ---------------------------------------------------------------------
        # These runtime mathematical checks confirm the mathematical accuracy of the output.
        # This evaluation is isolated to training operations because arbitrary testing or 
        # production validation splits are not expected to exhibit localized zero-mean attributes.
        scaled_means = X_scaled_array.mean(axis=0)
        scaled_stds = X_scaled_array.std(axis=0)

        # Evaluate mean convergence parameters against floating point tolerances
        mean_violations = [
            (feature_cols[i], float(scaled_means[i]))
            for i in range(len(feature_cols)) if abs(scaled_means[i]) > mean_tolerance
        ]
        # Evaluate standard deviation unit convergence boundaries
        std_violations = [
            (feature_cols[i], float(scaled_stds[i]))
            for i in range(len(feature_cols)) if abs(scaled_stds[i] - 1.0) > std_tolerance
        ]

        if mean_violations:
            logger.error(f"Post-scaling mean contract breached (expected μ ≈ 0): {mean_violations}")
            raise AssertionError(f"Scaled means deviate beyond tolerance threshold: {mean_violations}")

        if std_violations:
            logger.error(f"Post-scaling std contract breached (expected σ ≈ 1): {std_violations}")
            raise AssertionError(f"Scaled standard deviations deviate beyond tolerance: {std_violations}")

        logger.info("Post-scaling contract assertions passed: μ ≈ 0 and σ ≈ 1 verified.")
        
    else:
        # Inference/Serving track: Applies frozen training boundaries to incoming streams.
        # This strict structural isolation completely neutralizes training-serving data leakage risks.
        logger.info("Executing pipeline in SERVING/INFERENCE mode (transform only). Reusing locked state.")
        X_scaled_array = scaler.transform(X_input)

    # -------------------------------------------------------------------------
    # 4. SCHEMA CONSTRUCTION & TYPE HARDENING
    # -------------------------------------------------------------------------
    # Reconstruct the feature data matrix while matching rows with the initial indexing parameters.
    df_features = pd.DataFrame(X_scaled_array, columns=feature_cols, index=df.index)
    
    # Reattach descriptive identification keys (shadow attributes) required for profiling.
    # Concat by axis=1 ensures safe horizontal merging without changing data densities.
    df_scaled = pd.concat([df[metadata_cols].copy(), df_features], axis=1)

    # Hard downcast default float64 configurations to float32 allocations.
    # This halves matrix memory footprints, accelerating downstream distance calculations.
    for col in feature_cols:
        df_scaled[col] = df_scaled[col].astype('float32')

    # Data integrity check: confirms row count conservation across formatting operations.
    assert len(df_scaled) == len(df), (
        f"Row count breach: {len(df_scaled)} rows post-scaling vs {len(df)} rows pre-scaling."
    )

    return df_scaled, scaler