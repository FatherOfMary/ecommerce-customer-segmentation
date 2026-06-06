# =========================================================================
# MODULE: Environmental Determinism Configuration Core
# =========================================================================
# Purpose: Establishes a strict, reproducible execution runtime across local,
#          containerized (Docker), and cloud-orchestrated (CI/CD) environments.
#
# Side Effects: Modifies host operating system environment variables immediately
#               upon module import to intercept low-level hardware thread pools.
# =========================================================================

import os
import random
import logging
import numpy as np

# -------------------------------------------------------------------------
# 1. IMMEDIATE RUNTIME THREAD SEEDING (CRITICAL INITIALIZATION POSITION)
# -------------------------------------------------------------------------
# RATIONALE: Low-level linear algebra backends (OpenBLAS, Intel MKL, vecLib) 
# configure their internal thread pools the exact millisecond NumPy or 
# Scikit-Learn is first imported into memory. 
#
# DESIGN PATTERN: This block is placed at the root module level (outside the 
# function) to force single-threaded execution before downstream mathematical 
# multi-threaded race conditions can cause floating-point centroid drift during 
# K-Means distance calculations.
# -------------------------------------------------------------------------
THREAD_VARIABLES = [
    'OMP_NUM_THREADS',            # OpenMP thread control
    'MKL_NUM_THREADS',            # Intel Math Kernel Library thread control
    'OPENBLAS_NUM_THREADS',        # OpenBLAS execution thread limit
    'VECLIB_MAXIMUM_THREADS',      # Apple vecLib framework override
    'NUMEXPR_NUM_THREADS'          # Fast numerical expression evaluator engine
]

for var in THREAD_VARIABLES:
    # Restricting to 1 thread ensures 100% deterministic matrix addition sequences
    os.environ[var] = '1'

# Named pipeline logger linked to the global settings streaming infrastructure
logger = logging.getLogger("Retail_Clustering_Pipeline")


# -------------------------------------------------------------------------
# 2. GLOBAL SEED SYNCHRONIZATION ENGINE
# -------------------------------------------------------------------------
def seed_everything(default_seed: int = 42) -> int:
    """
    Synchronizes the random state engines across all active software layers.
    Prioritizes dynamic runtime environment variables over hardcoded script defaults
    to support parallelized CI/CD test matrices.

    Args:
        default_seed (int): Fallback seed token if no environment variable is set. 
                            Defaults to 42 (Standard ML Baseline).

    Returns:
        int: The active seed token applied to the active runtime session.
            Crucial for logging pipeline metadata artifacts downstream.
    """
    
    # Extract structural seed override from host environment (e.g., GitHub Actions Matrix)
    env_seed = os.environ.get("GLOBAL_PIPELINE_SEED")
    
    if env_seed is not None:
        try:
            # Cast string env-var to integer defensively
            active_seed = int(env_seed)
            logger.info("CI/CD Environment Override Detected. Using injected seed.")
        except ValueError:
            # Defensive Fallback: If env_seed is corrupted (e.g., "alpha"), do not crash.
            # Log a warning and gracefully revert to the safe default anchor.
            active_seed = default_seed
            logger.warning(
                f"Invalid environment seed string '{env_seed}'. "
                f"Reverting to fallback token: {default_seed}"
            )
    else:
        # Standard local execution pathway
        active_seed = default_seed

    # Enforce reproducibility across standard Python structures (e.g., dictionary hashing)
    random.seed(active_seed)
    os.environ['PYTHONHASHSEED'] = str(active_seed)
    
    # Enforce reproducibility across NumPy vectorized array shuffles and splits
    np.random.seed(active_seed)
        
    logger.info(
        f"System determinism locked. Active Seed Vector: {active_seed} | "
        f"Thread pools restricted to 1."
    )
    
    return active_seed