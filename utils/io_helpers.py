# =========================================================================
# MODULE: Defensive Input/Output (I/O) Engineering Utilities
# =========================================================================
# Purpose: Provides safe, verifiable file-system serialization utilities
#          for structured analytical data assets using atomic write patterns.
#
# Side Effects: Interacts directly with the local/network disk storage layers,
#               generating directories, creating transient temporary files,
#               and validating physical file metrics.
# =========================================================================

import os
import logging
from typing import TYPE_CHECKING

# SYSTEM OPTIMIZATION: TYPE_CHECKING evaluated as True ONLY during static analysis 
# (like running a MyPy linters or IDE autocomplete routines). During live execution, 
# it returns False, preventing the runtime environment from prematurely forcing 
# heavy module compilation structures into memory.
if TYPE_CHECKING:
    import pandas as pd

# Standard module-level logging instantiator
logger = logging.getLogger(__name__)


def export_dataframe_to_csv(df: "pd.DataFrame", target_path: str, index: bool = False) -> None:
    """
    Serializes a pandas DataFrame to disk defensively via an atomic write-and-swap
    pattern. Verifies directory topologies, minimizes file corruption vectors,
    and captures physical footprint analytics.

    Args:
        df (pd.DataFrame): The target data asset to be serialized to disk.
        target_path (str): The absolute or relative destination file system path.
        index (bool): Dictates whether row indices are persisted to the CSV payload. 
                      Defaults to False.

    Raises:
        ValueError: If the input dataframe structure is empty.
        FileNotFoundError: Triggered if disk write verification parameters fail post-swap.
        Exception: Cascades unhandled OS/System filesystem permissions or disk faults.
    """
    logger.info(f"Initiating CSV storage serialization protocol for target path: {target_path}")
    
    if df.empty:
        logger.warning(f"Performance Alert: Target DataFrame for {target_path} is empty.")

    # -------------------------------------------------------------------------
    # 1. PRE-WRITE CHECK: PATH STRUCTURE VALIDATION & AUTO-GENERATION
    # -------------------------------------------------------------------------
    parent_dir = os.path.dirname(target_path)
    if parent_dir and not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
            logger.info(f"Created missing directory infrastructure at: {parent_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize storage path structures: {str(e)}")
            raise
            
    # -------------------------------------------------------------------------
    # 2. ATOMIC EXECUTION PLANE: STAGED WRITE & TRANSACTIONAL RENAMING
    # -------------------------------------------------------------------------
    # RATIONALE: Writing directly to target paths exposes lines to corrupt state risks
    # if execution limits are triggered mid-write. This block staging mitigates data loss.
    # -------------------------------------------------------------------------
    temp_path = f"{target_path}.tmp"
    try:
        # Enforcing explicit UTF-8 encoding ensures strings containing currency symbols, 
        # textual descriptions, or special formatting characters don't corrupt downstream.
        df.to_csv(temp_path, index=index, encoding='utf-8')
        
        # Atomically replace target path with completed file transaction
        os.replace(temp_path, target_path)
        
        # Immediate Physical Verification Gate
        if os.path.exists(target_path):
            file_size_kb = os.path.getsize(target_path) / 1024
            logger.info("CSV storage serialization executed successfully via atomic swap.")
            logger.info(f"File Metadata -> Rows: {len(df):,} | Disk Footprint: {file_size_kb:.2f} KB")
        else:
            raise FileNotFoundError(
                f"Verification Failure: CSV file was not detected post-write at targeted address: {target_path}"
            )
            
    except Exception as e:
        logger.error(f"Critical I/O error during CSV file serialization processing: {str(e)}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("Cleanup successful: Transient CSV staging files purged.")
            except Exception as cleanup_error:
                logger.error(f"Staging leak occurred: Unable to flush temp files: {str(cleanup_error)}")
        raise


def export_dataframe_to_parquet(df: "pd.DataFrame", target_path: str, compression: str = "snappy") -> None:
    """
    Serializes a pandas DataFrame to disk using Apache Parquet format via an atomic
    write-and-swap sequence to guarantee type-safety, schema lock preservation, 
    and optimized storage footprint footprints.

    Args:
        df (pd.DataFrame): The schema-locked data asset to be serialized to disk.
        target_path (str): The absolute or relative destination file system path.
        compression (str): Compression codec to apply (e.g., 'snappy', 'gzip', 'brotli').
                           Defaults to 'snappy' for balanced read/write performance speeds.

    Raises:
        ValueError: If the input dataframe structure is empty.
        FileNotFoundError: Triggered if disk write verification parameters fail post-swap.
        ImportError: If required dependency engines ('pyarrow' or 'fastparquet') are missing.
        Exception: Cascades unhandled OS/System filesystem permissions or disk faults.
    """
    logger.info(f"Initiating Parquet storage serialization protocol for target path: {target_path}")
    
    if df.empty:
        logger.warning(f"Performance Alert: Target DataFrame for {target_path} is empty.")

    # -------------------------------------------------------------------------
    # 1. PRE-WRITE CHECK: PATH STRUCTURE VALIDATION & AUTO-GENERATION
    # -------------------------------------------------------------------------
    parent_dir = os.path.dirname(target_path)
    if parent_dir and not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
            logger.info(f"Created missing directory infrastructure at: {parent_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize storage path structures: {str(e)}")
            raise
            
    # -------------------------------------------------------------------------
    # 2. ATOMIC EXECUTION PLANE: STAGED WRITE & TRANSACTIONAL RENAMING
    # -------------------------------------------------------------------------
    temp_path = f"{target_path}.tmp"
    try:
        # Enforce pyarrow engine explicitly to ensure uniform dependency errors if missing
        df.to_parquet(temp_path, index=False, compression=compression, engine='pyarrow')
        
        # Atomically replace target path with completed file transaction
        os.replace(temp_path, target_path)
        
        # Immediate Physical Verification Gate
        if os.path.exists(target_path):
            file_size_kb = os.path.getsize(target_path) / 1024
            logger.info("Parquet columnar storage serialization executed successfully via atomic swap.")
            logger.info(f"File Metadata -> Rows: {len(df):,} | Disk Footprint: {file_size_kb:.2f} KB | Codec: {compression}")
        else:
            raise FileNotFoundError(
                f"Verification Failure: Parquet file was not detected post-write at targeted address: {target_path}"
            )
            
    except ImportError as engine_error:
        logger.critical(f"Engine Configuration Failure: Underlying Parquet dependency missing. Details: {str(engine_error)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise ImportError("Pipeline Execution Interrupted: Missing 'pyarrow' engine required for Parquet processing.") from engine_error
    except Exception as e:
        logger.error(f"Critical I/O error during Parquet file serialization processing: {str(e)}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("Cleanup successful: Transient Parquet staging files purged.")
            except Exception as cleanup_error:
                logger.error(f"Staging leak occurred: Unable to flush temp files: {str(cleanup_error)}")
        raise