# =========================================================================
# MODULE: Deterministic Data Cleansing Pipeline Engine
# =========================================================================
# Operational Context: Downstream of Data Ingestion & Diagnostic Gating.
# Purpose: Transforms raw transactional log structures into two distinct, 
#          isolated data assets: verified sales and segregated cancellations.
#
# Architectural Strategy: Enforces a strict functional, immutable paradigm.
#                         The raw input dataframe is cloned on entry to block
#                         accidental side-effects or state pollution across
#                         the global application workspace.
# =========================================================================

import logging
from typing import Tuple
import pandas as pd

# Inherits the logging infrastructure, handlers, and formats established in Phase 1
logger = logging.getLogger(__name__)

def execute_transactional_cleansing(df_source: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforms raw ledger entries into isolated, validated sales and cancellations tables.
    
    Args:
        df_source (pd.DataFrame): Raw ingested transaction logs sitting in the memory layer.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            [0]: pd.DataFrame - Cleansed, revenue-positive standard sales matrix.
            [1]: pd.DataFrame - Isolated, verified transaction cancellations archive.
        
    Raises:
        KeyError: If structural validation columns are absent from the incoming dataset.
        AssertionError: If cleansed data breaches post-transformation contract constraints.
    """
    logger.info("Initiating transactional cleansing pipeline...")
    
    # -------------------------------------------------------------------------
    # 1. PRE-ALLOCATION COMPLIANCE GATE
    # -------------------------------------------------------------------------
    # FAIL-FAST ARCHITECTURE: Scan schemas and verify structural prerequisites before 
    # executing high-memory copy allocations. Checking for missing keys up-front 
    # protects the runtime from mid-process crashes and keeps pipelines predictable.
    # -------------------------------------------------------------------------
    required_features = ['Customer ID', 'Price', 'Invoice', 'Quantity', 'InvoiceDate']
    for feature in required_features:
        if feature not in df_source.columns:
            logger.critical(f"Cleansing Aborted: Structural omission of target attribute: '{feature}'")
            raise KeyError(f"Pipeline Execution Aborted: Missing required column '{feature}'")

    # -------------------------------------------------------------------------
    # 2. IMMUTABLE RUNTIME STATE DECOUPLING
    # -------------------------------------------------------------------------
    # MEMORY ISOLATION PROTOCOL: Pandas defaults to creating shallow views when 
    # manipulating dataframes. To preserve the original source data frame as a 
    # read-only historical checkpoint, we force an explicit deep memory copy. 
    # This prevents any operations in this function from altering the raw dataset.
    # -------------------------------------------------------------------------
    working_df = df_source.copy()
    
    # -------------------------------------------------------------------------
    # 3. VECTORIZED FILTERING & TYPE STANDARDIZATION
    # -------------------------------------------------------------------------
    # STEP 3.1: Drop rows missing an explicit Customer ID mapping. Anonymous 
    # checkouts cannot be linked to distinct user profile nodes during downstream 
    # feature engineering or customer-centric vector similarity clustering.
    # -------------------------------------------------------------------------
    working_df = working_df.dropna(subset=['Customer ID'])
    
    # STEP 3.2: Filter out zero-cost items and negative adjustments. Enforcing 
    # a strict positive boundary on price arrays strips system errors, corporate 
    # giveaways, and bookkeeping corrections out of the operational data layer.
    working_df = working_df[working_df['Price'] > 0]
    
    # STEP 3.3: Type-cast the Invoice identifier column to a clean string format. 
    # Mixed-type parsing (integers combined with string characters) causes fatal 
    # data mismatches or type alignment errors during string searches.
    #
    # DESIGN PATTERN: Writing directly via '.loc[:, column]' targets the dataframe's 
    # underlying memory blocks, explicitly suppressing Pandas SettingWithCopyWarnings.
    working_df.loc[:, 'Invoice'] = working_df['Invoice'].astype(str)
    
    # -------------------------------------------------------------------------
    # 4. SEGREGATION & VECTOR SPLITTING (TRANSFORMATION FORK)
    # -------------------------------------------------------------------------
    # COMPUTE EFFICIENCY: Avoid utilizing slow Python-level row iteration loops. 
    # This block generates a high-efficiency boolean lookup array executed entirely 
    # via fast vector pathways to separate distinct transaction categories.
    #
    # BUSINESS LOGIC: Retail logging frameworks prefix return records with 'C'. 
    # We leverage this string marker to fork the data stream into two clear branches: 
    # active revenue-generating sales and post-purchase operational write-offs.
    # -------------------------------------------------------------------------
    is_cancellation = working_df['Invoice'].str.startswith('C', na=False)
    
    # Generate explicit, decoupled data matrices using isolated memory paths
    df_cancellations = working_df[is_cancellation].copy()
    df_sales_raw = working_df[~is_cancellation].copy()
    
    # STEP 4.1: Restrict the primary sales ledger to positive transaction volumes. 
    # This step filters out leftover returns or adjustments, leaving a clean set 
    # of successful purchase histories for downstream feature models.
    df_sales = df_sales_raw[df_sales_raw['Quantity'] > 0].copy()
    
    # -------------------------------------------------------------------------
    # 5. POST-CLEANSING INTEGRITY ASSERTERS (DATA CONTRACT INVARIANTS)
    # -------------------------------------------------------------------------
    # DEFENSIVE ENGINEERING: Never assume an upstream processing step worked. 
    # This block tests structural assumptions right before data leaves the module. 
    # If invalid records slip past our filters, these assertions trigger a hard 
    # system crash to prevent dirty data from reaching downstream modeling processes.
    # -------------------------------------------------------------------------
    try:
        assert df_sales['Customer ID'].isnull().sum() == 0, "Data Contract Violation: Null Customer IDs remain."
        assert (df_sales['Quantity'] <= 0).sum() == 0, "Data Contract Violation: Non-positive volumes remain in sales ledger."
        assert (df_sales['Price'] <= 0).sum() == 0, "Data Contract Violation: Non-positive pricing units remain."
    except AssertionError as validation_error:
        logger.critical(f"Pipeline Integrity Check Failed: {str(validation_error)}")
        raise validation_error

    # -------------------------------------------------------------------------
    # 6. TELEMETRY REPORTING UTILITY
    # -------------------------------------------------------------------------
    # FORMATTING SPECIFIER: The ':,' string format rule adds comma separators for 
    # thousands blocks, providing clean, readable production metrics when tracking 
    # system runs through cloud log management services.
    # -------------------------------------------------------------------------
    logger.info("--- CLEANING PIPELINE COMPLETE ---")
    logger.info(f"Cleaned Sales Database Volume : {df_sales.shape[0]:,} records verified.")
    logger.info(f"Isolated Cancellations Archive: {df_cancellations.shape[0]:,} records segregated.")
    
    return df_sales, df_cancellations


# =========================================================================
# 7. PIPELINE EXECUTION ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    # Keeps the file fully importable as an atomic library component without 
    # triggering accidental side effects or file writes during orchestration testing.
    logger.info("Cleansing pipeline module loaded standalone. System idling.")