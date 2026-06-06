# =========================================================================
# MODULE: Longitudinal Behavioral Feature Engineering Engine
# =========================================================================
# Operational Context: Downstream of Transform Segregation.
# Purpose: Compiles clean transactional ledgers into customer-level vectors
#          quantifying Recency, Frequency, and Monetary distributions.
#
# Architectural Strategy: Enforces strict data-type stability and functional
#                         immutability, preventing state contamination across
#                         the global notebook or application workspace.
# =========================================================================

import logging
import pandas as pd

# Inherits the logging infrastructure established in the orchestration root
logger = logging.getLogger(__name__)


def calculate_rfm_features(df_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms clean transactional logs into a schema-locked RFM feature matrix.
    
    Args:
        df_sales (pd.DataFrame): Cleansed, revenue-positive transaction database.
        
    Returns:
        pd.DataFrame: Structured, memory-optimized customer profile matrix.
        
    Raises:
        KeyError: If columns required for aggregation are missing from the schema.
        ValueError: If the incoming transaction dataframe contains zero records.
        AssertionError: If generated features violate analytical contract bounds.
    """
    logger.info("Initiating customer-level RFM feature engineering within module...")

    # -------------------------------------------------------------------------
    # 1. PRE-ALLOCATION COMPLIANCE GATE & EMPTY STATE FILTERS
    # -------------------------------------------------------------------------
    # FAIL-FAST ARCHITECTURE: Intercept structural or volume discrepancies before 
    # executing high-memory copy operations or grouping splits.
    # -------------------------------------------------------------------------
    required_features = ['Customer ID', 'InvoiceDate', 'Invoice', 'Quantity', 'Price']
    for feature in required_features:
        if feature not in df_sales.columns:
            logger.critical(f"Feature Generation Aborted: Missing required attribute: '{feature}'")
            raise KeyError(f"Pipeline Interrupted: Missing column '{feature}'")
            
    if df_sales.empty:
        logger.critical("Feature Generation Aborted: Incoming dataframe contains 0 observations.")
        raise ValueError("Pipeline Interrupted: Cannot engineer features on an empty transaction ledger.")

    # -------------------------------------------------------------------------
    # 2. IMMUTABLE RUNTIME DECOUPLING & METRIC VECTORIZATION
    # -------------------------------------------------------------------------
    # MEMORY ISOLATION PROTOCOL: Execute an explicit deep copy to isolate features 
    # and bypass SettingWithCopy warnings. Vector calculations run across contiguous 
    # memory arrays prior to group segregation for maximum processing speeds.
    # -------------------------------------------------------------------------
    working_df = df_sales.copy(deep=True)
    
    max_transaction_date = working_df['InvoiceDate'].max()
    anchor_date = max_transaction_date + pd.Timedelta(days=1)

    logger.info(f"Dataset Chronological Boundary Verified: {max_transaction_date}")
    logger.info(f"Calculative Anchoring Boundary Set  : {anchor_date}")

    # Build intermediate metrics via vectorized array pathways
    working_df.loc[:, 'LineTotal'] = working_df['Quantity'] * working_df['Price']
    working_df.loc[:, 'DaysSinceAnchor'] = (anchor_date - working_df['InvoiceDate']).dt.days

    # -------------------------------------------------------------------------
    # 3. HIGH-SPEED CYTHON-BASED GROUPBY AGGREGATION
    # -------------------------------------------------------------------------
    # COMPUTE EFFICIENCY: Avoid custom lambda groupings. NamedAgg links directly
    # to underlying C-optimized aggregators, reducing memory lookups.
    # -------------------------------------------------------------------------
    df_rfm = working_df.groupby('Customer ID').agg(
        Recency=pd.NamedAgg(column='DaysSinceAnchor', aggfunc='min'),
        Frequency=pd.NamedAgg(column='Invoice', aggfunc='nunique'),
        Monetary=pd.NamedAgg(column='LineTotal', aggfunc='sum')
    ).reset_index()

    # -------------------------------------------------------------------------
    # 4. FIELD REALIGNMENT & TYPING SCHEMA LOCKDOWN
    # -------------------------------------------------------------------------
    # DESIGN PATTERN: Renames key identifiers to standardize data consumption. 
    # Strict typing constraints compress the memory footprint and safeguard 
    # downstream vector space arrays from processing conflicts.
    # -------------------------------------------------------------------------
    df_rfm = df_rfm.rename(columns={'Customer ID': 'CustomerID'})

    schema_lock = {
        'CustomerID': 'int32',
        'Recency': 'int32',
        'Frequency': 'int32',
        'Monetary': 'float32'
    }
    df_rfm = df_rfm.astype(schema_lock)

    # -------------------------------------------------------------------------
    # 5. POST-AGGREGATION INTEGRITY ASSERTERS (DATA CONTRACT INVARIANTS)
    # -------------------------------------------------------------------------
    # DEFENSIVE ENGINEERING: Validates statistical distributions before releasing
    # vectors to scaling arrays. Breaks execution if anomalies crawl into metrics.
    # -------------------------------------------------------------------------
    try:
        assert df_rfm['CustomerID'].isnull().sum() == 0, "Contract Breach: Null CustomerIDs bypassed filter gates."
        assert (df_rfm['Recency'] < 0).sum() == 0, "Contract Breach: Negative temporal deltas generated."
        assert (df_rfm['Frequency'] <= 0).sum() == 0, "Contract Breach: Non-positive purchase intervals observed."
        assert df_rfm[['Recency', 'Frequency', 'Monetary']].notnull().all().all(), "Contract Breach: Missing metrics found."
    except AssertionError as contract_error:
        logger.critical(f"RFM Target Contract Failure: {str(contract_error)}")
        raise contract_error

    logger.info("Longitudinal behavioral feature matrix successfully locked and verified.")
    return df_rfm


# =========================================================================
# 6. PIPELINE RUNTIME ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    logger.info("Behavioral features engine module loaded standalone. System idling.")