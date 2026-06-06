# =========================================================================
# MODULE: Data Ingestion & Schema Initialization Engine
# =========================================================================
# Purpose: Programmatically fetches, caches, and memory-optimizes raw
#          longitudinal transactional datasets from remote repositories.
#
# Constraints: Enforces absolute data-type typing to minimize operational RAM 
#              footprints across large-scale data arrays.
# =========================================================================

import os
import logging
import kagglehub
import pandas as pd

# Standard module-level logger. Automatically inherits the formatting matrix 
# and stream routing rules established by the root settings initializer.
logger = logging.getLogger(__name__)


def load_and_optimize_retail_data(
    dataset_identifier: str = "mashlyn/online-retail-ii-uci", 
    filename: str = "online_retail_II.csv"
) -> pd.DataFrame:
    """
    Downloads the dataset from Kaggle, enforces a strict schema matrix to minimize 
    memory consumption, and validates file existence.
    
    Args:
        dataset_identifier (str): Remote directory token located on Kaggle repositories.
        filename (str): Target transactional file to be located within cached storage.

    Returns:
        pd.DataFrame: Schema-enforced, memory-optimized dataframe.
    """
    logger.info("Initializing remote data ingestion workflow via KaggleHub...")
    
    # -------------------------------------------------------------------------
    # 1. SECURE DATA ACQUISITION LAYER
    # -------------------------------------------------------------------------
    try:
        # Fetches and returns the absolute local path to the latest dataset version cache
        cache_dir = kagglehub.dataset_download(dataset_identifier)
        csv_path = os.path.join(cache_dir, filename)
        
        # Defensive File System Gate: Catches missing file variances instantly
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Target database file '{filename}' was not detected inside cache: {cache_dir}"
            )
            
        logger.info(f"Dataset artifact successfully verified on disk at: {csv_path}")
        
    except Exception as e:
        logger.error(f"Critical failure during dataset download/location validation: {str(e)}")
        raise
        
    # -------------------------------------------------------------------------
    # 2. DATABASE-STYLE SCHEMA DEFINITION
    # -------------------------------------------------------------------------
    # DESIGN CHOICE: Default pandas string object allocations are highly inefficient. 
    # Downcasting integers and floating-point parameters to 32-bit representations 
    # compresses the operational system RAM requirement by up to 50%, enabling 
    # stable vector calculations on standard local work environments.
    # -------------------------------------------------------------------------
    schema_definition = {
        'Invoice': 'object',      
        'StockCode': 'object',    
        'Description': 'object',
        'Quantity': 'int32',       # Downcasted from int64 to preserve room
        'Price': 'float32',        # Downcasted from float64
        'Customer ID': 'float32'   # Floats handle optional null elements natively
    }
    
    # -------------------------------------------------------------------------
    # 3. DISK READ OPERATION & ANALYTICAL TELEMETRY
    # -------------------------------------------------------------------------
    try:
        logger.info("Executing schema-enforced disk read operation...")
        
        df = pd.read_csv(
            csv_path,
            dtype=schema_definition,
            parse_dates=['InvoiceDate'],
            # Bypasses the expensive fallback datetime parsing logic for optimized speeds
            date_format="ISO8601"  
        )
        
        # Calculate precise structural metrics for execution verification log reports
        initial_rows, initial_cols = df.shape
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        
        logger.info("Data ingestion phase complete.")
        logger.info(f"Ledger Matrix Properties -> Records: {initial_rows:,} | Attributes: {initial_cols}")
        logger.info(f"Engine RAM Allocation Profile: {memory_usage_mb:.2f} MB")
        
        return df

    except Exception as e:
        logger.error(f"Inference or parsing error during transactional CSV compilation: {str(e)}")
        raise


# =========================================================================
# 4. RUNTIME SYSTEM ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    # Standard fallback test execution loop when script is evaluated independently
    df_raw = load_and_optimize_retail_data()