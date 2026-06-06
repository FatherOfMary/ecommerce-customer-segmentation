# =========================================================================
# MODULE: Structural Anomaly Diagnostics Engine
# =========================================================================
# Operational Context: Part of the automated ingestion validation layer.
# Purpose: Executes highly parallelized, zero-loop structural audits across 
#          transaction data frames to export deterministic health payloads.
#
# Downstream Impact: Serves as a hard gatekeeper for analytical pipelines, 
#                    preventing out-of-bounds metrics from corrupting model 
#                    training instances or skewing analytics dashboards.
# =========================================================================

import sys
import logging
import pandas as pd

# Inherits the logging format and output stream rules initialized in Phase 1
logger = logging.getLogger(__name__)


def execute_baseline_diagnostic(df: pd.DataFrame) -> dict:
    """
    Performs a deterministic, vectorized diagnostic scan on the raw retail dataframe.
    
    Args:
        df (pd.DataFrame): The target data layer instance awaiting evaluation.
        
    Returns:
        dict: A telemetry payload containing structural metrics and orchestration gate states.
        
    Raises:
        KeyError: If mandatory analytical tracking features are absent from the schema.
        ValueError: If the input array is completely devoid of execution rows.
    """
    logger.info("Executing structural diagnostic scan on raw transactional features...")
    
    # -------------------------------------------------------------------------
    # 1. DEFENSIVE VALIDATION: INITIAL BOUNDARY CHECKS
    # -------------------------------------------------------------------------
    # FAIL-FAST DESIGN PATTERN: Intercept bad data states before allocating memory
    # for full validation sweeps. Verifying column targets early prevents uncaught 
    # KeyErrors during vector operations, while halting on empty datasets blocks 
    # zero-division math errors during downstream percentage calculations.
    # -------------------------------------------------------------------------
    required_columns = ['Customer ID', 'Quantity', 'Price']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        error_msg = f"Critical schema violation. Missing required diagnostic features: {missing_cols}"
        logger.critical(error_msg)
        raise KeyError(error_msg)
        
    total_records = len(df)
    if total_records == 0:
        logger.error("Diagnostic execution aborted: Input DataFrame contains 0 records.")
        raise ValueError("Diagnostic execution aborted: Input DataFrame contains 0 records.")

    # -------------------------------------------------------------------------
    # 2. VECTORIZED ANOMALY EXTRACTION (HIGH-PERFORMANCE MASKING)
    # -------------------------------------------------------------------------
    # COMPUTE OPTIMIZATION: Row-by-row iteration paradigms are completely prohibited. 
    # These expressions generate contiguous boolean arrays processed inside optimized 
    # C-compiled vector blocks, executing instantly across millions of ledger elements.
    #
    # SERIALIZATION GUARDRAIL: Pandas/NumPy vector operations natively output custom 
    # types like 'np.int64'. We explicitly wrap outputs in native Python 'int()' because 
    # custom NumPy data types trigger fatal serialization errors if this payload is 
    # forwarded to standard JSON endpoints, NoSQL document databases, or remote APIs.
    # -------------------------------------------------------------------------
    missing_cust_mask = df['Customer ID'].isnull()
    neg_quantity_mask = df['Quantity'] < 0
    invalid_price_mask = df['Price'] <= 0

    # Bitwise scalar population tracking executed via fast C-level aggregations
    counts = {
        "missing_customers": int(missing_cust_mask.sum()),
        "negative_quantities": int(neg_quantity_mask.sum()),
        "invalid_prices": int(invalid_price_mask.sum())
    }

    # -------------------------------------------------------------------------
    # 3. STRUCTURED METADATA REPORTING LAYOUT
    # -------------------------------------------------------------------------
    # DESIGN CHOICE: Formats an isolated, unmutated metadata contract compatible with 
    # metrics monitors (e.g., Datadog, Prometheus) or validation tools (e.g., Great Expectations).
    #
    # PIPELINE GATE MARGIN: The 'system_health_compromised' boundary sets an explicit 
    # threshold at 25% anomalies. If inventory drops or cancellations exceed this, 
    # it indicates data engineering ingestion corruption, causing remote runners to halt.
    # -------------------------------------------------------------------------
    anomaly_report = {
        "metrics": {
            "missing_customers": {
                "count": counts["missing_customers"],
                # Explicit conversion to float ensures native Python JSON compatibility
                "percentage": float((counts["missing_customers"] / total_records) * 100)
            },
            "negative_quantities": {
                "count": counts["negative_quantities"],
                "percentage": float((counts["negative_quantities"] / total_records) * 100)
            },
            "invalid_prices": {
                "count": counts["invalid_prices"],
                "percentage": float((counts["invalid_prices"] / total_records) * 100)
            }
        },
        "pipeline_metadata": {
            "total_records_evaluated": total_records,
            "system_health_compromised": counts["negative_quantities"] > (total_records * 0.25)
        }
    }

    # -------------------------------------------------------------------------
    # 4. STANDARD OUTPUT LOGS
    # -------------------------------------------------------------------------
    # FORMATTING PROTOCOLS: Human-readability configurations applied explicitly. 
    # The ':,' specifier forces comma-separated formatting for thousands blocks, 
    # and ':.2f' truncates long float tails. This guarantees that logs remain clear 
    # and readable when scanned in terminal environments or cloud dash consoles.
    # -------------------------------------------------------------------------
    logger.info("--- DETECTED DATA ANOMALIES LOG ---")
    logger.info(
        f"Missing Customer IDs (Guest checkouts) : "
        f"{anomaly_report['metrics']['missing_customers']['count']:,} rows "
        f"({anomaly_report['metrics']['missing_customers']['percentage']:.2f}%)"
    )
    logger.info(
        f"Negative Quantities (Cancellations/Returns): "
        f"{anomaly_report['metrics']['negative_quantities']['count']:,} rows "
        f"({anomaly_report['metrics']['negative_quantities']['percentage']:.2f}%)"
    )
    logger.info(
        f"Invalid Unit Prices (System errors/Freebies): "
        f"{anomaly_report['metrics']['invalid_prices']['count']:,} rows "
        f"({anomaly_report['metrics']['invalid_prices']['percentage']:.2f}%)"
    )

    return anomaly_report


# =========================================================================
# 5. PIPELINE EXECUTION ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    # Provides clean operational isolation. If called standalone during continuous 
    # integration testing or routine systems checking, it registers safely without side effects.
    logger.info("Diagnostics utility script loaded standalone. System idling.")