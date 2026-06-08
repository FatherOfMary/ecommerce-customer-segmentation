# E-Commerce Customer Segmentation Engine

## Abstract
This project presents an end-to-end machine learning pipeline that transforms raw transactional data into structured, actionable business intelligence. Utilizing the classic Online Retail II dataset, the framework aggregates transactional logs into individual Recency, Frequency, and Monetary (RFM) vectors. To overcome severe data skew and outlier distortion typical of purchase histories, data undergoes systematic cleansing, Winsorization, and log-transformation before feature scaling. Unsupervised learning via the K-Means clustering algorithm isolates distinct customer behavioral profiles. The entire pipeline is engineered as an enterprise-grade, modular system architecture and served via an interactive Streamlit analytics platform for real-time strategic decision making.

---

## Background
In modern e-commerce, generic marketing campaigns yield low conversion rates and poor resource allocation. Customer retention and relationship management require deep behavioral understanding. Transactional systems continuously generate extensive invoice ledgers containing purchase history, quantity, price, and customer identifiers. However, these raw logs are unorganized. By applying behavioral framework metrics, specifically tracking how recently, how frequently, and how much money a customer spends, companies can extract deep behavioral patterns from historical transaction sequences.

---

## Problem Statement
Unstructured transactional databases hide customer behavior patterns. Mass marketing to an entire customer base overlooks critical variations in engagement, loyalty, and risk of churn. High-volume buyers require different retention strategies than fading, low-frequency accounts. Without automated, reproducible clustering mechanisms, data analytics teams struggle to scale customer insights, leading to misallocated marketing budgets and missed revenue opportunities.

---

## Objectives / Purpose
* **Pipeline Automation:** Build a scalable, modular data pipeline that ingests raw invoices, processes data features, and trains model clusters without manual script intervention.
* **Algorithmic Profiling:** Implement K-Means clustering optimized by Elbow and Silhouette metrics to separate an e-commerce customer base into distinct cohorts.
* **Operational Serving:** Deploy an intuitive Streamlit application interface allowing stakeholders to inspect cluster centers, visualize distributions, and track segment metrics dynamically.

---

## Benefits / Significance
* **Marketing ROI Optimization:** Enables precise, segment-specific campaigns (such as VIP loyalty rewards or automated win-back emails for churning cohorts) that maximize financial returns.
* **Engineering Cleanliness:** Demonstrates production-level codebase maturity by replacing monolithic notebook workflows with decoupled configuration, data processing, feature engineering, and modeling scripts.
* **Reproducible Design:** Standardizes data schemas and transformation states (`.parquet` and `.joblib`), ensuring identical processing results across development and production execution environments.

---

## Scope of Problem / Limitations
* **Temporal Boundary:** The scope is bounded by the historical limits of the underlying dataset, tracking transactional data from 2009 to 2011.
* **Feature Limits:** Clustering relies purely on engineering quantitative behavioral interactions. It completely excludes qualitative user attributes, such as age, location, gender, or web clickstream logs.
* **Algorithm Constraints:** K-Means assumes spherical cluster geometry and remains highly sensitive to extreme data boundaries and initial centroid seeding.

---

## Hypothesis
Transforming raw, highly skewed transactional data into normalized, log-scaled RFM vectors will allow the distance-based K-Means algorithm to isolate statistically distinct, non-overlapping customer cohorts that map to actionable business personas.

---

## METHODOLOGY AND SYSTEM DESIGN

The system uses a completely decoupled architecture. Each phase of the workflow passes verified data artifacts to the next directory layer.

```text
├── config/
│   ├── __init__.py
│   ├── environment.py             # Infrastructure and pathway definitions
│   └── settings.py                # Model hyperparameter configurations
├── data/
│   ├── raw/                       # Unmodified invoice ledgers
│   ├── processed/                 # Cleansed and transformed pipelines
│   │   ├── df_cancellations.csv
│   │   ├── df_sales.csv
│   │   ├── rfm_baseline.parquet
│   │   ├── rfm_scaled_modeling.parquet
│   │   ├── rfm_transformed_modeling.parquet
│   │   └── standard_scaler.joblib # Serialized scaling parameters
│   ├── __init__.py
│   ├── cleansing.py               # Handles missing values and invoice separation
│   └── ingestion.py               # Data loading protocols from source
├── features/
│   ├── __init__.py
│   ├── rfm.py                     # Aggregates raw transactions into metrics
│   ├── scale.py                   # Implements Z-score normalization
│   └── transform.py               # Manages log and Winsorization workflows
├── models/
│   ├── __init__.py
│   ├── centroid_decoding.py       # Translates vectors into business profiles
│   ├── cluster_assignment.py      # Infers cluster states for new data points
│   ├── cluster_tuning.py          # Iteratively measures WCSS and Silhouette paths
│   ├── cluster_visualization.py   # Renders 3D Plotly spatial distribution files
│   └── profile_synthesis.py       # Aggregates business descriptive metrics
├── notebooks/
│   └── 1.0-eda-and-transformation.ipynb # Exploratory analysis sandbox
├── reports/
│   └── figures/                   # Persisted diagnostic plotting graphics
│       ├── cluster_optimization_validation_20260605_215454.png
│       └── rfm_transformation_diagnostics_20260605_215339.png
├── src/
│   └── app.py                     # Interactive Streamlit runtime engine
├── utils/
│   ├── __init__.py
│   ├── diagnostics.py             # Distribution check and asymmetry analysis helpers
│   ├── io_helpers.py              # Centralized disk reader and writer systems
│   └── viz_helpers.py             # Layout and presentation styling sheets
└── venv/                          # Isolated virtual environment runtime

```

### Execution Flow Sequence

1. **Data Preparation:** `data/ingestion.py` extracts raw transaction ledgers, while `data/cleansing.py` filters invalid IDs and separates successful transactions from cancellations (`df_sales.csv` and `df_cancellations.csv`).
2. **Feature Generation:** `features/rfm.py` transforms long invoice records into wide rows indexed by individual customers, establishing Recency, Frequency, and Monetary baseline scores (`rfm_baseline.parquet`).
3. **Mathematical Processing:** `features/transform.py` corrects distribution skewness using log transforms and limits outlier impacts through Winsorization. The results are standardized via `features/scale.py` and saved as mathematical arrays alongside a serialized configuration file (`standard_scaler.joblib`).
4. **Algorithmic Modeling:** `models/cluster_tuning.py` analyzes the dataset's clustering structure. Once the target cluster size ($K$) is determined, `models/cluster_visualization.py` maps the spatial layout while `models/centroid_decoding.py` translates the resulting vectors back into meaningful business groups.
5. **Dashboard Serving:** `src/app.py` sources the fully evaluated artifacts, initializing an interactive analytics console for business user workflows.

---

## Analysis

E-commerce purchase patterns inherently contain extreme right-skewed distributions. A tiny fraction of users buy goods at massive volumes, while the vast majority generate low-frequency, low-monetary transactions.

Distance calculations in K-Means assume normally distributed, evenly scaled metrics. To ensure valid distances, the pipeline uses a strict preparation workflow:

* **Outlier Strategy:** Extreme values are capped at the 99th percentile using Winsorization to prevent massive orders from pulling cluster centers out of alignment.
* **Distribution Balancing:** Logarithmic transformations flatten wide transactional distributions into standard normal curves.
* **Feature Balancing:** Z-score normalization converts all feature columns to an identical numeric scale (mean = 0, variance = 1). This ensures that large monetary amounts do not mathematically overpower short recency durations during model training.

---

## Results

The mathematical pipeline outputs clean, distinctly separated groups without spatial overlaps. By matching cluster centers with their original business values via `models/centroid_decoding.py`, the engine automatically identifies distinct operational cohorts:

| Segment Designation | Recency Performance | Frequency Performance | Monetary Performance | Core Operational Focus |
| --- | --- | --- | --- | --- |
| **Champions** | Extremely Low (Recent) | Exceptionally High | Exceptionally High | Loyalty Programs / Pre-release access |
| **Loyal Customers** | Low | High | High | Upselling premium options |
| **At Risk / Hibernating** | High (Long Ago) | Low | Low | Automated win-back campaigns |
| **New Inclusions** | Extremely Low (Recent) | Low | Low | Onboarding engagement paths |

The pipeline successfully saves these model diagnostics directly to the file system as reference points:

* `reports/figures/rfm_transformation_diagnostics_20260605_215339.png`
* `reports/figures/cluster_optimization_validation_20260605_215454.png`

---

## Analysis of Test Results

Cluster configuration quality is validated using two clear mathematical criteria:

1. **Within-Cluster Sum of Squares (Elbow Method):** Tracks variance reduction as the cluster count ($K$) increases. The optimal model size is chosen at the visible structural inflection point, where adding more clusters yields diminishing returns.
2. **Silhouette Coefficients:** Evaluates how close each data point is to its assigned cluster relative to neighboring groups. High, uniform silhouette widths across all cohorts confirm clean, distinct boundaries and prove the model's structural stability.

---

## CONCLUSION

This project successfully transitions customer segmentation from an ad-hoc notebook exercise to a structured, production-ready system architecture. By decoupling raw ingestion pipelines, engineering distinct data transformations, and implementing a solid clustering framework, the codebase provides reliable customer analysis. The system successfully structures messy invoice ledgers into clear behavioral trends, and the interactive Streamlit dashboard makes these machine learning insights immediately useful for business execution.

---

## Suggestions / Recommendations

* **Incorporate Lifespan Metrics:** Integrate Customer Lifetime Value (CLV) calculations directly into the profile synthesis module to project long-term customer worth.
* **Automate Pipeline Retraining:** Deploy cron scheduling configurations to automatically pull fresh transaction logs, refresh cluster boundaries, and update cohort assignments over time.
* **Downstream Integration:** Connect the model's cluster assignment outputs to third-party marketing tools (such as Braze or Mailchimp) to power real-time, personalized messaging campaigns.

---

## Installation & Usage Instructions

### Prerequisites

Ensure Python 3.14+ and a virtual environment tool are available on your system.

### 1. Clone the Repository

```powershell
git clone git@github.com:YOUR_USERNAME/ecommerce-customer-segmentation.git
cd ecommerce-customer-segmentation

```

### 2. Configure Virtual Environment

Using PowerShell (pwsh) within VS Code:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

```

### 3. Install Required Dependencies

```powershell
pip install -r requirements.txt

```

### 4. Run the Streamlit Interface

```powershell
streamlit run src/app.py

```

---

## REFERENCES

1. Online Retail II Dataset, UCI Machine Learning Repository. Available at: [https://archive.ics.uci.edu/dataset/502/online+retail+ii](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
2. Kaggle Version Hosting, Online Retail II Dataset. Available at: [https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data)