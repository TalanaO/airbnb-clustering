# Airbnb Clustering Analysis

## Overview

This repository contains a Python script (`airbnb-clustering.py`) for loading, cleaning, preprocessing, and clustering Airbnb listing data. It supports:

- Column inspection
- Data loading with duplicate removal
- Data cleaning & feature engineering (prices, amenities, host experience, etc.)
- Outlier detection & removal
- Imputation method comparison (mean, median, KNN)
- Visualisations (price distributions, imputation comparisons, before/after outlier removal)
- Clustering (e.g., K-Means, PCA)

## Achievements

- Graded 90/100 on this project at University!


## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data

**Note:** The `.csv` dataset is **not** included in this repository because it exceeds the upload size limit. Please download your Airbnb listings CSV (e.g., `listings.csv`) separately and place it in the project root.

## Usage

### 1. Inspect columns

```bash
python airbnb-clustering.py --check-columns listings.csv
```

### 2. Load & clean data

```python
from airbnb_clustering import load_dataset, clean_data

df = load_dataset("listings.csv")
df = clean_data(df)
```

### 3. Remove outliers

```python
from airbnb_clustering import remove_outliers

df_clean = remove_outliers(df, columns=["price"])
```

### 4. Compare imputation methods

```python
from airbnb_clustering import compare_imputation_methods

results_df = compare_imputation_methods(df_clean)
results_df.to_csv("imputation_comparison_results.csv", index=True)
```

### 5. Generate visualisations

- `imputation_comparison.png`
- `price_before_after_outliers.png`
- Other distribution plots saved to `./plots/`

### 6. Clustering & PCA

Add your clustering calls (e.g., `KMeans`, `PCA`) in the script or an interactive notebook.

## Project Structure

```
.
├── airbnb-clustering.py     # Main data processing & analysis script
├── requirements.txt         # Python package requirements
├── README.md               # This file
└── listings.csv            # (Not included; see Data section)
```

## Contact

For questions or contributions, please open an issue or submit a pull request.
