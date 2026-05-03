# Tableau dashboard: California housing (ML-oriented)

Workbook **[`Book.twb`](Book.twb)** walks through the **Géron Chapter 2** end-to-end EDA workflow on the classical **California housing** dataset: **head / info / describe / hist / ratios / correlate**.

## Regenerating data

From the project root (with a virtualenv that has the dependencies in [`requirements.txt`](requirements.txt)):

```bash
python get-data.py
```

That script:

- Downloads the dataset via **scikit-learn** and writes **`california_housing.csv`**
- Writes the **EDA sidecar** CSVs Tableau uses (`housing_head_long.csv`, `housing_head.csv`, `feature_describe.csv`, `feature_cells_schema.csv`, `feature_cells_stats.csv`, `feature_histogram_bins.csv`, `housing_ratio_features.csv`, `housing_ratio_scatter.csv`, `feature_correlations_full.csv`, etc.)

Reopen or refresh the workbook in Tableau Desktop after running it so file-based sources stay in sync.

**Connection errors (“file … not found”):** The workbook expects CSVs beside `Book.twb`. **`*.csv` is gitignored**—CSV outputs are **not** in git; run **`python get-data.py`** after clone before opening Tableau. If you moved the folder, use **File → Locate in Folder** / **Edit connection** so the directory matches your working copy.

---

## Target and units (read this before interpreting charts)

| Column (CSV) | Meaning |
|--------------|--------|
| **`MedHouseVal`** | Median house value for the block group, **not in dollars**. In sklearn’s version it is stored in **units of \$100,000**. Example: `2.07` ≈ **\$207,000**; `5` ≈ **\$500,000**. |

### Why there is a spike at the top (~5)

The underlying **1990 California census** reporting used for this dataset **caps (top-codes)** median home values at about **\$500,000**. Block groups with a true median above that still appear **at the ceiling**. You will see many rows near **`5`** (and values like **`5.00001`** from floating-point scaling after dividing raw medians by 100,000).

**For modeling:** treat the upper tail as **censored**, not a dense “true” cluster of identical prices.

Other fields are documented in the Pace & Barry (1997) / sklearn dataset description; `MedInc` is typically described in **\$10,000** units (tens of thousands of dollars).

---

## Worksheets (`Book.twb`)

Each sheet maps to a step in the textbook EDA recipe.

| Sheet | Textbook analogue | What it shows |
|-------|-------------------|----------------|
| **Data preview (head)** | `housing.head()` | First 5 rows in **long format** (`housing_head_long.csv`): **Rows** = `row_id`, **Cols** = `measure`, **Text** = `SUM(value)` — avoids fragile `Measure Names` / `Multiple Values` XML. |
| **Schema (info)** | `housing.info()` | Text crosstab from **`feature_cells_schema.csv`**: Rows = `Feature`, Cols = `stat_rank / stat`, Text = `ATTR(cell)`. Skew is intentionally absent here (`dtype`, `non_null_*`, counts only). |
| **Describe table** | `housing.describe()` (+ skew summary) | Text crosstab from **`feature_cells_stats.csv`**: same grid pattern plus `skewness`, `skew_label`, and numeric describe metrics (avoids brittle `Measure Names` on the raw wide CSV). Wide **`feature_describe.csv`** stays for spreadsheets. |
| **Histograms with skew flags** | `housing.hist(bins=50)` | Heatmap from `feature_histogram_bins.csv`: rows = `Feature` (with `skew_label`), columns = uniform per-feature **`bin_index` 0..N-1**, color = `SUM(BIN_COUNT)`. The shared discrete x-axis is what makes every feature actually render as a small multiple. |
| **Ratio features** | `bedrooms_ratio = AveBedrms / AveRooms` | Histogram of the only genuinely new ratio (sklearn's `AveRooms` and `AveOccup` already encode the textbook's `rooms_per_house` and `people_per_house`). |
| **Ratio scatter** | feature vs target | `bedrooms_ratio` vs `MedHouseVal` raw scatter from `housing_ratio_scatter.csv`. |
| **ML - Correlation matrix** | feature inter-correlations | Heatmap from `feature_correlations_full.csv` with `Pearson_r` on color and label. |

---

## Dashboard

**California Housing Dashboard** stacks the workflow vertically:

1. Header text strip naming the six stages.
2. **Data preview (head)** full width.
3. **Schema (info)** | **Describe table**.
4. **Histograms with skew flags** (full width).
5. **Ratio features** | **Ratio scatter** | **ML - Correlation matrix** (with its color legend on the side).

Phone layout uses the same order as a vertical scroll.

---

## File map

| File | Role |
|------|------|
| [`Book.twb`](Book.twb) | Tableau workbook (XML). Paths point at **`/Users/walton/...`**; relocate the CSV folder or reconnect if you clone on another machine. |
| [`get-data.py`](get-data.py) | Single script: fetch + EDA-sidecar CSV exports. |
| `california_housing.csv` | Main fact table (from sklearn). |
| `housing_head_long.csv` | Molten first 5 rows (`row_id`, `measure`, `value`) powering **Data preview (head)**. |
| `housing_head.csv` | Same head in **wide** form (optional inspection in a spreadsheet). |
| `feature_describe.csv` | One row per feature with dtype + describe stats (+ skew metrics); spreadsheet-friendly mirror of the summaries. |
| `feature_cells_schema.csv` | Long/narrow `(Feature, stat_rank, stat, cell)` slice for **`Schema (info)`** (info-only columns). |
| `feature_cells_stats.csv` | Long/narrow slice for **`Describe table`** (describe + skew metrics as pre-formatted strings in `cell`). |
| `feature_correlations_full.csv` | Long-form correlation matrix for **ML - Correlation matrix**. |
| `feature_histogram_bins.csv` | Binned counts + profile strings + `bin_index` for **Histograms with skew flags**. |
| `housing_ratio_features.csv` | Histogram of `bedrooms_ratio` for **Ratio features**. |
| `housing_ratio_scatter.csv` | Raw `bedrooms_ratio, MedHouseVal` for **Ratio scatter**. |
| `feature_profiles.csv` | One row per column (dtype, skew, quantiles); kept as a profiling artefact. |

---

## References

- R. Kelley Pace and Ronald Barry, *Sparse Spatial Autoregressions*, Statistics and Probability Letters, 33:291–297, 1997.  
- [scikit-learn: California housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
