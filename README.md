# Tableau dashboard: California housing (ML-oriented)

Workbook **[`Book.twb`](Book.twb)** walks through the **Géron Chapter 2** end-to-end EDA workflow on the classical **California housing** dataset: **head / describe / hist / map / correlate**.

## Regenerating data

From the project root (with a virtualenv that has the dependencies in [`requirements.txt`](requirements.txt)):

```bash
python get-data.py
```

That script:

- Downloads the dataset via **scikit-learn** and writes **`california_housing.csv`**
- Writes the **EDA sidecar** CSVs Tableau uses (`housing_head_long.csv`, `housing_head.csv`, `feature_describe.csv`, `feature_cells_stats.csv`, `medhouseval_histogram.csv`, `housing_geo_map.csv`, `feature_correlations_full.csv`, etc.)

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
| **Describe table** | `housing.info()` / `housing.describe()` | Text crosstab from **`feature_cells_stats.csv`** (Rows = `Feature`, Cols = `stat_rank / stat`, Text = **Attribute**(cell)): describe metrics plus skew. **`feature_describe.csv`** is the wide table (includes `dtype`, null counts, and describe stats together) if you prefer a spreadsheet. |
| **MedHouseVal histogram** | `Series.hist` / bar counts | **`medhouseval_histogram.csv`** has **`bin_index`**, **`value_bin_mid`**, **`count`** only: forty equal-width **`pandas.cut`** bins over **`MedHouseVal`** alone (sklearn \$100k units; no KDE). **Rows** = **`SUM(count)`**, **Cols** = **`Bin index`** (discrete ordinal), **`Bar`** marks, tooltip **`AVG(value_bin_mid)`** for the midpoint in each bin. |
| **Med house value heat map** | spatial EDA | One row per block group from **`housing_geo_map.csv`**: **`Latitude`** / **`Longitude`** on the axes, circles colored by **`MedHouseVal_pct`** (pandas **percentile rank** in \([0,1]\)—spreads colors when **`MedHouseVal`** stacks at the census cap—see README **Target and units**), tooltip **`MedHouseVal`** in sklearn units (\$100k). Use **Map** → map layers / background maps in Tableau as needed after opening. |
| **ML - Correlation matrix** | feature inter-correlations | Heatmap from `feature_correlations_full.csv` with `Pearson_r` on color and label. |

---

## Dashboard

**California Housing Dashboard** stacks the workflow vertically:

1. Header text strip naming the five workflow beats.
2. **Data preview (head)** full width.
3. **Describe table** (full width).
4. **MedHouseVal histogram** (full width).
5. **Med house value heat map** (full width).
6. **ML - Correlation matrix** (full width, with its color legend on the side).

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
| `feature_describe.csv` | One row per feature with dtype + null counts + describe stats (+ skew metrics); spreadsheet-friendly mirror of the summaries. |
| `feature_cells_stats.csv` | Long/narrow slice for **`Describe table`** (describe + skew metrics as pre-formatted strings in `cell`). |
| `medhouseval_histogram.csv` | **`bin_index`**, **`value_bin_mid`**, **`count`** for **`MedHouseVal histogram`** (single bar chart over the target). |
| `housing_geo_map.csv` | Block-group lat/lon + **`MedHouseVal`** + **`MedHouseVal_pct`** (percentile rank) for **`Med house value heat map`**. |
| `feature_correlations_full.csv` | Long-form correlation matrix for **ML - Correlation matrix**. |
| `feature_profiles.csv` | One row per column (dtype, skew, quantiles); kept as a profiling artefact. |

---

## References

- R. Kelley Pace and Ronald Barry, *Sparse Spatial Autoregressions*, Statistics and Probability Letters, 33:291–297, 1997.  
- [scikit-learn: California housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
