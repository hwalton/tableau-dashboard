# Tableau dashboard: California housing (ML-oriented)

Workbook **[`Book.twb`](Book.twb)** explores the classical **California housing** dataset (block groups × census-style features) with exploratory views plus ML-style correlation and distribution summaries.

## Regenerating data

From the project root (with a virtualenv that has the dependencies in [`requirements.txt`](requirements.txt)):

```bash
python get-data.py
```

That script:

- Downloads the dataset via **scikit-learn** and writes **`california_housing.csv`**
- Writes **KPI** and **ML sidecar** CSVs Tableau uses (`kpi_summary.csv`, `feature_correlations_full.csv`, `feature_histogram_bins.csv`, etc.)

Reopen or refresh the workbook in Tableau Desktop after running it so file-based sources stay in sync.

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

| Sheet | What it shows |
|-------|----------------|
| **Sheet 1** | Default workbook placeholder (multi-measure text table). Safe to ignore or delete once you no longer need it. |
| **Drivers - Income vs Value** | Scatter of **`MedInc`** vs **`MedHouseVal`** (row-level block groups), colored by **`Coastal Flag`**. Good for seeing the income–value relationship and coastal vs inland separation. |
| **Residuals Map** | Grid of **`Lat Bin` / `Lon Bin`** with color = **`Value Residual`** (block median **`MedHouseVal`** minus coastal vs inland LOD averages). Palette is **Red–Blue Diverging reversed** so **warmer/red = above belt average** (pricier) and cooler/blue = below. Dashboard adds a **color legend** beside the map plus a short text note; **`MedHouseVal`** stays in **$100k** units. |
| **Drivers - Age vs Value** | **`HouseAge`** vs average **`MedHouseVal`**, colored by **`Income Tier`**. |
| **Affordability by Region** | **`Income Tier`** vs average **`Price-to-Income Ratio`** (`MedHouseVal / MedInc`), colored by **`Coastal Flag`**. |
| **KPI summary** | Small table from **`kpi_summary.csv`**: block group count and dataset-wide averages (house value, income, price-to-income). Matches the KPI strip intent in one place. |
| **ML - Correlation matrix** | Heatmap from **`feature_correlations_full.csv`**: **Pearson** correlation between numeric columns (`MedInc`, ages, rooms, geography, **`MedHouseVal`**, etc.). Color scale maps correlation strength; check the legend when opened in Tableau. |
| **ML - Profiles and distributions** | Density-style view from **`feature_histogram_bins.csv`**: rows = **`Feature`**, columns = binned **`Value_bin_mid`**, color = **`BIN_COUNT`**. Tooltip carries **`pandas_dtype`** and **`skew_label`** from the profiling step. Extremely skewed features (e.g. long tails) may put most counts in few bins—that reflects the marginal distribution, not a Tableau bug. |

---

## Dashboard

**California Housing Dashboard** lays out **`KPI summary`** across the top and pairs of analysis sheets below (**Drivers / Residuals** + residuals **legend and caption**, then **Age / Affordability**). The ML worksheets are tabs only unless you drag them onto the dashboard in Desktop.

Recommended follow-ups entirely in Tableau (no repo change required):

- Add **filter actions** (e.g. map → scatter) via **Dashboard → Actions**
- Drag **`Min Population`** (parameter) onto the dashboard if you use it as a filter
- Tune color palettes (diverging for residuals and correlations; sequential for histogram counts)

---

## File map

| File | Role |
|------|------|
| [`Book.twb`](Book.twb) | Tableau workbook (XML). Paths point at **`/Users/walton/...`**; relocate the CSV folder or reconnect if you clone on another machine. |
| [`get-data.py`](get-data.py) | Single script: fetch + KPI + ML CSV exports. |
| `california_housing.csv` | Main fact table (from sklearn). |
| `kpi_summary.csv` | Two-column KPI table for **KPI summary**. |
| `feature_correlations_full.csv` | Long-form correlation matrix for **ML - Correlation matrix**. |
| `feature_histogram_bins.csv` | Binned counts + profile strings for **ML - Profiles and distributions**. |
| `feature_profiles.csv` | One row per column (dtype, skew, quantiles); useful for documentation or future sheets. |

---

## References

- R. Kelley Pace and Ronald Barry, *Sparse Spatial Autoregressions*, Statistics and Probability Letters, 33:291–297, 1997.  
- [scikit-learn: California housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
