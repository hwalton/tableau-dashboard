"""Fetch California housing from sklearn and write raw CSV plus Tableau/ML sidecar files."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def skew_label(skewness: float) -> str:
    if abs(skewness) < 0.5:
        return "~ symmetric"
    if skewness >= 1.5:
        return "strong right tail"
    if skewness >= 0.5:
        return "right-skew"
    if skewness <= -1.5:
        return "strong left tail"
    return "left-skew"


def build_ml_assets(df: pd.DataFrame, root: Path) -> None:
    """Correlation matrix, profiles, melted numeric features."""
    num = df.select_dtypes(include=["number"]).copy()

    melted = (
        num.melt(var_name="Feature", value_name="Value")
        .dropna(subset=["Value"])
        .astype({"Feature": str})
    )
    melted.to_csv(root / "housing_features_long.csv", index=False)

    c = num.corr(method="pearson", numeric_only=True)
    idx = np.tril_indices_from(c.values)
    pd.DataFrame(
        {
            "Variable_X": c.index[idx[0]].astype(str),
            "Variable_Y": c.columns[idx[1]].astype(str),
            "Pearson_r": c.values[idx],
        }
    ).to_csv(root / "feature_correlations_tri.csv", index=False)

    long_full = c.stack().rename("Pearson_r").reset_index()
    long_full.columns = ["Variable_X", "Variable_Y", "Pearson_r"]
    long_full.to_csv(root / "feature_correlations_full.csv", index=False)

    rows = []
    for col in df.columns:
        series = df[col]
        inferred = pd.api.types.infer_dtype(series, skipna=True)
        skewness = (
            pd.NA
            if not pd.api.types.is_numeric_dtype(series)
            else round(series.skew(), 4)
        )
        interp = (
            skew_label(float(skewness)) if pd.notna(skewness) else "n/a (non-numeric)"
        )
        rows.append(
            {
                "Feature": col,
                "Inferred_type": inferred,
                "pandas_dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "mean": round(series.mean(), 6)
                if pd.api.types.is_numeric_dtype(series)
                else pd.NA,
                "std": round(series.std(), 6)
                if pd.api.types.is_numeric_dtype(series)
                else pd.NA,
                "skewness": skewness,
                "skew_label": interp,
                "min": series.min()
                if pd.api.types.is_numeric_dtype(series)
                else pd.NA,
                "p25": series.quantile(0.25)
                if pd.api.types.is_numeric_dtype(series)
                else pd.NA,
                "median": series.median()
                if pd.api.types.is_numeric_dtype(series)
                else pd.NA,
                "p75": series.quantile(0.75)
                if pd.api.types.is_numeric_dtype(series)
                else pd.NA,
                "max": series.max()
                if pd.api.types.is_numeric_dtype(series)
                else pd.NA,
            }
        )

    profiles = pd.DataFrame(rows)
    profiles.to_csv(root / "feature_profiles.csv", index=False)


def write_medhouseval_histogram(df: pd.DataFrame, root: Path, n_bins: int = 40) -> None:
    """Pre-binned histogram for target MedHouseVal only (sklearn $100k units); Tableau bar chart."""

    series = df["MedHouseVal"].astype(float).dropna()
    if len(series) < 2:
        return
    binned = pd.cut(series, bins=n_bins, duplicates="drop")
    mids = binned.map(
        lambda iv: float(iv.mid) if isinstance(iv, pd.Interval) else np.nan
    )
    g = pd.DataFrame({"value_bin_mid": mids}).dropna(subset=["value_bin_mid"])
    if g.empty:
        return
    cnt = (
        g.groupby("value_bin_mid", observed=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("value_bin_mid", kind="mergesort")
        .reset_index(drop=True)
    )
    cnt.insert(0, "bin_index", np.arange(len(cnt), dtype=int))
    cnt.to_csv(root / "medhouseval_histogram.csv", index=False)


def write_housing_geo_map(df: pd.DataFrame, root: Path) -> None:
    """Geographic map feeds: percentile-ranked MedHouseVal gives a fuller color ramp when values pile at the census cap."""
    m = df["MedHouseVal"].astype(float)
    pct = m.rank(pct=True, method="average").astype(float)
    pd.DataFrame(
        {
            "Latitude": df["Latitude"].astype(float),
            "Longitude": df["Longitude"].astype(float),
            "MedHouseVal": m,
            "MedHouseVal_pct": pct,
        }
    ).to_csv(root / "housing_geo_map.csv", index=False)


def _describe_cell_str(metric: str, raw: object) -> str:
    """Format one cell for Tableau text tables (everything as string labels)."""
    if raw is None:
        return ""
    if isinstance(raw, float) and np.isnan(raw):
        return ""
    if metric in ("dtype", "skew_label"):
        return str(raw)
    if metric in ("non_null_count", "missing_count", "count"):
        try:
            return str(int(raw))
        except (TypeError, ValueError):
            return str(raw)
    if metric == "skewness":
        return f"{float(raw):.4f}"
    if metric in ("mean", "std", "min", "p25", "median", "p75", "max"):
        s = f"{float(raw):.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(raw)


def write_feature_describe(df: pd.DataFrame, root: Path) -> None:
    """Wide spreadsheet table + melted stats CSV for Tableau Describe worksheet."""
    rows = []
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        rows.append(
            {
                "Feature": col,
                "dtype": str(s.dtype),
                "non_null_count": int(s.notna().sum()),
                "missing_count": int(s.isna().sum()),
                "count": int(s.count()),
                "mean": round(float(s.mean()), 6) if is_num else None,
                "std": round(float(s.std()), 6) if is_num else None,
                "min": float(s.min()) if is_num else None,
                "p25": float(s.quantile(0.25)) if is_num else None,
                "median": float(s.median()) if is_num else None,
                "p75": float(s.quantile(0.75)) if is_num else None,
                "max": float(s.max()) if is_num else None,
                "skewness": round(float(s.skew()), 4) if is_num else None,
                "skew_label": skew_label(float(s.skew())) if is_num else "n/a",
            }
        )
    desc = pd.DataFrame(rows)
    desc.to_csv(root / "feature_describe.csv", index=False)

    stats_keys = [
        "count",
        "mean",
        "std",
        "min",
        "p25",
        "median",
        "p75",
        "max",
        "skewness",
        "skew_label",
    ]

    melted_stats = []
    for _, r in desc.iterrows():
        fname = str(r["Feature"])
        for rk, m in enumerate(stats_keys):
            melted_stats.append(
                {
                    "Feature": fname,
                    "stat_rank": rk,
                    "stat": m,
                    "cell": _describe_cell_str(m, r.get(m)),
                }
            )
    pd.DataFrame(melted_stats).to_csv(root / "feature_cells_stats.csv", index=False)


def write_housing_head(df: pd.DataFrame, root: Path, n: int = 5) -> None:
    """Top n rows with a leading row_id (wide CSV for inspection / spreadsheets)."""
    head = df.head(n).copy()
    head.insert(0, "row_id", range(len(head)))
    head.to_csv(root / "housing_head.csv", index=False)


def write_housing_head_long(df: pd.DataFrame, root: Path, n: int = 5) -> None:
    """Molten head(): row_id × measure × value — Tableau text table without Measure Names pivot."""
    head = df.head(n).copy()
    head.insert(0, "row_id", range(len(head)))
    measure_cols = list(df.columns)
    parts = []
    for _, row in head.iterrows():
        rid = int(row["row_id"])
        for c in measure_cols:
            parts.append({"row_id": rid, "measure": str(c), "value": float(row[c])})
    pd.DataFrame(parts).to_csv(root / "housing_head_long.csv", index=False)


def main() -> None:
    root = Path(__file__).resolve().parent
    out_path = root / "california_housing.csv"

    bunch = fetch_california_housing(as_frame=True)
    df = bunch.frame
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path.name}")

    write_feature_describe(df, root)
    print("Wrote feature_describe.csv, feature_cells_stats.csv")

    write_housing_head(df, root)
    print("Wrote housing_head.csv")

    write_housing_head_long(df, root)
    print("Wrote housing_head_long.csv")

    write_medhouseval_histogram(df, root)
    print("Wrote medhouseval_histogram.csv")

    write_housing_geo_map(df, root)
    print("Wrote housing_geo_map.csv")

    build_ml_assets(df, root)
    print(
        "Wrote feature_correlations_*.csv, feature_profiles.csv, housing_features_long.csv"
    )


if __name__ == "__main__":
    main()
