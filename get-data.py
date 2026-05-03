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
    """Correlation matrix, profiles, melted features, histogram bins."""
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

    meta = profiles[
        ["Feature", "pandas_dtype", "Inferred_type", "skew_label", "skewness"]
    ].rename(columns={"Inferred_type": "inferred_type"})
    melted_enriched = melted.merge(meta, on="Feature", how="left")

    parts = []
    for _fname, grp in melted_enriched.groupby("Feature"):
        bins = pd.cut(grp["Value"].astype(float), bins=40, duplicates="drop")
        grp = grp.copy()
        grp["Value_bin_mid"] = bins.map(
            lambda iv: float(iv.mid) if isinstance(iv, pd.Interval) else np.nan
        )
        parts.append(grp)
    melted_enriched = pd.concat(parts, ignore_index=True)
    melted_enriched.to_csv(root / "housing_features_long_enriched.csv", index=False)

    hist_counts = (
        melted_enriched.dropna(subset=["Value_bin_mid"])
        .groupby(["Feature", "Value_bin_mid"], observed=False)
        .size()
        .rename("BIN_COUNT")
        .reset_index()
        .astype({"Feature": str})
    )
    hist_counts = hist_counts.merge(meta, on="Feature", how="left")
    hist_counts.to_csv(root / "feature_histogram_bins.csv", index=False)


def main() -> None:
    root = Path(__file__).resolve().parent
    out_path = root / "california_housing.csv"

    bunch = fetch_california_housing(as_frame=True)
    df = bunch.frame
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path.name}")

    build_ml_assets(df, root)
    print(
        "Wrote feature_correlations_*.csv, feature_profiles.csv, "
        "housing_features_long*.csv, feature_histogram_bins.csv"
    )


if __name__ == "__main__":
    main()
