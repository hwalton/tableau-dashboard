"""Fetch California housing and write california_housing.csv."""

from pathlib import Path

from sklearn.datasets import fetch_california_housing

out_path = Path(__file__).resolve().parent / "california_housing.csv"

bunch = fetch_california_housing(as_frame=True)
bunch.frame.to_csv(out_path, index=False)