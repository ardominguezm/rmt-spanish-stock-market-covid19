"""
03_compute_correlations.py

Compute correlation matrices for different periods (pre, during, post, full).
Uses log returns stored in data/processed/log_returns.parquet.

Outputs:
- data/processed/corr_pre_covid.parquet
- data/processed/corr_during_covid.parquet
- data/processed/corr_post_covid.parquet
- data/processed/corr_full.parquet
"""

import argparse

import pandas as pd

from utils import (
    ensure_directories,
    PROCESSED_DIR,
    PERIODS,
    slice_period,
)


def compute_correlation_matrices(
    logret_file: str = "log_returns.parquet",
) -> None:
    ensure_directories()
    logret_path = PROCESSED_DIR / logret_file

    print(f"Loading log returns from {logret_path}...")
    log_returns = pd.read_parquet(logret_path)

    for period_name, (start, end) in PERIODS.items():
        print(f"Computing correlation matrix for {period_name} ({start} to {end})")
        sub = slice_period(log_returns, start, end)
        corr = sub.corr()
        out_path = PROCESSED_DIR / f"corr_{period_name}.parquet"
        corr.to_parquet(out_path)
        print(f"Saved {period_name} correlation matrix to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute correlation matrices for all periods.")
    parser.add_argument(
        "--logret_file",
        type=str,
        default="log_returns.parquet",
        help="Log returns file name in data/processed/",
    )

    args = parser.parse_args()
    compute_correlation_matrices(logret_file=args.logret_file)
