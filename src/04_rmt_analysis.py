"""
04_rmt_analysis.py

Perform Random Matrix Theory (RMT) analysis on the correlation matrices:
- Compute eigenvalues and eigenvectors
- Compute Marčenko–Pastur bounds
- Save spectra and related information to data/processed/rmt_*.parquet

This script focuses on:
- Static spectra for each period
- Eigenvalues/eigenvectors (sorted descending)
"""

import argparse
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd

from utils import (
    ensure_directories,
    PROCESSED_DIR,
    PERIODS,
    marchenko_pastur_bounds,
)


@dataclass
class RMTResult:
    period: str
    n_assets: int
    t_samples: int
    q: float
    lambda_min_mp: float
    lambda_max_mp: float
    lambda_empirical: np.ndarray


def rmt_for_period(
    period_name: str,
    corr: pd.DataFrame,
    log_returns: pd.DataFrame,
) -> Dict:
    """
    Compute RMT quantities for a single period.

    Returns dict with eigenvalues, eigenvectors, and MP bounds.
    """
    # Only keep columns present in both
    common_cols = corr.columns.intersection(log_returns.columns)
    corr = corr.loc[common_cols, common_cols]

    # Basic dimensions
    n_assets = corr.shape[0]
    t_samples = log_returns.shape[0]
    q = n_assets / t_samples

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(corr.values)
    # Sort descending
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    lam_min, lam_max = marchenko_pastur_bounds(q)

    # Save eigenvalues and eigenvectors
    eigvals_df = pd.DataFrame(
        {"eigenvalue": eigvals},
        index=pd.Index(range(1, len(eigvals) + 1), name="index"),
    )
    eigvecs_df = pd.DataFrame(
        eigvecs,
        index=common_cols,
        columns=[f"v{k}" for k in range(1, len(eigvals) + 1)],
    )

    eigvals_path = PROCESSED_DIR / f"eigenvalues_{period_name}.parquet"
    eigvecs_path = PROCESSED_DIR / f"eigenvectors_{period_name}.parquet"

    eigvals_df.to_parquet(eigvals_path)
    eigvecs_df.to_parquet(eigvecs_path)

    print(f"[{period_name}] Saved eigenvalues to {eigvals_path}")
    print(f"[{period_name}] Saved eigenvectors to {eigvecs_path}")

    result = RMTResult(
        period=period_name,
        n_assets=n_assets,
        t_samples=t_samples,
        q=q,
        lambda_min_mp=lam_min,
        lambda_max_mp=lam_max,
        lambda_empirical=eigvals,
    )
    return asdict(result)


def run_rmt_analysis(
    logret_file: str = "log_returns.parquet",
) -> None:
    ensure_directories()

    logret_path = PROCESSED_DIR / logret_file
    print(f"Loading log returns from {logret_path}...")
    log_returns_full = pd.read_parquet(logret_path)

    rmt_summary = []

    for period_name, (start, end) in PERIODS.items():
        corr_path = PROCESSED_DIR / f"corr_{period_name}.parquet"
        print(f"Loading correlation matrix for {period_name} from {corr_path}...")
        corr = pd.read_parquet(corr_path)

        # Slice log returns to this period
        logret_period = log_returns_full.loc[start:end]

        result_dict = rmt_for_period(period_name, corr, logret_period)
        rmt_summary.append(result_dict)

    summary_df = pd.DataFrame(rmt_summary)
    summary_path = PROCESSED_DIR / "rmt_summary.parquet"
    summary_df.to_parquet(summary_path)
    print(f"Saved RMT summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Matrix Theory analysis of correlation matrices.")
    parser.add_argument(
        "--logret_file",
        type=str,
        default="log_returns.parquet",
        help="Log returns file name in data/processed/",
    )

    args = parser.parse_args()
    run_rmt_analysis(logret_file=args.logret_file)
