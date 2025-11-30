"""
utils.py

Utility functions and global configuration for the RMT analysis of the
Spanish stock market around COVID-19.

This module centralizes paths, date ranges, and common helper functions.
"""

import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Paths & basic configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Date ranges as in the paper (YYYY-MM-DD)
PERIODS: Dict[str, Tuple[str, str]] = {
    "pre_covid": ("2018-01-02", "2020-01-31"),
    "during_covid": ("2020-02-03", "2021-01-02"),
    "post_covid": ("2021-03-03", "2023-03-17"),
    "full": ("2018-01-02", "2023-03-17"),
}


def ensure_directories() -> None:
    """Create all required directories if they do not exist."""
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR,
              RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Ticker handling
# -------------------------------------------------------------------

def load_tickers_from_csv(csv_path: Path = None) -> pd.DataFrame:
    """
    Load the list of tickers from a CSV file.

    Expected columns: 'ticker', optional: 'name', 'sector', 'index'.

    If csv_path is None, defaults to data/external/ibex_tickers.csv.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with at least a 'ticker' column.
    """
    if csv_path is None:
        csv_path = EXTERNAL_DIR / "ibex_tickers.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Ticker file not found at {csv_path}. "
            f"Please create a CSV with at least a 'ticker' column."
        )

    df = pd.read_csv(csv_path)
    if "ticker" not in df.columns:
        raise ValueError("CSV must contain a 'ticker' column.")
    return df


# -------------------------------------------------------------------
# Financial helpers
# -------------------------------------------------------------------

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Price time series, indexed by date, columns = tickers.

    Returns
    -------
    log_returns : pd.DataFrame
        Log returns aligned with prices (one row less).
    """
    return np.log(prices / prices.shift(1)).dropna(how="all")


def slice_period(
    df: pd.DataFrame,
    start: str,
    end: str,
    closed: str = "both"
) -> pd.DataFrame:
    """
    Slice a time series DataFrame by date (index must be DatetimeIndex).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with DatetimeIndex.
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    closed : {'both', 'left', 'right'}
        Inclusion of endpoints (as in pandas IntervalIndex).

    Returns
    -------
    df_slice : pd.DataFrame
    """
    return df.loc[start:end]


# -------------------------------------------------------------------
# Random Matrix Theory helpers
# -------------------------------------------------------------------

def marchenko_pastur_bounds(q: float, sigma2: float = 1.0) -> Tuple[float, float]:
    """
    Compute the theoretical Marčenko–Pastur bounds λ_min, λ_max.

    Parameters
    ----------
    q : float
        Aspect ratio q = N / T, where N = number of assets, T = length
        of the time series.
    sigma2 : float, default 1.0
        Variance of the underlying variables (usually = 1 for correlation).

    Returns
    -------
    (lambda_min, lambda_max) : tuple of float
    """
    if q <= 0:
        raise ValueError("q must be positive.")
    lambda_min = sigma2 * (1 - np.sqrt(1 / q)) ** 2
    lambda_max = sigma2 * (1 + np.sqrt(1 / q)) ** 2
    return lambda_min, lambda_max


def marchenko_pastur_pdf(
    lambdas: np.ndarray,
    q: float,
    sigma2: float = 1.0
) -> np.ndarray:
    """
    Compute the Marčenko–Pastur probability density on given eigenvalues.

    Parameters
    ----------
    lambdas : np.ndarray
        Array of λ values where the density is evaluated.
    q : float
        Aspect ratio N / T.
    sigma2 : float
        Variance (1 for correlation matrices).

    Returns
    -------
    rho : np.ndarray
        MP density evaluated at 'lambdas'. Values outside [λ_min, λ_max]
        are set to 0.
    """
    lam_min, lam_max = marchenko_pastur_bounds(q, sigma2)
    rho = np.zeros_like(lambdas, dtype=float)
    mask = (lambdas >= lam_min) & (lambdas <= lam_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho[mask] = (
            np.sqrt((lam_max - lambdas[mask]) * (lambdas[mask] - lam_min))
            / (2 * np.pi * sigma2 * lambdas[mask] * (1 / q))
        )
    rho[~np.isfinite(rho)] = 0.0
    return rho


# -------------------------------------------------------------------
# Sliding window helper
# -------------------------------------------------------------------

def sliding_windows(
    df: pd.DataFrame,
    window_size: int,
    step: int = 1
) -> List[Tuple[pd.DatetimeIndex, pd.DataFrame]]:
    """
    Generate sliding windows over the rows of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame.
    window_size : int
        Number of rows per window.
    step : int, default 1
        Step size between windows.

    Returns
    -------
    windows : list of (index, df_window)
    """
    idx = df.index
    windows = []
    for start in range(0, len(df) - window_size + 1, step):
        end = start + window_size
        window_df = df.iloc[start:end]
        windows.append((window_df.index, window_df))
    return windows
