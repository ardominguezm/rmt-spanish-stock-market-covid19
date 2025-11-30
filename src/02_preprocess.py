"""
02_preprocess.py

Preprocess raw price data:
- Load prices from data/raw/prices.parquet
- Align and clean series (drop all-NaN tickers, forward/backward fill)
- Compute log returns
- Save processed data to data/processed/log_returns.parquet
"""

import argparse

import pandas as pd

from utils import (
    ensure_directories,
    RAW_DIR,
    PROCESSED_DIR,
    compute_log_returns,
)


def preprocess_prices(
    input_file: str = None,
    output_file: str = None,
) -> None:
    ensure_directories()

    if input_file is None:
        input_path = RAW_DIR / "prices.parquet"
    else:
        input_path = RAW_DIR / input_file

    if output_file is None:
        output_path = PROCESSED_DIR / "log_returns.parquet"
    else:
        output_path = PROCESSED_DIR / output_file

    print(f"Loading prices from {input_path}...")
    prices = pd.read_parquet(input_path)

    # Remove columns that are entirely NaN
    prices = prices.dropna(axis=1, how="all")

    # Fill internal gaps if needed
    prices = prices.sort_index()
    prices = prices.ffill().bfill()

    # Compute log returns
    log_returns = compute_log_returns(prices)

    log_returns.to_parquet(output_path)
    print(f"Saved log returns to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess price data and compute log returns.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input prices file in data/raw/ (default: prices.parquet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output log returns file in data/processed/ (default: log_returns.parquet)",
    )

    args = parser.parse_args()
    preprocess_prices(input_file=args.input, output_file=args.output)
