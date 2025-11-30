"""
01_download_data.py

Download daily adjusted close prices for IBEX35 + IBEX Medium Cap stocks
from Yahoo Finance using the tickers specified in data/external/ibex_tickers.csv.

The resulting price panel is stored in data/raw/prices.parquet
"""

import argparse
from pathlib import Path

import yfinance as yf
import pandas as pd

from utils import (
    ensure_directories,
    RAW_DIR,
    load_tickers_from_csv,
)


def download_prices(
    start: str = "2018-01-01",
    end: str = "2023-03-31",
    output_file: Path = None,
) -> None:
    """
    Download adjusted close prices for all tickers and save as a parquet file.

    Parameters
    ----------
    start : str
        Start date (YYYY-MM-DD) for download.
    end : str
        End date (YYYY-MM-DD) for download.
    output_file : Path
        Where to store the resulting prices DataFrame.
    """
    ensure_directories()
    if output_file is None:
        output_file = RAW_DIR / "prices.parquet"

    tickers_df = load_tickers_from_csv()
    tickers = tickers_df["ticker"].tolist()

    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=True,
        group_by="ticker",
        threads=True,
    )

    # Build a wide DataFrame: index = Date, columns = tickers, values = Adj Close
    prices = pd.DataFrame(index=data.index)
    for ticker in tickers:
        try:
            prices[ticker] = data[ticker]["Adj Close"]
        except KeyError:
            print(f"Warning: Ticker {ticker} not found in yfinance download.")

    prices = prices.sort_index()
    prices.to_parquet(output_file)
    print(f"Saved prices to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download IBEX prices from Yahoo Finance.")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-03-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (parquet). Default: data/raw/prices.parquet",
    )

    args = parser.parse_args()
    out_path = Path(args.output) if args.output is not None else None

    download_prices(start=args.start, end=args.end, output_file=out_path)
