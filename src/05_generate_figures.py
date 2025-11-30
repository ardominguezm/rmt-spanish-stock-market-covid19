"""
05_generate_figures.py

Generate all figures used in the RMT Spanish stock market analysis:

- Figure 1: Heatmaps of correlation matrices (pre, during, post, full)
- Figure 2: Eigenvalue distributions vs Marčenko–Pastur
- Figure 3: Eigenvectors of the largest eigenvalue (period comparison)
- Figure 4: Eigenvectors of the second largest eigenvalue (by sector)
- Figure 5: Relationship between eigenportfolio (λ_max) and mean return
- Figure 6: Eigenportfolios for top 4 eigenvalues vs mean return
- Figure 7: Time-varying λ_max using sliding windows

NOTE: This script assumes sector metadata is available in
      data/external/ibex_tickers.csv with a 'sector' column.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    ensure_directories,
    PROCESSED_DIR,
    FIGURES_DIR,
    PERIODS,
    EXTERNAL_DIR,
    marchenko_pastur_pdf,
    sliding_windows,
)


plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = False


def load_sector_metadata():
    path = EXTERNAL_DIR / "ibex_tickers.csv"
    if not path.exists():
        print(f"Warning: {path} not found. Sector information will be unavailable.")
        return None
    meta = pd.read_csv(path)
    if "ticker" not in meta.columns:
        raise ValueError("Metadata file must contain 'ticker' column.")
    meta = meta.set_index("ticker")
    return meta


# -------------------------------------------------------------------
# Figure 1: Heatmaps of correlation matrices
# -------------------------------------------------------------------

def figure_1_heatmaps():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    period_labels = {
        "pre_covid": "Pre-COVID-19",
        "during_covid": "During COVID-19",
        "post_covid": "Post-COVID-19",
        "full": "Full period",
    }

    for ax, (period_name, label) in zip(axes, period_labels.items()):
        corr_path = PROCESSED_DIR / f"corr_{period_name}.parquet"
        corr = pd.read_parquet(corr_path)
        sns.heatmap(
            corr,
            ax=ax,
            cmap="coolwarm",
            center=0,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_title(label)

    plt.tight_layout()
    out = FIGURES_DIR / "figure1_corr_heatmaps.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 1 to {out}")


# -------------------------------------------------------------------
# Figure 2: Eigenvalue distributions vs MP
# -------------------------------------------------------------------

def figure_2_eigenvalue_distributions():
    from utils import marchenko_pastur_bounds

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    period_labels = {
        "pre_covid": "Pre-COVID-19",
        "during_covid": "During COVID-19",
        "post_covid": "Post-COVID-19",
        "full": "Full period",
    }

    summary = pd.read_parquet(PROCESSED_DIR / "rmt_summary.parquet")

    for ax, (period_name, label) in zip(axes, period_labels.items()):
        ev_path = PROCESSED_DIR / f"eigenvalues_{period_name}.parquet"
        eigvals_df = pd.read_parquet(ev_path)
        eigvals = eigvals_df["eigenvalue"].values

        row = summary[summary["period"] == period_name].iloc[0]
        q = float(row["q"])

        # Empirical histogram
        ax.hist(eigvals, bins=20, density=True, alpha=0.6, label="Empirical")

        # MP theoretical density
        lambdas_grid = np.linspace(eigvals.min(), eigvals.max(), 500)
        rho_mp = marchenko_pastur_pdf(lambdas_grid, q)
        ax.plot(lambdas_grid, rho_mp, lw=2, label="Marčenko–Pastur")

        ax.set_title(label)
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    out = FIGURES_DIR / "figure2_eigenvalue_distributions.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 2 to {out}")


# -------------------------------------------------------------------
# Figure 3 & 4: Eigenvectors of λ1 and λ2 (sector-colored)
# -------------------------------------------------------------------

def figure_3_4_eigenvectors():
    meta = load_sector_metadata()

    period_labels = {
        "pre_covid": "Pre-COVID-19",
        "during_covid": "During COVID-19",
        "post_covid": "Post-COVID-19",
    }

    # Figure 3: v1 (largest eigenvalue)
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Figure 4: v2 (second largest eigenvalue)
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for i, (period_name, label) in enumerate(period_labels.items()):
        eigvecs_path = PROCESSED_DIR / f"eigenvectors_{period_name}.parquet"
        eigvecs = pd.read_parquet(eigvecs_path)

        tickers = eigvecs.index
        x = np.arange(len(tickers))

        v1 = eigvecs["v1"].values
        v2 = eigvecs["v2"].values

        # Sector colors (optional)
        if meta is not None and "sector" in meta.columns:
            sectors = meta.reindex(tickers)["sector"]
            # Map sectors to colors
            unique_sectors = sectors.dropna().unique()
            color_map = {
                s: c for s, c in zip(unique_sectors, sns.color_palette("tab20", len(unique_sectors)))
            }
            colors = [color_map.get(s, "gray") for s in sectors]
        else:
            colors = "gray"

        # Figure 3: v1
        ax1 = axes1[i]
        ax1.bar(x, v1, color=colors)
        ax1.set_title(f"{label} – Eigenvector of largest eigenvalue")
        ax1.set_ylabel("Component")
        ax1.set_xticks([])

        # Figure 4: v2
        ax2 = axes2[i]
        ax2.bar(x, v2, color=colors)
        ax2.set_title(f"{label} – Eigenvector of second largest eigenvalue")
        ax2.set_ylabel("Component")
        ax2.set_xticks(range(len(tickers)))
        ax2.set_xticklabels(tickers, rotation=90, fontsize=6)

    plt.tight_layout()
    out1 = FIGURES_DIR / "figure3_eigenvector_v1.png"
    fig1.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved Figure 3 to {out1}")

    out2 = FIGURES_DIR / "figure4_eigenvector_v2.png"
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved Figure 4 to {out2}")


# -------------------------------------------------------------------
# Figures 5 & 6: Eigenportfolios vs mean return
# -------------------------------------------------------------------

def compute_eigenportfolios(
    log_returns: pd.DataFrame,
    eigvecs: pd.DataFrame,
    k: int,
) -> pd.Series:
    """
    Compute eigenportfolio time series for eigenvalue k.

    G_k(t) = (sum_j v_{jk} r_j(t)) / (sum_j v_{jk})

    Parameters
    ----------
    log_returns : DataFrame (T x N)
    eigvecs : DataFrame (N x K)
    k : int
        Eigenvector index (1-based).

    Returns
    -------
    Series of eigenportfolio returns.
    """
    col = f"v{k}"
    w = eigvecs[col]
    # Align columns
    common = log_returns.columns.intersection(eigvecs.index)
    w = w.loc[common]
    R = log_returns[common]
    numer = R.dot(w)
    denom = w.sum()
    return numer / denom


def figure_5_6_eigenportfolios():
    from scipy.stats import linregress

    logret = pd.read_parquet(PROCESSED_DIR / "log_returns.parquet")

    # Use "full" period for Figures 5 & 6 (as in the paper)
    start, end = PERIODS["full"]
    logret_full = logret.loc[start:end]
    mean_ret = logret_full.mean(axis=1)

    eigvecs_path = PROCESSED_DIR / "eigenvectors_full.parquet"
    eigvecs = pd.read_parquet(eigvecs_path)

    # Figure 5: λ_max eigenportfolio vs <r>
    G1 = compute_eigenportfolios(logret_full, eigvecs, k=1)

    slope, intercept, r_value, p_value, std_err = linregress(mean_ret, G1)

    plt.figure(figsize=(6, 4))
    plt.scatter(mean_ret, G1, alpha=0.5, label="Daily returns")
    x_line = np.linspace(mean_ret.min(), mean_ret.max(), 200)
    y_line = intercept + slope * x_line
    plt.plot(x_line, y_line, lw=2, label=f"Fit (R² = {r_value**2:.3f})")
    plt.xlabel("Average market return ⟨r⟩")
    plt.ylabel("Eigenportfolio G_λmax(t)")
    plt.title("Eigenportfolio of largest eigenvalue vs average return")
    plt.legend()
    out5 = FIGURES_DIR / "figure5_eigenportfolio_max_vs_mean.png"
    plt.savefig(out5, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 5 to {out5}")

    # Figure 6: eigenportfolios for first 4 eigenvalues
    plt.figure(figsize=(10, 8))

    for k in range(1, 5):
        Gk = compute_eigenportfolios(logret_full, eigvecs, k=k)
        plt.subplot(2, 2, k)
        plt.scatter(mean_ret, Gk, alpha=0.5)
        slope_k, intercept_k, r_k, *_ = linregress(mean_ret, Gk)
        x_line = np.linspace(mean_ret.min(), mean_ret.max(), 200)
        plt.plot(x_line, intercept_k + slope_k * x_line, lw=1.5)
        plt.xlabel("Average market return ⟨r⟩")
        plt.ylabel(f"G_λ{k}(t)")
        plt.title(f"Eigenportfolio k={k} (R² = {r_k**2:.3f})")

    plt.tight_layout()
    out6 = FIGURES_DIR / "figure6_eigenportfolios_top4.png"
    plt.savefig(out6, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 6 to {out6}")


# -------------------------------------------------------------------
# Figure 7: Sliding window λ_max
# -------------------------------------------------------------------

def figure_7_sliding_lambda_max(window_sizes=(25, 50, 100, 150)):
    logret = pd.read_parquet(PROCESSED_DIR / "log_returns.parquet")

    start, end = PERIODS["full"]
    logret_full = logret.loc[start:end]

    plt.figure(figsize=(10, 8))

    for i, w in enumerate(window_sizes, 1):
        windows = sliding_windows(logret_full, window_size=w, step=1)
        lambda_max_series = []
        dates = []

        for idx, df_win in windows:
            C = df_win.corr()
            eigvals, _ = np.linalg.eigh(C.values)
            lambda_max_series.append(eigvals.max())
            dates.append(idx[-1])

        lambda_max_series = np.array(lambda_max_series)
        dates = np.array(dates)

        plt.subplot(2, 2, i)
        plt.plot(dates, lambda_max_series, lw=1.5)
        plt.title(f"Window size = {w} days")
        plt.xlabel("Date")
        plt.ylabel("λ_max")

    plt.tight_layout()
    out7 = FIGURES_DIR / "figure7_lambda_max_sliding_windows.png"
    plt.savefig(out7, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 7 to {out7}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def generate_all_figures():
    ensure_directories()
    figure_1_heatmaps()
    figure_2_eigenvalue_distributions()
    figure_3_4_eigenvectors()
    figure_5_6_eigenportfolios()
    figure_7_sliding_lambda_max()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all figures for the RMT Spanish market analysis.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all figures (default).",
    )

    args = parser.parse_args()
    generate_all_figures()
