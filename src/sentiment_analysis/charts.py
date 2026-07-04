"""Matplotlib chart generation for the analysis outputs."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .analysis import (
    cohort_comparison,
    daily_timeseries,
    long_short_bias,
    sentiment_summary,
)

log = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def _colors(index):
    return [config.SENTIMENT_COLORS[x] for x in index]


def plot_total_pnl(df: pd.DataFrame, out_dir: Path) -> Path:
    s = sentiment_summary(df).set_index("sentiment").reindex(config.SENTIMENT_ORDER).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(s["sentiment"], s["total_pnl"] / 1e6, color=_colors(s["sentiment"]))
    ax.set_ylabel("Total Realized PnL ($M)")
    ax.set_title("Total Trader PnL by Market Sentiment Regime")
    for b, v in zip(bars, s["total_pnl"] / 1e6):
        ax.text(b.get_x() + b.get_width() / 2, v, f"${v:.2f}M", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = out_dir / "chart_total_pnl.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_win_rate(df: pd.DataFrame, out_dir: Path) -> Path:
    s = sentiment_summary(df).set_index("sentiment").reindex(config.SENTIMENT_ORDER).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(s["sentiment"], s["win_rate"] * 100, color=_colors(s["sentiment"]))
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Trade Win Rate by Market Sentiment Regime")
    ax.set_ylim(0, 100)
    for b, v in zip(bars, s["win_rate"] * 100):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = out_dir / "chart_win_rate.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_efficiency(df: pd.DataFrame, out_dir: Path) -> Path:
    s = sentiment_summary(df).set_index("sentiment").reindex(config.SENTIMENT_ORDER).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(s["sentiment"], s["pnl_per_usd_traded"] * 100, color=_colors(s["sentiment"]))
    ax.set_ylabel("Realized PnL per $ Traded (%)")
    ax.set_title("Capital Efficiency by Market Sentiment Regime")
    for b, v in zip(bars, s["pnl_per_usd_traded"] * 100):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = out_dir / "chart_efficiency.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_long_short(df: pd.DataFrame, out_dir: Path) -> Path:
    bias = long_short_bias(df).reindex(config.SENTIMENT_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bias.index, bias["Long"], label="Long", color="#16a34a")
    ax.bar(bias.index, bias["Short"], bottom=bias["Long"], label="Short", color="#dc2626")
    ax.set_ylabel("% of New Positions Opened")
    ax.set_title("Long vs. Short Positioning by Market Sentiment")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "chart_long_short.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_cohort(df: pd.DataFrame, out_dir: Path, top_n: int = 5) -> Path:
    cohort = cohort_comparison(df, top_n=top_n)[config.SENTIMENT_ORDER]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(config.SENTIMENT_ORDER))
    w = 0.35
    ax.bar([i - w / 2 for i in x], cohort.loc["Top N"], width=w, label=f"Top {top_n} Accounts", color="#1d4ed8")
    ax.bar([i + w / 2 for i in x], cohort.loc["Rest"], width=w, label="Rest of Accounts", color="#93c5fd")
    ax.set_xticks(list(x))
    ax.set_xticklabels(config.SENTIMENT_ORDER)
    ax.set_ylabel("Avg Realized PnL per Trade ($)")
    ax.set_title(f"Top-{top_n} Accounts vs. Rest: Avg PnL per Trade by Sentiment")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "chart_cohort.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_timeseries(df: pd.DataFrame, out_dir: Path) -> Path:
    d = daily_timeseries(df)
    d["pnl_7d"] = d["daily_pnl"].rolling(7, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax1.plot(d["date"], d["pnl_7d"], color="#1d4ed8", linewidth=1.6, label="7-day avg daily PnL ($)")
    ax1.set_ylabel("7-day Avg Daily Realized PnL ($)", color="#1d4ed8")
    ax1.tick_params(axis="y", labelcolor="#1d4ed8")
    ax1.axhline(0, color="gray", linewidth=0.6)

    ax2 = ax1.twinx()
    ax2.plot(d["date"], d["fg_value"], color="#dc2626", linewidth=1, alpha=0.5, label="Fear/Greed Index")
    ax2.set_ylabel("Fear/Greed Index (0-100)", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.set_title("Trader PnL vs. Market Sentiment Index Over Time")
    fig.tight_layout()
    path = out_dir / "chart_timeseries.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def generate_all(df: pd.DataFrame, out_dir: Path = config.OUTPUT_DIR) -> list[Path]:
    """Generate every chart and return the list of file paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_total_pnl(df, out_dir),
        plot_win_rate(df, out_dir),
        plot_efficiency(df, out_dir),
        plot_long_short(df, out_dir),
        plot_cohort(df, out_dir),
        plot_timeseries(df, out_dir),
    ]
    log.info("Wrote %d charts to %s", len(paths), out_dir)
    return paths
