"""Loading, cleaning, and merging the two source datasets."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)


def load_fear_greed(path: Path = config.FEAR_GREED_FILE) -> pd.DataFrame:
    """Load the daily Fear & Greed Index CSV.

    Expected columns: timestamp, value, classification, date
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Fear & Greed file not found at {path}. "
            f"Place it in the data/ directory or set FEAR_GREED_FILE."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "value", "classification"]].rename(
        columns={"value": "fg_value", "classification": "sentiment"}
    )
    log.info("Loaded %d sentiment days from %s", len(df), path)
    return df


def load_trades(path: Path = config.TRADES_FILE) -> pd.DataFrame:
    """Load the raw Hyperliquid trade-level history CSV.

    Expected columns include: Account, Coin, Execution Price, Size Tokens,
    Size USD, Side, Timestamp IST, Start Position, Direction, Closed PnL,
    Fee, Trade ID, Timestamp.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Trades file not found at {path}. "
            f"Place it in the data/ directory or set TRADES_FILE."
        )
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce")
    df["date"] = df["dt"].dt.normalize()
    df["is_close"] = df["Direction"].isin(config.CLOSE_DIRECTIONS)
    df["is_win"] = df["Closed PnL"] > 0
    log.info("Loaded %d trades from %s", len(df), path)
    return df


def merge_datasets(trades: pd.DataFrame, fear_greed: pd.DataFrame) -> pd.DataFrame:
    """Inner-join trades to the sentiment reading for their calendar date."""
    merged = trades.merge(fear_greed, on="date", how="inner")
    merged["sentiment"] = pd.Categorical(
        merged["sentiment"], categories=config.SENTIMENT_ORDER, ordered=True
    )
    match_rate = len(merged) / len(trades) if len(trades) else 0
    log.info(
        "Matched %d/%d trades to a sentiment day (%.1f%%)",
        len(merged), len(trades), match_rate * 100,
    )
    return merged


def load_and_merge(
    fg_path: Path = config.FEAR_GREED_FILE, trades_path: Path = config.TRADES_FILE
) -> pd.DataFrame:
    """Convenience wrapper: load both files and return the merged dataset."""
    fg = load_fear_greed(fg_path)
    trades = load_trades(trades_path)
    return merge_datasets(trades, fg)
