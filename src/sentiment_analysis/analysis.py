"""Core analytical computations over the merged trades + sentiment dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def closing_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows that represent a realized-PnL closing event."""
    return df[df["is_close"]].copy()


def opening_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows that represent a new position being opened."""
    return df[df["Direction"].isin(config.OPEN_DIRECTIONS)].copy()


def sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate realized trading performance by sentiment regime."""
    closes = closing_trades(df)
    out = closes.groupby("sentiment", observed=True).agg(
        trades=("Closed PnL", "count"),
        total_pnl=("Closed PnL", "sum"),
        avg_pnl=("Closed PnL", "mean"),
        median_pnl=("Closed PnL", "median"),
        win_rate=("is_win", "mean"),
        total_volume_usd=("Size USD", "sum"),
        avg_trade_size_usd=("Size USD", "mean"),
        total_fees=("Fee", "sum"),
    ).reset_index()
    out["pnl_per_usd_traded"] = out["total_pnl"] / out["total_volume_usd"]
    return out


def long_short_bias(df: pd.DataFrame) -> pd.DataFrame:
    """% of newly opened positions that are Long vs. Short, by sentiment."""
    opens = opening_trades(df)
    opens["side_bucket"] = opens["Direction"].map(
        lambda d: "Long" if "Long" in str(d) else ("Short" if "Short" in str(d) else "Other")
    )
    bias = opens.groupby(["sentiment", "side_bucket"], observed=True).size().unstack(fill_value=0)
    bias_pct = bias.div(bias.sum(axis=1), axis=0) * 100
    return bias_pct.round(1)


def position_size_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Average / median new-position size (USD), by sentiment."""
    opens = opening_trades(df)
    return (
        opens.groupby("sentiment", observed=True)["Size USD"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )


def account_leaderboard(df: pd.DataFrame, top: int | None = None) -> pd.DataFrame:
    """Rank accounts by total realized PnL."""
    closes = closing_trades(df)
    lb = closes.groupby("Account").agg(
        trades=("Closed PnL", "count"),
        total_pnl=("Closed PnL", "sum"),
        win_rate=("is_win", "mean"),
    ).sort_values("total_pnl", ascending=False)
    return lb.head(top) if top else lb


def cohort_comparison(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Compare avg PnL/trade of the top-N accounts vs. everyone else, by sentiment."""
    closes = closing_trades(df)
    leaderboard = account_leaderboard(df)
    top_accounts = leaderboard.head(top_n).index
    closes["cohort"] = np.where(closes["Account"].isin(top_accounts), "Top N", "Rest")
    return closes.groupby(["cohort", "sentiment"], observed=True)["Closed PnL"].mean().unstack()


def coin_volume(df: pd.DataFrame, top: int | None = None) -> pd.Series:
    """Total traded volume (USD) by coin, on closing trades."""
    closes = closing_trades(df)
    vol = closes.groupby("Coin")["Size USD"].sum().sort_values(ascending=False)
    return vol.head(top) if top else vol


def daily_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Daily realized PnL / trade count / volume, joined with the sentiment score."""
    closes = closing_trades(df)
    daily = closes.groupby("date", observed=True).agg(
        daily_pnl=("Closed PnL", "sum"),
        daily_trades=("Closed PnL", "count"),
        daily_volume=("Size USD", "sum"),
    ).reset_index()
    fg = df[["date", "fg_value"]].drop_duplicates("date")
    return daily.merge(fg, on="date", how="left").sort_values("date")
