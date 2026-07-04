"""Unit tests for sentiment_analysis.analysis, using small synthetic frames
so tests run without the real (large) CSV data files."""
import pandas as pd
import pytest

from sentiment_analysis import analysis


@pytest.fixture
def sample_df() -> pd.DataFrame:
    data = [
        # Account, Coin, Direction, Closed PnL, Size USD, Fee, date, sentiment
        ("A1", "BTC", "Open Long", 0, 1000, 0.1, "2024-01-01", "Fear"),
        ("A1", "BTC", "Close Long", 100, 1000, 0.1, "2024-01-02", "Fear"),
        ("A1", "BTC", "Open Short", 0, 500, 0.05, "2024-02-01", "Greed"),
        ("A1", "BTC", "Close Short", -50, 500, 0.05, "2024-02-02", "Greed"),
        ("A2", "ETH", "Open Long", 0, 2000, 0.2, "2024-01-01", "Fear"),
        ("A2", "ETH", "Close Long", 200, 2000, 0.2, "2024-01-02", "Fear"),
    ]
    df = pd.DataFrame(
        data,
        columns=["Account", "Coin", "Direction", "Closed PnL", "Size USD", "Fee", "date", "sentiment"],
    )
    df["date"] = pd.to_datetime(df["date"])
    df["is_close"] = df["Direction"].isin(analysis.config.CLOSE_DIRECTIONS)
    df["is_win"] = df["Closed PnL"] > 0
    df["sentiment"] = pd.Categorical(
        df["sentiment"], categories=analysis.config.SENTIMENT_ORDER, ordered=True
    )
    df["fg_value"] = df["sentiment"].map({"Fear": 25, "Greed": 75})
    return df


def test_closing_trades_excludes_opens(sample_df):
    closes = analysis.closing_trades(sample_df)
    assert set(closes["Direction"]) == {"Close Long", "Close Short"}
    assert len(closes) == 3


def test_sentiment_summary_totals(sample_df):
    summary = analysis.sentiment_summary(sample_df)
    fear_row = summary[summary["sentiment"] == "Fear"].iloc[0]
    assert fear_row["trades"] == 2
    assert fear_row["total_pnl"] == 300
    assert fear_row["win_rate"] == 1.0

    greed_row = summary[summary["sentiment"] == "Greed"].iloc[0]
    assert greed_row["trades"] == 1
    assert greed_row["total_pnl"] == -50
    assert greed_row["win_rate"] == 0.0


def test_long_short_bias(sample_df):
    bias = analysis.long_short_bias(sample_df)
    # Fear regime: 2 opens, both Long -> 100% long
    assert bias.loc["Fear", "Long"] == 100.0
    # Greed regime: 1 open, Short -> 100% short
    assert bias.loc["Greed", "Short"] == 100.0


def test_account_leaderboard_ranks_by_pnl(sample_df):
    lb = analysis.account_leaderboard(sample_df)
    assert lb.index[0] == "A2"  # A2 has the highest total PnL (200)
    assert lb.loc["A2", "total_pnl"] == 200


def test_coin_volume(sample_df):
    vol = analysis.coin_volume(sample_df)
    assert vol["BTC"] == 1500  # two BTC closes: 1000 + 500
    assert vol["ETH"] == 2000
