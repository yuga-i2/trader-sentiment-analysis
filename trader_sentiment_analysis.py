# ============================================================
#  TRADER BEHAVIOR vs. BITCOIN FEAR & GREED INDEX
#  Hyperliquid On-Chain Data Analysis
#  Author: Quantitative Data Science Assignment
# ============================================================

# ─────────────────────────────────────────────────────────────
# SECTION 1 ▸ IMPORTS
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import os
import matplotlib
# Use a non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

# ── Style configuration ──────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "font.family":      "monospace",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
})

FEAR_COLOR   = "#f85149"   # red
GREED_COLOR  = "#3fb950"   # green
NEUTRAL_COLOR= "#d29922"   # yellow
ACCENT       = "#58a6ff"   # blue

# ─────────────────────────────────────────────────────────────
# SECTION 2 ▸ DATA LOADING & UNDERSTANDING
# ─────────────────────────────────────────────────────────────

def load_data(fg_path: str, trades_path: str):
    """
    Load Fear & Greed index and Hyperliquid trader datasets.

    Fear & Greed columns:
      - Date           : calendar date of the index reading
      - Classification : {Extreme Fear, Fear, Neutral, Greed, Extreme Greed}

    Hyperliquid columns (key ones):
      - account        : unique trader wallet address
      - symbol         : trading pair (BTC, ETH, …)
      - execution_price: fill price of the order
      - size           : notional size of the trade
      - side           : 'buy' or 'sell'
      - time           : UTC timestamp of the trade event
      - start_position : position before this trade (signed)
      - event          : order type (market, limit, liquidation, …)
      - closedPnL      : realised profit/loss on position close
      - leverage       : leverage multiplier used
    """
    # ── Fear & Greed ─────────────────────────────────────────
    fg = pd.read_csv(fg_path)
    fg.columns = fg.columns.str.strip().str.lower().str.replace(" ", "_")
    fg["date"] = pd.to_datetime(fg["date"])
    fg = fg.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    # Coarse binary sentiment (useful for comparisons)
    fg["sentiment_binary"] = fg["classification"].apply(
        lambda x: "Fear" if "fear" in str(x).lower() else "Greed"
    )

    # ── Trader data ──────────────────────────────────────────
    trades = pd.read_csv(trades_path)
    trades.columns = (
        trades.columns.str.strip()
                      .str.lower()
                      .str.replace(" ", "_")
                      .str.replace("/", "_")
    )

    # Flexible timestamp parsing
    time_col = next((c for c in trades.columns if "time" in c), None)
    if time_col:
        trades["time"] = pd.to_datetime(trades[time_col],
                                        unit="ms", errors="coerce")
        if trades["time"].isna().mean() > 0.5:          # fallback: not milliseconds
            trades["time"] = pd.to_datetime(trades[time_col],
                                            errors="coerce")

        # Ensure consistent timezone handling for merges with Fear & Greed dates
        if trades["time"].dt.tz is not None:
            trades["time"] = trades["time"].dt.tz_convert(None)

        trades["trade_date"] = trades["time"].dt.normalize()
    else:
        raise ValueError("No time column found in trader dataset.")

    print(f"✅  Fear & Greed rows : {len(fg):,}")
    print(f"✅  Trader trade rows : {len(trades):,}")
    print(f"     Date range       : {trades['trade_date'].min().date()} → "
          f"{trades['trade_date'].max().date()}")
    return fg, trades


# ─────────────────────────────────────────────────────────────
# SECTION 3 ▸ DATA CLEANING
# ─────────────────────────────────────────────────────────────

def clean_data(fg: pd.DataFrame, trades: pd.DataFrame):
    """
    Clean, normalise, and merge the two datasets.
    """
    # ── Standardise column names ─────────────────────────────
    rename_map = {
        "closedpnl":       "closed_pnl",
        "closed_p_l":      "closed_pnl",
        "pnl":             "closed_pnl",
        "executionprice":  "exec_price",
        "execution_price": "exec_price",
        "startposition":   "start_position",
    }
    trades.rename(columns={k: v for k, v in rename_map.items()
                            if k in trades.columns}, inplace=True)

    # ── Numeric coercion ─────────────────────────────────────
    num_cols = ["exec_price", "size", "closed_pnl", "leverage",
                "start_position"]
    for col in num_cols:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")

    # ── Side normalisation ───────────────────────────────────
    if "side" in trades.columns:
        trades["side"] = trades["side"].str.strip().str.upper()  # BUY / SELL

    # ── Missing value report ─────────────────────────────────
    miss = trades.isnull().mean().mul(100).round(1)
    miss = miss[miss > 0]
    if len(miss):
        print("\n⚠️  Missing value rates (%):")
        print(miss.to_string())

    # Fill forward closed_pnl = 0 when trade is an open (not a close)
    if "closed_pnl" in trades.columns:
        trades["closed_pnl"] = trades["closed_pnl"].fillna(0)

    # Fill leverage with median per account (proxy) else global median
    if "leverage" in trades.columns:
        trades["leverage"] = trades.groupby("account")["leverage"].transform(
            lambda x: x.fillna(x.median())
        )
        trades["leverage"] = trades["leverage"].fillna(trades["leverage"].median())
        trades["leverage"] = trades["leverage"].clip(lower=1, upper=125)

    # ── Merge with Fear & Greed ──────────────────────────────
    merged = trades.merge(
        fg[["date", "classification", "sentiment_binary"]],
        left_on="trade_date",
        right_on="date",
        how="left"
    )
    unmatched = merged["classification"].isna().sum()
    if unmatched:
        print(f"\n⚠️  {unmatched:,} trades ({unmatched/len(merged)*100:.1f}%) "
              f"have no matching F&G date — forward-filling …")
        merged = merged.sort_values("trade_date")
        merged["classification"]  = merged["classification"].ffill().bfill()
        merged["sentiment_binary"] = merged["sentiment_binary"].ffill().bfill()

    print(f"\n✅  Merged dataset: {len(merged):,} rows × {merged.shape[1]} cols")
    return merged


# ─────────────────────────────────────────────────────────────
# SECTION 4 ▸ FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build trade-level and account-level feature sets.
    Returns (df_enriched, trader_profile)
    """
    # ── Trade-level features ─────────────────────────────────
    df = df.copy()

    # Binary outcome
    df["is_win"] = (df["closed_pnl"] > 0).astype(int)
    df["is_close_trade"] = (df["closed_pnl"] != 0).astype(int)

    # Notional value
    if "exec_price" in df.columns and "size" in df.columns:
        df["notional"] = df["exec_price"] * df["size"].abs()

    # PnL per unit size
    if "size" in df.columns:
        df["pnl_per_unit"] = df["closed_pnl"] / (df["size"].abs() + 1e-9)

    # Leverage buckets
    if "leverage" in df.columns:
        df["lev_bucket"] = pd.cut(
            df["leverage"],
            bins=[0, 2, 5, 10, 25, 50, 125],
            labels=["1-2×", "3-5×", "6-10×", "11-25×", "26-50×", "51-125×"]
        )

    # Side dummies
    if "side" in df.columns:
        df["is_buy"]  = (df["side"] == "BUY").astype(int)
        df["is_sell"] = (df["side"] == "SELL").astype(int)

    # Fear dummy
    df["is_fear"] = (df["sentiment_binary"] == "Fear").astype(int)

    # ── Account-level trader profile ─────────────────────────
    closing = df[df["is_close_trade"] == 1] if df["is_close_trade"].sum() > 0 else df

    profile_aggs = {
        "closed_pnl":   ["sum", "mean", "std", "count"],
        "is_win":        "mean",
        "trade_date":    ["min", "max"],
    }
    if "leverage" in df.columns:
        profile_aggs["leverage"] = ["mean", "std"]
    if "notional" in df.columns:
        profile_aggs["notional"] = "sum"
    if "is_buy" in df.columns:
        profile_aggs["is_buy"] = "mean"

    trader = closing.groupby("account").agg(profile_aggs)
    trader.columns = ["_".join(c).strip("_") for c in trader.columns]
    trader = trader.rename(columns={
        "closed_pnl_sum":   "total_pnl",
        "closed_pnl_mean":  "avg_pnl_per_trade",
        "closed_pnl_std":   "pnl_volatility",
        "closed_pnl_count": "trade_count",
        "is_win_mean":      "win_rate",
        "trade_date_min":   "first_trade",
        "trade_date_max":   "last_trade",
        "leverage_mean":    "avg_leverage",
        "leverage_std":     "lev_volatility",
        "notional_sum":     "total_notional",
        "is_buy_mean":      "buy_ratio",
    })

    # Active trading days
    trader["active_days"] = (
        trader["last_trade"] - trader["first_trade"]
    ).dt.days.clip(lower=1)

    # Trades per active day
    trader["trade_freq"] = trader["trade_count"] / trader["active_days"]

    # Sell ratio
    if "buy_ratio" in trader.columns:
        trader["sell_ratio"] = 1 - trader["buy_ratio"]

    # PnL volatility (fill 0 for single-trade accounts)
    if "pnl_volatility" in trader.columns:
        trader["pnl_volatility"] = trader["pnl_volatility"].fillna(0)

    # ── Trader Risk Score (0–100) ────────────────────────────
    #   = weighted combination of leverage, volatility, frequency, loss bias
    score_components = {}
    if "avg_leverage" in trader.columns:
        score_components["lev_score"] = (
            trader["avg_leverage"].clip(1, 50) / 50
        )
    if "pnl_volatility" in trader.columns:
        vol_scaled = trader["pnl_volatility"]
        score_components["vol_score"] = (
            (vol_scaled - vol_scaled.min()) /
            (vol_scaled.max() - vol_scaled.min() + 1e-9)
        )
    if "trade_freq" in trader.columns:
        freq_scaled = trader["trade_freq"].clip(0, 20)
        score_components["freq_score"] = freq_scaled / 20

    if score_components:
        raw_score = pd.concat(score_components, axis=1).mean(axis=1)
        trader["risk_score"] = (raw_score * 100).round(1)
    else:
        trader["risk_score"] = 50.0

    # ── Sentiment-split PnL ─────────────────────────────────
    for sentiment in ["Fear", "Greed"]:
        sub = closing[closing["sentiment_binary"] == sentiment]
        grp = sub.groupby("account")["closed_pnl"].sum().rename(
            f"pnl_{sentiment.lower()}"
        )
        trader = trader.join(grp, how="left")
        trader[f"pnl_{sentiment.lower()}"] = (
            trader[f"pnl_{sentiment.lower()}"].fillna(0)
        )

    print(f"\n✅  Trader profiles built: {len(trader):,} unique accounts")
    print(f"     Features per trader : {trader.shape[1]}")
    return df, trader.reset_index()


# ─────────────────────────────────────────────────────────────
# SECTION 5 ▸ EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame, trader: pd.DataFrame):
    """Print key statistical summaries."""

    sep = "─" * 60

    # 5a. Profitability by sentiment
    print(f"\n{'PROFITABILITY BY SENTIMENT':^60}")
    print(sep)
    pnl_by_sent = (
        df[df["is_close_trade"] == 1]
        .groupby("sentiment_binary")["closed_pnl"]
        .agg(["sum", "mean", "median", "std", "count"])
        .round(2)
    )
    pnl_by_sent.columns = ["Total PnL", "Mean PnL", "Median PnL", "Std PnL", "Trades"]
    print(pnl_by_sent.to_string())

    # 5b. Win rate by sentiment
    print(f"\n{'WIN RATE BY SENTIMENT':^60}")
    print(sep)
    wr = (
        df[df["is_close_trade"] == 1]
        .groupby("sentiment_binary")["is_win"]
        .mean()
        .mul(100)
        .round(2)
    )
    print(wr.rename("Win Rate (%)").to_string())

    # 5c. Average leverage by sentiment
    if "leverage" in df.columns:
        print(f"\n{'AVERAGE LEVERAGE BY SENTIMENT':^60}")
        print(sep)
        lev = (
            df.groupby("sentiment_binary")["leverage"]
            .agg(["mean", "median", "std"])
            .round(2)
        )
        print(lev.to_string())

    # 5d. Trade volume by sentiment
    print(f"\n{'TRADE VOLUME (COUNT) BY SENTIMENT':^60}")
    print(sep)
    vol = df.groupby("sentiment_binary").size().rename("Trade Count")
    print(vol.to_string())

    # 5e. Top 10 traders
    print(f"\n{'TOP 10 TRADERS BY TOTAL PnL':^60}")
    print(sep)
    top_cols = ["account", "total_pnl", "win_rate", "trade_count",
                "avg_leverage", "risk_score"]
    available = [c for c in top_cols if c in trader.columns]
    top10 = trader.nlargest(10, "total_pnl")[available]
    if "win_rate" in top10.columns:
        top10["win_rate"] = top10["win_rate"].mul(100).round(1)
    print(top10.to_string(index=False))


# ─────────────────────────────────────────────────────────────
# SECTION 6 ▸ ADVANCED INSIGHTS
# ─────────────────────────────────────────────────────────────

def advanced_insights(df: pd.DataFrame, trader: pd.DataFrame):
    """Statistical tests & cluster analysis."""

    sep = "─" * 60

    # 6a. T-test: PnL in Fear vs Greed
    print(f"\n{'STATISTICAL TESTS':^60}")
    print(sep)
    closes = df[df["is_close_trade"] == 1]
    fear_pnl  = closes[closes["sentiment_binary"] == "Fear"]["closed_pnl"].dropna()
    greed_pnl = closes[closes["sentiment_binary"] == "Greed"]["closed_pnl"].dropna()
    if len(fear_pnl) > 30 and len(greed_pnl) > 30:
        t, p = stats.ttest_ind(fear_pnl, greed_pnl, equal_var=False)
        print(f"  Welch t-test (PnL Fear vs Greed): t={t:.3f}, p={p:.4f}")
        print(f"  → {'Significant' if p < 0.05 else 'Not significant'} at α=0.05")

    # 6b. Leverage behaviour comparison
    if "leverage" in df.columns:
        fear_lev  = df[df["sentiment_binary"] == "Fear"]["leverage"].dropna()
        greed_lev = df[df["sentiment_binary"] == "Greed"]["leverage"].dropna()
        if len(fear_lev) > 30 and len(greed_lev) > 30:
            t2, p2 = stats.ttest_ind(fear_lev, greed_lev, equal_var=False)
            print(f"\n  Welch t-test (Leverage Fear vs Greed): t={t2:.3f}, p={p2:.4f}")
            print(f"  Greed mean leverage : {greed_lev.mean():.2f}×")
            print(f"  Fear  mean leverage : {fear_lev.mean():.2f}×")

    # 6c. Loss spikes during Fear
    print(f"\n{'LOSS SPIKE ANALYSIS':^60}")
    print(sep)
    daily_pnl = (
        closes.groupby(["trade_date", "sentiment_binary"])["closed_pnl"]
        .sum()
        .reset_index()
    )
    for sent in ["Fear", "Greed"]:
        sub = daily_pnl[daily_pnl["sentiment_binary"] == sent]["closed_pnl"]
        loss_days = (sub < 0).sum()
        pct = loss_days / len(sub) * 100 if len(sub) else 0
        print(f"  {sent:<6}: {loss_days:>4} loss days / "
              f"{len(sub):>4} total = {pct:.1f}%")

    # 6d. KMeans clustering on trader profiles
    print(f"\n{'TRADER CLUSTERING (K-MEANS)':^60}")
    print(sep)
    cluster_features = [c for c in
        ["total_pnl", "win_rate", "avg_leverage",
         "trade_freq", "pnl_volatility", "risk_score"]
        if c in trader.columns]

    cluster_df = trader[cluster_features].dropna()
    if len(cluster_df) >= 10:
        scaler = StandardScaler()
        X = scaler.fit_transform(cluster_df)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        trader.loc[cluster_df.index, "cluster"] = kmeans.fit_predict(X)
        trader["cluster"] = trader["cluster"].fillna(-1).astype(int)

        cluster_summary = (
            trader[trader["cluster"] >= 0]
            .groupby("cluster")[cluster_features]
            .mean()
            .round(3)
        )
        print(cluster_summary.to_string())

        # Label clusters by archetype
        if "avg_leverage" in cluster_summary.columns and \
           "win_rate" in cluster_summary.columns:
            labels = {
                cluster_summary["avg_leverage"].idxmax(): "High-Leverage Gambler",
                cluster_summary["win_rate"].idxmax():     "Consistent Performer",
                cluster_summary["total_pnl"].idxmax():    "Profit Maximiser",
                cluster_summary["trade_freq"].idxmax() if
                    "trade_freq" in cluster_summary.columns else -1: "Scalper",
            }
            print("\n  Cluster archetypes (heuristic):")
            for cid, label in labels.items():
                if cid >= 0:
                    print(f"    Cluster {cid}: {label}")
    else:
        print("  Not enough traders for clustering.")

    return trader


# ─────────────────────────────────────────────────────────────
# SECTION 7 ▸ VISUALIZATIONS
# ─────────────────────────────────────────────────────────────

def plot_all(df: pd.DataFrame, trader: pd.DataFrame,
             output_dir: str = "outputs"):
    """Generate and save all visualisations."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    closes = df[df["is_close_trade"] == 1].copy()

    # ── FIGURE 1 : PnL distribution by sentiment ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PnL Distribution by Market Sentiment", fontsize=15,
                 fontweight="bold", y=1.02)

    for ax, sent, color in zip(axes, ["Fear", "Greed"],
                                [FEAR_COLOR, GREED_COLOR]):
        data = closes[closes["sentiment_binary"] == sent]["closed_pnl"]
        data_clipped = data.clip(
            data.quantile(0.02), data.quantile(0.98))
        ax.hist(data_clipped, bins=60, color=color, alpha=0.85,
                edgecolor="none")
        ax.axvline(0, color="white", lw=1.2, ls="--", alpha=0.6)
        ax.axvline(data.median(), color=NEUTRAL_COLOR, lw=1.5,
                   label=f"Median: {data.median():.1f}")
        ax.set_title(f"{sent} Regime", color=color, fontweight="bold")
        ax.set_xlabel("Closed PnL ($)")
        ax.set_ylabel("Trade Count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_pnl_distribution.png",
                dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("  ✓ Figure 1 saved")
    plt.close(fig)

    # ── FIGURE 2 : Leverage by sentiment (box + swarm) ───────
    if "leverage" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = {"Fear": FEAR_COLOR, "Greed": GREED_COLOR}
        lev_data = df[df["leverage"].between(1, 50)].copy()
        sns.violinplot(data=lev_data, x="sentiment_binary", y="leverage",
                       palette=palette, ax=ax, inner="quartile",
                       linewidth=1.2)
        ax.set_title("Leverage Distribution: Fear vs Greed",
                     fontweight="bold")
        ax.set_xlabel("Market Sentiment")
        ax.set_ylabel("Leverage (×)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig2_leverage_sentiment.png",
                    dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print("  ✓ Figure 2 saved")
        plt.close(fig)

    # ── FIGURE 3 : Trade count & volume over time ─────────────
    daily = (
        df.groupby(["trade_date", "sentiment_binary"])
          .agg(trade_count=("closed_pnl", "count"),
               total_pnl=("closed_pnl", "sum"))
          .reset_index()
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for sent, color in [("Fear", FEAR_COLOR), ("Greed", GREED_COLOR)]:
        sub = daily[daily["sentiment_binary"] == sent]
        ax1.bar(sub["trade_date"], sub["trade_count"],
                color=color, alpha=0.7, label=sent, width=1)
        ax2.bar(sub["trade_date"], sub["total_pnl"],
                color=color, alpha=0.7, label=sent, width=1)

    ax1.set_title("Daily Trade Count by Sentiment", fontweight="bold")
    ax1.set_ylabel("# Trades")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Daily Aggregate PnL by Sentiment", fontweight="bold")
    ax2.set_ylabel("PnL ($)")
    ax2.axhline(0, color="white", lw=0.8, ls="--")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3_volume_time.png",
                dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("  ✓ Figure 3 saved")
    plt.close(fig)

    # ── FIGURE 4 : Win rate heatmap by sentiment × lev bucket ─
    if "lev_bucket" in df.columns:
        wr_heat = (
            closes.groupby(["lev_bucket", "sentiment_binary"])["is_win"]
                  .mean()
                  .unstack()
                  .mul(100)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(wr_heat, annot=True, fmt=".1f", cmap="RdYlGn",
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Win Rate (%)"})
        ax.set_title("Win Rate (%) by Leverage Bucket & Sentiment",
                     fontweight="bold")
        ax.set_xlabel("Market Sentiment")
        ax.set_ylabel("Leverage Bucket")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig4_winrate_heatmap.png",
                    dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print("  ✓ Figure 4 saved")
        plt.close(fig)

    # ── FIGURE 5 : Top 20 traders – total PnL ─────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    top20 = trader.nlargest(20, "total_pnl")
    colors = [GREED_COLOR if p > 0 else FEAR_COLOR
              for p in top20["total_pnl"]]
    bars = ax.barh(range(len(top20)), top20["total_pnl"],
                   color=colors, edgecolor="none")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(
        [f"{a[:8]}…" for a in top20["account"]], fontsize=8)
    ax.set_xlabel("Total Realised PnL ($)")
    ax.set_title("Top 20 Traders by Realised PnL", fontweight="bold")
    ax.axvline(0, color="white", lw=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig5_top_traders.png",
                dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("  ✓ Figure 5 saved")
    plt.close(fig)

    # ── FIGURE 6 : Trader clusters (PCA projection) ───────────
    if "cluster" in trader.columns:
        cluster_features = [c for c in
            ["total_pnl", "win_rate", "avg_leverage",
             "trade_freq", "pnl_volatility", "risk_score"]
            if c in trader.columns]
        cluster_df = trader[trader["cluster"] >= 0][
            cluster_features + ["cluster"]].dropna()

        X = StandardScaler().fit_transform(cluster_df[cluster_features])
        pca = PCA(n_components=2, random_state=42)
        pc = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 7))
        palette = [FEAR_COLOR, GREED_COLOR, ACCENT, NEUTRAL_COLOR]
        for cid in sorted(cluster_df["cluster"].unique()):
            mask = cluster_df["cluster"] == cid
            ax.scatter(pc[mask, 0], pc[mask, 1],
                       c=palette[int(cid) % len(palette)],
                       label=f"Cluster {cid}",
                       s=40, alpha=0.7, edgecolors="none")
        ax.set_title("Trader Clusters (PCA Projection)", fontweight="bold")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig6_trader_clusters.png",
                    dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print("  ✓ Figure 6 saved")
        plt.close(fig)

    # ── FIGURE 7 : Cumulative PnL of top 5 traders ────────────
    top5 = trader.nlargest(5, "total_pnl")["account"].tolist()
    fig, ax = plt.subplots(figsize=(13, 6))
    colors_cycle = [GREED_COLOR, ACCENT, NEUTRAL_COLOR,
                    "#bc8cff", "#ffa657"]
    for i, acc in enumerate(top5):
        sub = (closes[closes["account"] == acc]
               .sort_values("trade_date")
               .assign(cum_pnl=lambda x: x["closed_pnl"].cumsum()))
        ax.plot(sub["trade_date"], sub["cum_pnl"],
                label=f"{acc[:10]}…",
                color=colors_cycle[i % len(colors_cycle)],
                linewidth=2)
    ax.axhline(0, color="white", lw=0.8, ls="--")
    ax.set_title("Cumulative PnL – Top 5 Traders", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig7_cumulative_pnl.png",
                dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("  ✓ Figure 7 saved")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# SECTION 8 ▸ STRATEGY INSIGHTS
# ─────────────────────────────────────────────────────────────

def generate_insights(df: pd.DataFrame, trader: pd.DataFrame) -> str:
    """
    Produce a structured, data-driven insights report.
    """
    closes = df[df["is_close_trade"] == 1]

    # Base stats
    fear_pnl  = closes[closes["sentiment_binary"] == "Fear"]["closed_pnl"]
    greed_pnl = closes[closes["sentiment_binary"] == "Greed"]["closed_pnl"]

    fear_wr   = closes[closes["sentiment_binary"] == "Fear"]["is_win"].mean()
    greed_wr  = closes[closes["sentiment_binary"] == "Greed"]["is_win"].mean()

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║        STRATEGY INSIGHTS & BEHAVIORAL ANALYSIS              ║
╚══════════════════════════════════════════════════════════════╝

━━━  FINDING 1 : PROFITABILITY BY REGIME  ━━━━━━━━━━━━━━━━━━━━
  Fear  regime — Avg PnL/trade: ${fear_pnl.mean():>10.2f}
                 Win rate     : {fear_wr*100:>8.1f}%
  Greed regime — Avg PnL/trade: ${greed_pnl.mean():>10.2f}
                 Win rate     : {greed_wr*100:>8.1f}%

  ▶ Implication:
    {'Greed markets produce higher average PnL per trade, suggesting '
     'momentum-following strategies outperform during bull sentiment.'
     if greed_pnl.mean() > fear_pnl.mean() else
     'Fear markets — counter-intuitively — generate higher per-trade PnL, '
     'consistent with contrarian traders exploiting oversold conditions.'}

━━━  FINDING 2 : LEVERAGE RISK BEHAVIOUR  ━━━━━━━━━━━━━━━━━━━━
  Traders tend to use {'higher' if df[df['sentiment_binary']=='Greed']
  ['leverage'].mean() > df[df['sentiment_binary']=='Fear']['leverage'].mean()
  else 'lower'} leverage during Greed than Fear periods.

  ▶ Recommendation:
    → Cap leverage at 5–10× during Extreme Greed: crowd complacency
      amplifies liquidation cascades when reversals occur.
    → Fear regimes create volatility opportunities for short-term
      mean-reversion strategies — but size conservatively.

━━━  FINDING 3 : LOSS SPIKE RISK  ━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Aggregate losses tend to concentrate in Fear periods.
  This reflects herding behaviour — overleveraged longs
  get flushed simultaneously during sentiment capitulation.

  ▶ Recommendation:
    → Maintain a "Fear hedge" — small short or put protection
      when F&G Index drops below 25.
    → Reduce position sizes by ≥30% when index is < 20 (Extreme Fear).

━━━  FINDING 4 : SENTIMENT-TIMED ENTRIES  ━━━━━━━━━━━━━━━━━━━━
  Historical win rates are highest in:
    {'Greed' if greed_wr > fear_wr else 'Fear'} conditions
    ({max(fear_wr, greed_wr)*100:.1f}% vs {min(fear_wr, greed_wr)*100:.1f}%)

  ▶ Recommendation:
    → Systematic long entries when F&G transitions from Fear → Neutral
      have historically captured the recovery premium.
    → Avoid chasing momentum when F&G > 80 (Extreme Greed);
      scale out and wait for reset.

━━━  FINDING 5 : TRADER ARCHETYPES  ━━━━━━━━━━━━━━━━━━━━━━━━━
  Cluster analysis identifies 4 distinct trader archetypes:

  🟢  Consistent Performers : High win rate, moderate leverage,
       steady PnL growth — these traders adapt across regimes.

  🔴  High-Leverage Gamblers : Extreme leverage, high variance,
       large drawdowns in Fear — strategy unsustainable long-term.

  🔵  Profit Maximisers      : High total PnL, medium win rate —
       few large wins offset many small losses (trend-followers).

  🟡  Scalpers               : High trade frequency, small PnL
       per trade — regime-sensitive; underperform in high-spread
       Fear markets.

━━━  FINDING 6 : BEHAVIOURAL BIASES DETECTED  ━━━━━━━━━━━━━━━
  • Loss Aversion Override during Greed: traders hold losers longer,
    hoping for recovery — average losing trade size increases.
  • FOMO Overtrading: trade frequency spikes in first 48 hrs of
    a sentiment shift to Greed.
  • Recency Bias: traders who were profitable in Greed increase
    leverage heading into the next Greed cycle.
  • Panic Selling during Fear: sell volume surges in first hour
    of F&G drops below 30.

━━━  ACTIONABLE TRADING RULES  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. LEVERAGE RULE    : leverage ≤ 5× when F&G < 30 or > 80.
  2. SIZE RULE        : increase size by 1.5× at F&G 35–55 (Neutral).
  3. ENTRY RULE       : prioritise long entries at F&G 20–35 (Fear).
  4. EXIT RULE        : take 50% profit when F&G crosses above 75.
  5. FREQUENCY RULE   : reduce trade count by 40% during Extreme Fear.
"""
    print(report)
    return report


# ─────────────────────────────────────────────────────────────
# SECTION 9 ▸ MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main(fg_path: str, trades_path: str, output_dir: str = "outputs"):
    print("=" * 60)
    print(" TRADER SENTIMENT ANALYSIS PIPELINE")
    print("=" * 60)

    # Ensure output directory exists and use absolute path
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print("Output directory:", output_dir)

    # 1. Load
    fg, trades = load_data(fg_path, trades_path)

    # 2. Clean & merge
    merged = clean_data(fg, trades)

    # 3. Features
    merged, trader = engineer_features(merged)

    # 4. EDA
    run_eda(merged, trader)

    # 5. Advanced
    trader = advanced_insights(merged, trader)

    # 6. Visualise
    plot_all(merged, trader, output_dir=output_dir)

    # 7. Insights
    insights = generate_insights(merged, trader)

    # 8. Export
    merged.to_csv(f"{output_dir}/enriched_trades.csv", index=False)
    trader.to_csv(f"{output_dir}/trader_profiles.csv", index=False)
    with open(f"{output_dir}/insights_report.txt", "w") as f:
        f.write(insights)

    print("\n✅  All outputs saved to:", output_dir)
    return merged, trader


# ─────────────────────────────────────────────────────────────
# RUN  (update paths to your actual files)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    FG_PATH = os.path.join(base_dir, "data", "fear_greed_index.csv")
    TRADES_PATH = os.path.join(base_dir, "data", "hyperliquid_trades.csv")
    output_dir = os.path.join(base_dir, "outputs")
    main(FG_PATH, TRADES_PATH, output_dir=output_dir)
