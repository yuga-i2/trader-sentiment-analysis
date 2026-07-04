# 📊 Trader Behavior vs. Bitcoin Fear & Greed Index
### Hyperliquid On-Chain Data Science Assignment

---

## 🧭 Overview

This project investigates how market sentiment — measured by the **Bitcoin Fear & Greed Index** — influences trader behavior on the **Hyperliquid** decentralised exchange. Using real on-chain trade data, it surfaces hidden patterns in profitability, leverage usage, win rates, and behavioral biases across Fear and Greed market regimes.

---

## 🗂️ Repository Structure

```
trader_analysis/
│
├── data/
│   ├── fear_greed_index.csv        ← Bitcoin Fear & Greed Index (Date, Classification)
│   └── hyperliquid_trades.csv      ← Raw on-chain trade records
│
├── notebooks/
│   └── trader_sentiment_analysis.py   ← Full pipeline (load → clean → model → insights)
│
├── outputs/
│   ├── enriched_trades.csv         ← Merged & feature-engineered trade records
│   ├── trader_profiles.csv         ← Per-account aggregated features + risk scores
│   ├── insights_report.txt         ← Written strategy recommendations
│   ├── fig1_pnl_distribution.png
│   ├── fig2_leverage_sentiment.png
│   ├── fig3_volume_time.png
│   ├── fig4_winrate_heatmap.png
│   ├── fig5_top_traders.png
│   ├── fig6_trader_clusters.png
│   └── fig7_cumulative_pnl.png
│
├── src/
│   └── (modular helper scripts if refactored)
│
├── requirements.txt
└── README.md
```

---

## 📦 Setup

```bash
git clone https://github.com/yourname/trader-sentiment-analysis
cd trader-sentiment-analysis

pip install -r requirements.txt

# Run the full pipeline
python notebooks/trader_sentiment_analysis.py
```

---

## 🔬 Methodology

### 1. Data Understanding
| Dataset | Key Columns | Purpose |
|---|---|---|
| Fear & Greed | `date`, `classification` | Daily market sentiment label |
| Hyperliquid | `account`, `closedPnL`, `leverage`, `side`, `time` | Individual trade records |

### 2. Data Cleaning
- Datetime parsing with ms-epoch fallback
- Column name normalisation across CSV variants
- Missing leverage imputed via per-account median → global median
- Unmatched sentiment dates forward-filled

### 3. Feature Engineering
| Feature | Description |
|---|---|
| `win_rate` | % of closing trades with PnL > 0 |
| `avg_leverage` | Mean leverage per account |
| `pnl_volatility` | Std dev of per-trade PnL |
| `trade_freq` | Trades per active day |
| `risk_score` | Composite 0–100 score (leverage + volatility + frequency) |
| `pnl_fear` / `pnl_greed` | Total PnL split by sentiment regime |

### 4. Statistical Tests
- **Welch's t-test**: PnL and leverage differences between Fear vs Greed
- **KMeans clustering** (k=4) on normalised trader features → 4 archetypes
- **PCA** 2D projection for cluster visualisation

---

## 💡 Key Findings

1. **Sentiment predicts profitability**: Win rates and average PnL differ significantly between Fear and Greed regimes (p < 0.05).
2. **Leverage creep during Greed**: Traders systematically increase leverage during Greed — elevating liquidation risk.
3. **Loss concentration in Fear**: Aggregate daily losses spike during Fear periods, confirming herding-driven liquidation cascades.
4. **4 trader archetypes identified**:
   - 🟢 Consistent Performers
   - 🔴 High-Leverage Gamblers
   - 🔵 Profit Maximisers
   - 🟡 Scalpers

---

## 📈 Actionable Trading Rules

| Rule | Condition | Action |
|---|---|---|
| Leverage Cap | F&G < 30 or > 80 | Max 5× leverage |
| Size Up | F&G 35–55 (Neutral) | +50% position size |
| Long Entry | F&G 20–35 (Fear) | Accumulate longs |
| Profit Take | F&G crosses 75 | Exit 50% of position |
| Frequency | Extreme Fear | Reduce trades by 40% |

---

## 📊 Visualizations

| Figure | Description |
|---|---|
| Fig 1 | PnL distribution histograms — Fear vs Greed |
| Fig 2 | Leverage violin plots by sentiment |
| Fig 3 | Daily trade count & aggregate PnL over time |
| Fig 4 | Win rate heatmap: leverage bucket × sentiment |
| Fig 5 | Top 20 traders by total realised PnL |
| Fig 6 | Trader clusters — PCA projection |
| Fig 7 | Cumulative PnL curves — top 5 traders |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- `pandas`, `numpy` — data wrangling
- `matplotlib`, `seaborn` — visualisation
- `scikit-learn` — clustering, PCA, scaling
- `scipy` — statistical testing

---

## 👤 Author

Data Science Assignment — Quantitative Analysis of Trader Behavior  
Submitted for evaluation by a Web3 trading firm.
