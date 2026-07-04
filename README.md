# Trader Performance vs. Market Sentiment

Analyzes the relationship between trader performance on Hyperliquid and the
Bitcoin Fear & Greed Index — built as part of the Primetrade.ai AI Internship
assignment.

**Prototype AI features with strict outputs → this project's outputs are
CSV/tabular and chart-based, computed with pandas, and exposed via a small,
logged CLI.**

## What it does

- Merges trade-level history with the daily Fear & Greed Index by date
- Computes performance (PnL, win rate, capital efficiency) broken down by
  sentiment regime (Extreme Fear → Extreme Greed)
- Compares long/short positioning bias and position sizing across regimes
- Ranks accounts by realized PnL and compares "skilled" vs. "average" traders'
  performance within each sentiment regime
- Generates charts and an optional polished Word report

## Key findings (from the reference run)

| Sentiment | Trades | Total PnL | Avg PnL/Trade | Win Rate | Volume |
|---|---|---|---|---|---|
| Extreme Fear | 10,411 | $739K | $70.99 | 76.2% | $56.9M |
| Fear | 29,877 | $3.36M | $112.37 | 87.1% | $239.7M |
| Neutral | 18,216 | $1.29M | $70.98 | 82.1% | $100.9M |
| Greed | 25,355 | $2.15M | $84.80 | 76.3% | $138.7M |
| Extreme Greed | 20,865 | $2.72M | $130.13 | 89.1% | $58.0M |

- **Extreme Greed** had the best risk-adjusted trading (highest win rate and
  capital efficiency) — not Extreme Fear, despite conventional contrarian wisdom.
- **Fear** produced the largest total profit pool and volume, but not the best
  per-trade efficiency — it's where the market is busiest, not necessarily most
  profitable per dollar risked.
- **Skilled traders (top 5 accounts) pull further ahead of the rest of the
  cohort** specifically during Greed/Extreme Greed — average accounts'
  relative edge compresses when the crowd is euphoric.

Full write-up with charts: see [`reports/`](reports/) (or `outputs/Trader_Sentiment_Analysis_Report.docx`
once generated).

## Project structure

```
.
├── data/                       # place source CSVs here (not committed)
├── src/sentiment_analysis/
│   ├── config.py                # paths & shared constants
│   ├── data_loader.py           # load + merge datasets
│   ├── analysis.py              # all metric computations
│   ├── charts.py                # matplotlib chart generation
│   └── cli.py                   # command-line interface
├── reports/
│   └── build_report.js          # optional .docx report generator (Node/docx)
├── tests/
│   └── test_analysis.py         # unit tests on synthetic data
├── outputs/                     # generated charts/CSVs/report land here
├── .github/workflows/ci.yml     # runs pytest on push/PR
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
git clone <your-fork-url>
cd trader-sentiment-analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the two source CSVs in `data/` (see [`data/README.md`](data/README.md)
for expected filenames/columns), or point at existing files via environment
variables:

```bash
export FEAR_GREED_FILE=/path/to/fear_greed_index.csv
export TRADES_FILE=/path/to/historical_data.csv
```

## Usage

```bash
# Performance summary by sentiment regime
python -m sentiment_analysis.cli summary

# Long/short positioning bias by sentiment
python -m sentiment_analysis.cli bias

# Position sizing by sentiment
python -m sentiment_analysis.cli sizing

# Top 10 accounts by realized PnL
python -m sentiment_analysis.cli leaderboard --top 10

# Top 10 traded coins by volume
python -m sentiment_analysis.cli coins --top 10

# Generate all charts into outputs/
python -m sentiment_analysis.cli charts

# Export every summary table as CSV into outputs/
python -m sentiment_analysis.cli export
```

Every run logs to both stdout and `cli.log` (timestamps, command, row counts,
errors) — see `src/sentiment_analysis/cli.py`.

### Optional: build the polished Word report

```bash
python -m sentiment_analysis.cli charts   # make sure charts exist first
cd reports && npm install && npm run build
```

See [`reports/README.md`](reports/README.md) for details.

## Tests

```bash
pytest -v
```

Tests run against small synthetic in-memory data, so they don't require the
real (large, private) CSV files and run in CI on every push/PR
(`.github/workflows/ci.yml`).

## Methodology notes / limitations

- Realized-PnL analysis uses only *closing* events (Close Long, Close Short,
  Sell, and related exits); opening trades carry $0 PnL by construction.
- No account equity/margin data is available, so position size (USD) is used
  as a proxy for risk appetite rather than true leverage.
- The Fear & Greed Index is a market-wide (largely BTC-driven) sentiment
  signal, applied here as a daily macro backdrop rather than an asset-specific
  one — validate per-coin before acting on it.
- Findings are based on ~32 accounts, which limits statistical generalization.

## License

MIT — see [`LICENSE`](LICENSE).
