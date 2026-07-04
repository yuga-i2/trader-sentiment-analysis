#!/usr/bin/env python3
"""
Trader Sentiment Analysis CLI
------------------------------
Explore the relationship between trader performance (Hyperliquid) and
market sentiment (Fear & Greed Index) from the command line.

Examples:
    python -m sentiment_analysis.cli summary
    python -m sentiment_analysis.cli bias
    python -m sentiment_analysis.cli leaderboard --top 10
    python -m sentiment_analysis.cli coins --top 10
    python -m sentiment_analysis.cli charts
    python -m sentiment_analysis.cli export
"""
from __future__ import annotations

import argparse
import logging
import sys

from . import analysis, charts, config
from .data_loader import load_and_merge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("cli.log"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sentiment-analysis-cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trader vs. market sentiment analysis CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("summary", help="Performance summary by sentiment regime")
    sub.add_parser("bias", help="Long/short positioning by sentiment regime")
    sub.add_parser("sizing", help="Position sizing by sentiment regime")

    lb = sub.add_parser("leaderboard", help="Top accounts by realized PnL")
    lb.add_argument("--top", type=int, default=10)

    co = sub.add_parser("coins", help="Top traded coins by volume")
    co.add_argument("--top", type=int, default=10)

    sub.add_parser("charts", help="Generate all charts into the outputs/ directory")
    sub.add_parser("export", help="Export all summary tables as CSVs into outputs/")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    log.info("Running command: %s", args.command)
    try:
        df = load_and_merge()
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    try:
        if args.command == "summary":
            print(analysis.sentiment_summary(df).to_string(index=False))
        elif args.command == "bias":
            print(analysis.long_short_bias(df).to_string())
        elif args.command == "sizing":
            print(analysis.position_size_by_sentiment(df).to_string(index=False))
        elif args.command == "leaderboard":
            print(analysis.account_leaderboard(df, top=args.top).to_string())
        elif args.command == "coins":
            print(analysis.coin_volume(df, top=args.top).to_string())
        elif args.command == "charts":
            paths = charts.generate_all(df)
            for p in paths:
                print(f"wrote {p}")
        elif args.command == "export":
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            analysis.sentiment_summary(df).to_csv(config.OUTPUT_DIR / "sentiment_summary.csv", index=False)
            analysis.long_short_bias(df).to_csv(config.OUTPUT_DIR / "long_short_bias.csv")
            analysis.position_size_by_sentiment(df).to_csv(config.OUTPUT_DIR / "size_by_sentiment.csv", index=False)
            analysis.account_leaderboard(df).to_csv(config.OUTPUT_DIR / "account_leaderboard.csv")
            analysis.cohort_comparison(df).to_csv(config.OUTPUT_DIR / "cohort_by_sentiment.csv")
            analysis.coin_volume(df).to_csv(config.OUTPUT_DIR / "coin_volume.csv")
            analysis.daily_timeseries(df).to_csv(config.OUTPUT_DIR / "daily_timeseries.csv", index=False)
            print(f"Exported summary CSVs to {config.OUTPUT_DIR}/")
    except Exception as e:
        log.error("Command '%s' failed: %s", args.command, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
