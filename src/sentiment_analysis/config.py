"""Central configuration: file locations and shared constants.

All paths can be overridden via environment variables, which keeps the
package usable both locally and in CI without editing source code:

    DATA_DIR           default: ./data
    FEAR_GREED_FILE    default: <DATA_DIR>/fear_greed_index.csv
    TRADES_FILE        default: <DATA_DIR>/historical_data.csv
    OUTPUT_DIR         default: ./outputs
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
FEAR_GREED_FILE = Path(os.environ.get("FEAR_GREED_FILE", DATA_DIR / "fear_greed_index.csv"))
TRADES_FILE = Path(os.environ.get("TRADES_FILE", DATA_DIR / "historical_data.csv"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))

# Direction values in the trader dataset that represent a *closing* / realized
# PnL event. Opening trades (Open Long, Open Short) always carry 0 PnL by
# construction, so they are excluded from PnL-based analysis.
CLOSE_DIRECTIONS = [
    "Close Long",
    "Close Short",
    "Sell",
    "Short > Long",
    "Long > Short",
    "Auto-Deleveraging",
    "Liquidated Isolated Short",
    "Settlement",
]

OPEN_DIRECTIONS = ["Open Long", "Open Short"]

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

SENTIMENT_COLORS = {
    "Extreme Fear": "#7f1d1d",
    "Fear": "#dc2626",
    "Neutral": "#9ca3af",
    "Greed": "#16a34a",
    "Extreme Greed": "#14532d",
}
