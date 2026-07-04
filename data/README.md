# data/

Place the two source CSV files here (not committed to git — see `.gitignore`):

| File | Expected name | Source |
|---|---|---|
| Bitcoin Fear & Greed Index | `fear_greed_index.csv` | columns: `timestamp, value, classification, date` |
| Hyperliquid historical trades | `historical_data.csv` | columns include: `Account, Coin, Execution Price, Size Tokens, Size USD, Side, Timestamp IST, Start Position, Direction, Closed PnL, Fee, Trade ID, Timestamp` |

You can override these paths/names without touching code by setting environment
variables before running the CLI:

```bash
export FEAR_GREED_FILE=/path/to/your_fg_file.csv
export TRADES_FILE=/path/to/your_trades_file.csv
```
