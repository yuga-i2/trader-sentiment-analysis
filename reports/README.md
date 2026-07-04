# reports/

Optional: generates a polished `.docx` report (executive summary, charts, tables,
insights) from the chart images produced by the CLI.

**Note:** the narrative text and the summary table in `build_report.js` reflect the
specific dataset used when this project was built. If you run the analysis on
different/updated data, regenerate the numbers (`python -m sentiment_analysis.cli
summary`, `bias`, etc.) and update the corresponding text/table values in
`build_report.js` before rebuilding the report.

## Usage

```bash
# 1. From the repo root, generate the charts the report embeds
python -m sentiment_analysis.cli charts

# 2. Install the one dependency and build the report
cd reports
npm install
npm run build
# -> writes ../outputs/Trader_Sentiment_Analysis_Report.docx
```

Requires Node.js (>=18) and npm.
