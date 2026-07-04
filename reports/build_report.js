const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow,
  TableCell, WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle,
  PageBreak
} = require("docx");

// Charts are expected in ../outputs/ (generated via:
//   python -m sentiment_analysis.cli charts
const A = path.join(__dirname, "..", "outputs") + path.sep;
const NAVY = "1d4ed8";
const GRAY = "6b7280";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 250, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 150 },
    children: [new TextRun({ text, ...opts })],
  });
}
function bullet(text, opts = {}) {
  return new Paragraph({
    bullet: { level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, ...opts })],
  });
}
function img(path, width, height) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new ImageRun({ data: fs.readFileSync(path), transformation: { width, height }, type: "png" })],
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 250 },
    children: [new TextRun({ text, italics: true, size: 18, color: GRAY })],
  });
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "1d4ed8" } : undefined,
    children: [new Paragraph({
      alignment: opts.align || AlignmentType.CENTER,
      children: [new TextRun({ text, bold: opts.header, color: opts.header ? "FFFFFF" : "000000", size: 20 })],
    })],
  });
}

function dataTable(headers, rows, widths) {
  const colWidths = widths || headers.map(() => Math.floor(9000 / headers.length));
  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({ children: headers.map((h, i) => cell(h, { header: true, width: colWidths[i] })) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(String(c), { width: colWidths[i] })) })),
    ],
  });
}

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 } } },
    children: [
      new Paragraph({
        spacing: { after: 60 },
        children: [new TextRun({ text: "Trader Performance vs. Market Sentiment", bold: true, size: 40, color: NAVY })],
      }),
      new Paragraph({
        spacing: { after: 300 },
        children: [new TextRun({ text: "An analysis of Hyperliquid trading activity against the Bitcoin Fear & Greed Index", size: 24, color: GRAY, italics: true })],
      }),

      h1("1. Executive Summary"),
      p("This report analyzes 211,224 trades across 32 accounts on Hyperliquid between May 2023 and May 2025, joined against the daily Bitcoin Fear & Greed Index. The objective was to uncover how trader behavior and profitability shift with market sentiment, and to translate those patterns into actionable trading-strategy guidance."),
      p("Three findings stand out:"),
      bullet("Capital efficiency and win rate are both highest during Extreme Greed (4.68% PnL-per-dollar-traded, 89.1% win rate) \u2014 not Extreme Fear as conventional \u201cbe fearful when others are greedy\u201d wisdom might predict."),
      bullet("Fear regimes generate the largest pool of trading activity and total profit ($3.36M realized PnL, $240M volume), even though per-trade efficiency is more modest \u2014 fear is where the market is busiest, not necessarily where it is most profitable per unit of capital."),
      bullet("Skilled traders (top 5 accounts by PnL) pull further ahead of the rest of the cohort specifically during Greed and Extreme Greed, while the average account's edge compresses \u2014 suggesting euphoric markets reward selective skill and punish the crowd."),

      h1("2. Data & Methodology"),
      p("Two datasets were merged on calendar date:"),
      bullet("Historical Trader Data (Hyperliquid): 211,224 order-level records \u2014 account, coin, execution price, size, side, direction, closed PnL, fees, timestamps."),
      bullet("Bitcoin Fear & Greed Index: daily sentiment score (0-100) and classification (Extreme Fear \u2192 Extreme Greed)."),
      p("100% of trades matched to a sentiment day (May 2023 \u2013 May 2025). Realized-PnL analysis uses only closing events (Close Long, Close Short, Sell, and related exit types) since opening trades carry zero PnL by construction \u2014 104,408 of 211,224 records. Position-sizing and directional-bias analysis uses the 89,636 position-opening events (Open Long / Open Short)."),

      h1("3. Performance by Market Sentiment"),
      img(A + "chart_total_pnl.png", 550, 344),
      caption("Figure 1. Total realized PnL is highest during Fear, driven by trading volume rather than per-trade edge."),
      img(A + "chart_win_rate.png", 550, 344),
      caption("Figure 2. Win rate rises steadily as sentiment moves toward Greed and peaks in Extreme Greed."),
      img(A + "chart_efficiency.png", 550, 344),
      caption("Figure 3. Capital efficiency (realized PnL per dollar traded) is markedly higher in Extreme Greed than any other regime."),

      dataTable(
        ["Sentiment", "Trades", "Total PnL", "Avg PnL/Trade", "Win Rate", "Volume"],
        [
          ["Extreme Fear", "10,411", "$739K", "$70.99", "76.2%", "$56.9M"],
          ["Fear", "29,877", "$3.36M", "$112.37", "87.1%", "$239.7M"],
          ["Neutral", "18,216", "$1.29M", "$70.98", "82.1%", "$100.9M"],
          ["Greed", "25,355", "$2.15M", "$84.80", "76.3%", "$138.7M"],
          ["Extreme Greed", "20,865", "$2.72M", "$130.13", "89.1%", "$58.0M"],
        ],
        [1800, 1200, 1400, 1600, 1200, 1800]
      ),
      p(""),

      h1("4. Positioning Behavior"),
      img(A + "chart_long_short.png", 550, 344),
      caption("Figure 4. Traders flip from a long bias in Fear (68.8% long during Extreme Fear) to a short bias in Greed (57.7% short)."),
      p("This is classic contrarian-leaning behavior \u2014 buying into weakness, fading strength. Yet the sentiment regime where this cohort is most net-short (Greed / Extreme Greed) is also where win rates and efficiency are strongest, implying the short bias is not simply a losing habit here; a meaningful share of these accounts appear to be successfully trading momentum reversals or hedging into rallies rather than blindly fading them."),

      h1("5. Skilled vs. Average Traders"),
      img(A + "chart_cohort.png", 570, 340),
      caption("Figure 5. The top-5 accounts' advantage over the rest of the cohort widens sharply in Greed and Extreme Greed."),
      p("In Neutral markets the gap between the top 5 accounts and everyone else is smallest ($103.69 vs. $34.73 avg PnL/trade). In Extreme Greed the gap is largest by far ($272.81 vs. $77.03). This pattern suggests euphoric markets are where trading skill (entry timing, sizing, exit discipline) is most differentiating \u2014 and where less experienced accounts are most exposed."),

      h1("6. PnL vs. Sentiment Over Time"),
      img(A + "chart_timeseries.png", 620, 285),
      caption("Figure 6. 7-day rolling average of daily realized PnL (blue, left axis) plotted against the Fear & Greed Index (red, right axis)."),

      h1("7. Actionable Insights for Trading Strategy"),
      bullet("Lean in during Extreme Greed, don't just fade it. This dataset's most capital-efficient and highest win-rate trading happens in Extreme Greed. A pure contrarian \u201csell euphoria\u201d rule would have missed the best risk-adjusted regime in the sample."),
      bullet("Treat Fear as a volume opportunity, not necessarily an edge opportunity. Fear produces the most trades and the largest total profit pool, but the lowest capital efficiency alongside Extreme Fear \u2014 sizing and selectivity matter more than raw activity here."),
      bullet("Apply tighter risk controls for average accounts specifically in Greed/Extreme Greed. Since the performance gap between skilled and average traders widens most in these regimes, this is where retail-style accounts are most likely to get out-traded \u2014 a candidate signal for dynamic leverage/position limits."),
      bullet("Watch Neutral regimes for choppiness. Neutral has the lowest avg PnL per trade despite a respectable win rate, consistent with smaller, indecisive price moves \u2014 a case for reduced position sizing or wider profit targets rather than frequent scalping."),
      bullet("Concentration risk: BTC, HYPE, SOL and ETH account for the large majority of volume; sentiment-based strategies should be validated coin-by-coin since the Fear & Greed Index is a BTC/market-wide signal and may not track altcoin-specific sentiment."),

      h1("8. Limitations"),
      bullet("PnL reflects only realized (closed) trades; open positions at the end of the sample are not marked-to-market."),
      bullet("No account equity or margin data was available, so true leverage could not be computed \u2014 position size in USD was used as a proxy for risk appetite."),
      bullet("The Fear & Greed Index is a market-wide (largely BTC-driven) sentiment measure; it is applied here as a daily macro backdrop rather than an asset-specific signal."),
      bullet("Results are drawn from 32 accounts, which limits statistical generalization to the broader trader population."),

      h1("Appendix: Top / Bottom Performing Accounts"),
      dataTable(
        ["Account (truncated)", "Trades", "Total PnL", "Win Rate"],
        [
          ["0xb1231a4a...bfed23", "6,531", "$2.14M", "76.1%"],
          ["0x083384f8...52a9012", "1,732", "$1.60M", "79.3%"],
          ["0xbaaaf657...10637864", "9,997", "$940K", "99.1%"],
          ["0x513b8629...249c4ff1", "5,515", "$840K", "89.0%"],
          ["0xbee1707d...45437aab", "22,556", "$836K", "76.3%"],
        ],
        [3600, 1500, 2000, 1900]
      ),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  const outPath = A + "Trader_Sentiment_Analysis_Report.docx";
  fs.writeFileSync(outPath, buf);
  console.log(`Report written to ${outPath}`);
});
