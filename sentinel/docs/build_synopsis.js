/* Builds the SENTINEL-TS project synopsis — a focused overview covering the
   idea, dataset, problem statement, objectives, and an explicit mapping of
   every model to the specific case it covers. */
const fs = require("fs");
const d = require("docx");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageBreak, LevelFormat, Footer, PageNumber,
} = d;

const ACCENT = "1F6F78";
const DARK = "123B42";
const GREY = "595959";
const W = 9360;

const P = (text, opts = {}) =>
  new Paragraph({
    spacing: { after: opts.after ?? 120, line: 276 },
    alignment: opts.align,
    children: [new TextRun({ text, size: opts.size ?? 21, color: opts.color, bold: opts.bold, italics: opts.italics })],
  });

const RP = (parts, opts = {}) =>
  new Paragraph({
    spacing: { after: opts.after ?? 120, line: 276 },
    children: parts.map((p) =>
      Array.isArray(p)
        ? new TextRun(Object.assign({ text: p[0], size: 21 }, p[1]))
        : new TextRun({ text: p, size: 21 })
    ),
  });

const H1 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 340, after: 150 },
    children: [new TextRun({ text, bold: true, size: 29, color: DARK })],
  });

const H2 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 110 },
    children: [new TextRun({ text, bold: true, size: 23, color: ACCENT })],
  });

const BULLET = (text) =>
  new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 70, line: 276 },
    children: [new TextRun({ text, size: 21 })],
  });

const NUM = (text, instance = 0) =>
  new Paragraph({
    numbering: { reference: "numbers", level: 0, instance },
    spacing: { after: 70, line: 276 },
    children: [new TextRun({ text, size: 21 })],
  });

const CAPTION = (text) =>
  new Paragraph({
    spacing: { after: 200, before: 60 },
    children: [new TextRun({ text, size: 17, italics: true, color: GREY })],
  });

function TABLE(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cols = widths.map((w) => Math.round((w / total) * W));
  const prose = opts.prose;

  const cell = (text, i, { head = false, bold = false } = {}) =>
    new TableCell({
      width: { size: cols[i], type: WidthType.DXA },
      shading: head
        ? { type: ShadingType.CLEAR, fill: ACCENT }
        : bold
        ? { type: ShadingType.CLEAR, fill: "E8F1F2" }
        : undefined,
      margins: { top: 60, bottom: 60, left: 90, right: 90 },
      children: [
        new Paragraph({
          spacing: { after: 0, line: 240 },
          alignment: prose || i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
          children: [
            new TextRun({
              text: String(text),
              size: head ? 17 : 18,
              bold: head || bold,
              color: head ? "FFFFFF" : undefined,
            }),
          ],
        }),
      ],
    });

  return new Table({
    columnWidths: cols,
    width: { size: W, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color: "BFD3D6" },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: "BFD3D6" },
      left: { style: BorderStyle.SINGLE, size: 2, color: "BFD3D6" },
      right: { style: BorderStyle.SINGLE, size: 2, color: "BFD3D6" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: "DCE6E8" },
      insideVertical: { style: BorderStyle.SINGLE, size: 1, color: "DCE6E8" },
    },
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) => cell(h, i, { head: true })),
      }),
      ...rows.map(
        (r) =>
          new TableRow({
            children: r.map((c, i) =>
              cell(c, i, { bold: opts.boldRows && opts.boldRows.includes(r[0]) })
            ),
          })
      ),
    ],
  });
}

const children = [];

/* ------------------------------------------------------------- header */
children.push(
  new Paragraph({
    spacing: { after: 40 },
    children: [new TextRun({ text: "SENTINEL-TS", bold: true, size: 44, color: DARK })],
  }),
  new Paragraph({
    spacing: { after: 30 },
    children: [new TextRun({ text: "Hybrid Time-Series Anomaly Intelligence for Cloud Infrastructure Telemetry", size: 24, color: ACCENT })],
  }),
  new Paragraph({
    spacing: { after: 300 },
    children: [new TextRun({ text: "Project Synopsis  ·  Time Series Data Analysis (TSDA) Mini Project  ·  Dept. of Computer Engineering, SPIT", size: 18, color: GREY })],
  })
);

/* =========================== 1. PROJECT IDENTITY ======================== */
children.push(
  H1("1. Project at a Glance"),
  TABLE(
    ["Field", "Detail"],
    [
      ["Project name", "SENTINEL-TS"],
      ["Full title", "Hybrid Time-Series Anomaly Intelligence for Cloud Infrastructure Telemetry"],
      ["Domain", "Site Reliability Engineering / IT infrastructure monitoring"],
      ["Core task", "Forecast expected server behaviour, then detect and classify deviations from it"],
      ["Dataset", "Numenta Anomaly Benchmark (NAB) — realAWSCloudwatch family"],
      ["Data nature", "Genuine Amazon CloudWatch production server telemetry"],
      ["Scale", "6 services · 25,477 observations · 5-minute sampling · ~14 days each"],
      ["Ground truth", "11 human-curated anomaly windows"],
      ["Models compared", "7 (2 baselines, 2 classical, 1 machine learning, 2 hybrid)"],
      ["Deliverable", "Working application: Python analysis engine + REST API + browser console"],
    ],
    [22, 78],
    { prose: true }
  ),
  CAPTION("Table 1 — Project identity.")
);

/* =========================== 2. DATASET ================================= */
children.push(
  H1("2. Dataset"),
  H2("2.1 Source"),
  RP([
    "The dataset is the ",
    ["realAWSCloudwatch", { bold: true }],
    " family of the ",
    ["Numenta Anomaly Benchmark (NAB)", { bold: true }],
    ", published openly at github.com/numenta/NAB under AGPL-3.0. These are genuine Amazon CloudWatch measurements taken from production Amazon Web Services infrastructure — they are not simulated traces.",
  ]),
  P(
    "Real telemetry was chosen over a synthetic generator deliberately. Synthetic data contains exactly the structure its author put there, so a model that assumes that structure will always appear to succeed. Real server data contains flat idle stretches, reporting gaps, mixed noise regimes and anomalies of several different shapes, and it punishes assumptions that do not hold."
  ),

  H2("2.2 Structure and scale"),
  TABLE(
    ["Attribute", "Value"],
    [
      ["Columns", "2 — timestamp (ISO 8601 datetime, UTC), value (float)"],
      ["Native sampling frequency", "5 minutes (288 observations per day)"],
      ["Duration per series", "approximately 14 days"],
      ["Observations per series", "4,032 (4,730 for the two longer series)"],
      ["Total observations", "25,477 across 6 series"],
      ["Analysis frequency", "resampled to 15 minutes (96 buckets/day); user-selectable 5 min to 1 hour"],
      ["Labels", "anomaly windows (start, end) — not point labels"],
    ],
    [30, 70],
    { prose: true }
  ),
  CAPTION("Table 2 — Dataset structure."),

  H2("2.3 The six services"),
  TABLE(
    ["Series file", "Metric measured", "Unit", "Windows"],
    [
      ["ec2_cpu_utilization_5f5533", "EC2 instance CPU utilisation", "%", "2"],
      ["rds_cpu_utilization_cc0c53", "RDS database CPU utilisation", "%", "2"],
      ["ec2_network_in_257a54", "EC2 inbound network volume", "bytes", "1"],
      ["elb_request_count_8c0756", "Load balancer request count", "req/5 min", "2"],
      ["ec2_disk_write_bytes_1ef3de", "EC2 disk write throughput", "bytes", "1"],
      ["grok_asg_anomaly", "Auto-scaling group aggregate load", "units", "3"],
    ],
    [33, 36, 16, 15],
    {}
  ),
  CAPTION("Table 3 — The six telemetry series. Each is modelled independently."),
  P(
    "A note on usable ground truth: 11 windows exist, but at a 30% chronological hold-out only 5 of them fall inside a test set. On two services the single labelled window lies entirely within the training portion, so detection cannot be scored there at all. The evaluation therefore rests on 5 windows across 4 services, and this constrains how much confidence any ranking deserves."
  ),
  new Paragraph({ children: [new PageBreak()] })
);

/* =========================== 3. PROBLEM ================================= */
children.push(
  H1("3. Problem Statement"),

  H2("3.1 Statement"),
  new Paragraph({
    spacing: { after: 180, line: 300 },
    shading: { type: ShadingType.CLEAR, fill: "F0F6F7" },
    border: {
      left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT },
    },
    indent: { left: 180, right: 180 },
    children: [
      new TextRun({
        text: "Cloud services emit continuous telemetry, and incidents appear in that stream before customers notice them. Static thresholds fail to detect those incidents because server load is strongly periodic: the decision rule is stationary while the process it monitors is not. The problem is to build a system that learns what each metric should be at each moment in time, flags departures from that expectation, and converts those flags into a small number of classified, actionable incidents that an on-call engineer can act on without first investigating the alert itself.",
        size: 21,
        italics: true,
      }),
    ],
  }),

  H2("3.2 What exactly we are solving"),
  P("The failure mode is specific, well known to anyone who has carried a pager, and expensive:"),
  BULLET("Infrastructure load is strongly diurnal and weekly. A production API may run at 70% CPU during business hours and 20% overnight."),
  BULLET("A threshold set at 85% to avoid overnight noise cannot see a service that silently drops from 20% to 2% at 03:00 because an upstream dependency died — a severe incident with no threshold crossing at all."),
  BULLET("A threshold set tight enough to catch that would fire every single morning as load legitimately ramps up."),
  BULLET("Teams resolve this by widening thresholds until paging stops. The alert is then still configured, still shows green on a dashboard, and detects nothing. Its effective recall is zero while appearing to provide coverage."),
  RP([
    "So the problem is ",
    ["not insufficient data and not insufficient alerting", { bold: true }],
    ". It is that a fixed threshold cannot represent a time-varying normal. Our solution replaces the fixed threshold with a forecast: model the expected value at each timestamp, and alert on the residual between observed and expected. The decision boundary then moves with time of day automatically.",
  ]),

  H2("3.3 What we are trying to do — objectives"),
  NUM("Build a forecasting model of normal behaviour that captures daily periodicity, slow level drift and short-memory autocorrelation, so the expected value is time-dependent rather than constant.", 1),
  NUM("Detect anomalies as departures from that expectation, with the threshold calibrated on training data alone so the detector cannot adapt to the incidents it is meant to find.", 1),
  NUM("Design and evaluate a hybrid model combining a linear time-series model with a nonlinear learner, and quantify what the hybridisation actually contributes rather than assuming it helps.", 1),
  NUM("Compare seven approaches on identical chronological splits under metrics that reflect operational cost — including alert burden and detection latency, not accuracy alone.", 1),
  NUM("Deliver a working interactive application where a user supplies input, triggers analysis and explores results — not a static notebook.", 1),
  NUM("Convert raw detections into decision support: classified incident shapes, severity, deviation from forecast, recommended first actions, and a forward capacity outlook.", 1),

  H2("3.4 Intended users"),
  TABLE(
    ["User", "What the system gives them"],
    [
      ["Site-reliability / platform engineer", "Knowledge that a service is abnormal for this time of day, with enough context to begin diagnosis"],
      ["On-call engineer", "A small number of pages worth waking for, each with a classified incident shape and a concrete first action"],
      ["Capacity planner", "A forward forecast with uncertainty band and an estimate of when a service ceiling is likely to be breached"],
      ["Engineering manager", "Evidence that coverage is real — expressed as detection rate and alert burden, not a count of configured alerts"],
    ],
    [27, 73],
    { prose: true }
  ),
  CAPTION("Table 4 — Intended users."),
  new Paragraph({ children: [new PageBreak()] })
);

/* =========================== 4. MODELS ================================== */
children.push(
  H1("4. Models Used for Comparison"),
  P(
    "Seven models are fitted on identical chronological splits of the same preprocessed series. Two are deliberately simple, and they are not straw men: on strongly periodic telemetry a seasonal-naive forecast is hard to beat, and the rolling z-score is what the majority of production monitoring stacks actually ship today."
  ),
  TABLE(
    ["#", "Model", "Family", "Forecasts?", "Detects?"],
    [
      ["1", "Seasonal Naive", "Baseline", "Yes", "Yes (via residual)"],
      ["2", "Rolling Z-Score (3.5σ)", "Statistical", "No", "Yes"],
      ["3", "Holt-Winters ETS", "Classical", "Yes", "Yes (via residual)"],
      ["4", "SARIMAX + Fourier", "Classical", "Yes", "Yes (via residual)"],
      ["5", "Isolation Forest", "Machine learning", "No", "Yes"],
      ["6", "Hybrid SARIMAX + GBR", "Hybrid", "Yes", "Yes (dual channel)"],
      ["7", "Hybrid SARIMAX + Neural", "Hybrid (ablation)", "Yes", "Yes (dual channel)"],
    ],
    [6, 30, 22, 20, 22],
    { prose: true }
  ),
  CAPTION("Table 5 — The seven models compared."),

  H2("4.1 What each model is applied for — role and case covered"),
  P(
    "Each model is in the comparison to answer a specific question. This is the mapping an examiner is most likely to probe, so it is set out explicitly."
  ),
  TABLE(
    ["Model", "Applied for / the exact case it covers"],
    [
      [
        "1. Seasonal Naive",
        "REFERENCE BENCHMARK. Predicts each value from the observation exactly one day earlier. Its in-sample error is the denominator of MASE, so it defines what 'good' means — any model with MASE ≥ 1 has not earned its complexity. Case covered: establishing the floor that every sophisticated model must beat.",
      ],
      [
        "2. Rolling Z-Score (3.5σ)",
        "INDUSTRY-DEFAULT CONTROL. Trailing window supplies a robust median and MAD-scaled spread; anything beyond 3.5σ alerts. Carries no model of seasonality whatsoever. Case covered: quantifying the exact cost of ignoring time-of-day — it is the method our project argues against, included so the argument is measured rather than asserted.",
      ],
      [
        "3. Holt-Winters ETS",
        "LEVEL + TREND + SEASONALITY, SMOOTHED. Additive error, damped additive trend, additive daily seasonality in a compact state vector. Case covered: series dominated by smooth level drift with a stable repeating daily profile, where an interpretable low-parameter state-space model suffices.",
      ],
      [
        "4. SARIMAX + Fourier",
        "AUTOCORRELATION + LONG SEASONAL PERIOD. ARMA(p,d,q) dynamics estimated jointly with 4 Fourier harmonic pairs encoding the 24-hour cycle; order chosen by AIC search. Case covered: series with genuine short-memory autocorrelation and a daily period too long (m = 96) for a textbook seasonal ARIMA. This is also the linear stage of the hybrid.",
      ],
      [
        "5. Isolation Forest",
        "MODEL-FREE, SHAPE-SENSITIVE DETECTION. Random axis-aligned splits over a causal feature window (lags, rolling moments, differences, calendar). Assumes no normality, linearity or stationarity. Case covered: multivariate-shaped oddities a univariate residual threshold misses — e.g. an ordinary value reached at an extraordinary rate. Produces no forecast, so it has no capacity-planning value.",
      ],
      [
        "6. Hybrid SARIMAX + GBR",
        "THE PRINCIPAL CONTRIBUTION. Stage 1 SARIMAX extracts the linear structure; stage 2 gradient-boosted trees learn what remains in its residuals; detection fuses a magnitude z-score with an Isolation Forest over residual shape, then debounces. Case covered: series where the linear model leaves structured residuals (confirmed here by a failed Ljung-Box test) AND where both magnitude anomalies and slow ramps must be caught.",
      ],
      [
        "7. Hybrid SARIMAX + Neural",
        "ABLATION OF THE RESIDUAL LEARNER. Identical architecture with a feed-forward network (or LSTM where PyTorch is installed) replacing gradient boosting in stage 2. Case covered: isolating whether the hybrid's advantage comes from the two-stage architecture itself or from the particular learner bolted into it. If both score alike the gain is architectural; if they diverge, the learner must be tuned per series.",
      ],
    ],
    [23, 77],
    { prose: true, boldRows: ["6. Hybrid SARIMAX + GBR"] }
  ),
  CAPTION("Table 6 — Model-to-case mapping. This is the central design rationale of the comparison."),
  new Paragraph({ children: [new PageBreak()] })
);

/* =========================== 5. HYBRID ================================== */
children.push(
  H1("5. The Hybrid Model in Detail"),
  P(
    "The premise, following Zhang (2003), is that a series decomposes into a linear component a well-specified ARMA can capture and a nonlinear component it structurally cannot. Fitting one model to both forces a compromise; fitting them in sequence lets each do what it is good at."
  ),
  TABLE(
    ["Stage", "Component", "Purpose and what it specifically handles"],
    [
      ["1", "SARIMAX + Fourier", "Extracts autocorrelation and the daily cycle. Produces the linear forecast L̂ₜ. The daily period m = 96 is encoded as 4 Fourier harmonic pairs (8 coefficients) instead of 96 seasonal lags, which is what makes the model estimable from 941 training observations."],
      ["2", "Gradient-boosted trees", "Learns the residual eₜ = yₜ − L̂ₜ from its own 12 preceding lags plus calendar features. Captures burst clustering, asymmetric recovery after a spike, and weekday effects that are not pure harmonics. Removed 70.3% of the variance the linear model left behind."],
      ["3a", "Robust z-score channel", "Magnitude detection on the hybrid residual, calibrated on training residuals only. Weight 0.70. Sharp on size but blind to shape."],
      ["3b", "Isolation Forest channel", "Shape detection over residual behaviour — local volatility, momentum, persistence, slope. Weight 0.30, capped at 6σ-equivalent so it can tip a borderline reading over the line but never raise an alert alone. Catches slow ramps that never exceed σ."],
      ["4", "Debounce", "Requires 2 consecutive buckets above threshold. Converts a large share of false alarms into nothing, at the cost of one bucket of latency."],
    ],
    [8, 22, 70],
    { prose: true }
  ),
  CAPTION("Table 7 — Hybrid architecture, stage by stage."),

  H2("5.1 Why each stage exists"),
  P(
    "No stage was added speculatively. Each responds to a measured deficiency of the one before it, and this is the strongest defence of the design:"
  ),
  TABLE(
    ["Evidence found", "Stage it forced"],
    [
      ["Ljung-Box test at lag 24 gives p ≈ 0.000 — residual autocorrelation remains after the linear fit", "Stage 2 (nonlinear residual learner)"],
      ["A magnitude threshold alone cannot see a slow ramp that never exceeds σ", "Stage 3b (residual-shape channel)"],
      ["Single-bucket exceedances are overwhelmingly noise", "Stage 4 (debounce)"],
      ["Unclamped stage 2 explained away sustained anomalies and went silent mid-incident (0/2 windows on one service)", "2σ clamp on the correction"],
      ["Unbounded shape channel flagged 42% of a hold-out on a very regular training window", "6σ cap on the shape channel"],
    ],
    [60, 40],
    { prose: true }
  ),
  CAPTION("Table 8 — Every design element traced to the evidence that demanded it."),
  new Paragraph({ children: [new PageBreak()] })
);

/* =========================== 6. EVALUATION ============================== */
children.push(
  H1("6. How the Comparison is Judged"),
  TABLE(
    ["Metric", "What it answers", "Why it is in the comparison"],
    [
      ["MASE", "Did it beat the seasonal-naive benchmark?", "Scale-free; MAPE is undefined at zero and byte counts legitimately idle at zero"],
      ["RMSE / MAE", "How far off is the forecast?", "Standard forecast accuracy, reported per-service only (units differ across services)"],
      ["Point-adjusted F1", "Was each incident caught at all?", "The headline — one page is enough to engage a human"],
      ["Point-wise F1", "Did it agree on every bucket?", "Guards against a detector that wins by firing constantly"],
      ["Segment recall", "What fraction of windows were recovered?", "The coverage question a manager actually asks"],
      ["Detection latency", "How long until the first alert?", "A perfect but slow detector is still a bad pager"],
      ["False alarms / day", "How often does it page for nothing?", "Decides whether the alert stays switched on at all"],
    ],
    [20, 34, 46],
    { prose: true }
  ),
  CAPTION("Table 9 — Evaluation metrics and their justification."),
  P(
    "A single composite operational score orders the leaderboard: point-adjusted F1 at 0.45, segment recall 0.25, point-wise F1 0.15, detection latency 0.10 and alert burden 0.05. Alert burden carries negative weight so that no model can win by flagging everything."
  ),

  H2("6.1 Headline results"),
  TABLE(
    ["Model", "Macro comp.", "PA F1", "Windows", "FA/day"],
    [
      ["Rolling Z-Score (3.5σ)", "0.787", "0.901", "5 / 5", "2.52"],
      ["Holt-Winters ETS", "0.749", "0.823", "5 / 5", "4.10"],
      ["SARIMAX + Fourier", "0.709", "0.708", "5 / 5", "1.86"],
      ["Hybrid SARIMAX + GBR", "0.707", "0.687", "5 / 5", "1.36"],
      ["Seasonal Naive", "0.697", "0.760", "5 / 5", "4.87"],
      ["Isolation Forest", "0.552", "0.655", "3 / 5", "1.06"],
      ["Hybrid SARIMAX + Neural", "0.532", "0.456", "4 / 5", "0.92"],
    ],
    [34, 18, 16, 16, 16],
    { boldRows: ["Rolling Z-Score (3.5σ)", "Hybrid SARIMAX + GBR"] }
  ),
  CAPTION("Table 10 — Fleet benchmark across the labelled corpus."),
  RP([
    "The result is reported as found. The simple rolling z-score leads on the composite score. Three honest reasons: the composite weights are ours and, once every model recovers all five windows, the ranking turns largely on point-wise agreement across wide NAB windows; the evidence base is five windows across four services, too thin to separate models differing by 0.08; and on the series examined in detail the daily cycle is weak, so a trailing-window detector tracks the drifting level perfectly well. What should be added in its defence: the hybrid is ",
    ["the quietest detector that still recovers every labelled window", { bold: true }],
    " — 1.36 false alarms per day against the baseline's 2.52 — and it wins outright on two of six services. Under an alert-burden-sensitive weighting it is preferable.",
  ]),
  P(
    "On the single-service run (EC2 CPU utilisation), the Hybrid SARIMAX + GBR ranks first at a composite of 0.860 with zero false alarms per day and all forecasters beating the seasonal-naive benchmark decisively on MASE (0.44–0.48 against 1.30)."
  ),

  H2("6.2 What the system produces for the user"),
  BULLET("A selected model per service, chosen by running the full comparison rather than fixed in advance."),
  BULLET("Incidents merged from contiguous flags, each with severity, classified shape (spike, dip, level shift, ramp, volatility change), duration, peak value, expected value, and percentage deviation from forecast."),
  BULLET("A recommended first action per incident shape — for a ramp, inspect memory and connection pools; for a level shift, check the deploy timeline; for a volatility change, look for contention rather than load."),
  BULLET("A forward capacity outlook with a 95% prediction interval, reporting the first bucket whose upper bound crosses the service ceiling and how many minutes away that is."),
  new Paragraph({ children: [new PageBreak()] })
);

/* =========================== 7. SUMMARY ================================= */
children.push(
  H1("7. One-Paragraph Summary"),
  new Paragraph({
    spacing: { after: 200, line: 300 },
    shading: { type: ShadingType.CLEAR, fill: "F0F6F7" },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT } },
    indent: { left: 180, right: 180 },
    children: [
      new TextRun({
        text: "SENTINEL-TS is a time-series anomaly detection system for cloud server telemetry, built on the Numenta Anomaly Benchmark's realAWSCloudwatch corpus of genuine Amazon CloudWatch data. It solves the problem that static monitoring thresholds cannot represent a time-varying normal: server load is strongly periodic, so any fixed ceiling is either blind at the daily trough or deafening at the peak. The system instead forecasts what each metric should be at each moment and alerts on the residual. Seven models are compared on identical chronological splits — a seasonal-naive benchmark, an industry-default rolling z-score, Holt-Winters ETS, SARIMAX with Fourier seasonal regressors, an Isolation Forest, and two variants of a two-stage hybrid that corrects the SARIMAX forecast with a nonlinear residual learner before fusing magnitude and shape detection channels. The nonlinear stage removes 70.3% of the variance the linear model leaves behind. Detections are converted into classified incidents with severity, deviation from forecast, and recommended operator actions, delivered through a working application with a Python analysis engine, a REST API and a browser console.",
        size: 21,
      }),
    ],
  }),

  H2("Key numbers"),
  TABLE(
    ["Quantity", "Value"],
    [
      ["Raw observations per series", "4,032 (5-minute sampling, ~14 days)"],
      ["After resampling to 15 minutes", "1,345"],
      ["Train / test split", "941 / 404 (30% chronological hold-out)"],
      ["Seasonal period", "m = 96 buckets per day"],
      ["Fourier harmonics", "K = 4 → 8 coefficients (vs 96 seasonal lags)"],
      ["Selected SARIMAX order", "(3, 1, 3), AIC 2,616.99"],
      ["Stationarity verdict", "Non-stationary (ADF p = 0.897, KPSS p ≤ 0.01) → d = 1"],
      ["Residual variance removed by stage 2", "70.3%"],
      ["Best single-service model", "Hybrid SARIMAX + GBR, composite 0.860"],
      ["Best fleet model", "Rolling Z-Score, macro composite 0.787"],
      ["Labelled windows evaluated", "5 of 11 (the rest fall inside training)"],
      ["Test coverage", "101 Python tests + 27 console checks"],
    ],
    [46, 54],
    { prose: true }
  ),
  CAPTION("Table 11 — Figures to have at hand.")
);

/* --------------------------------------------------------------- build */
const doc = new Document({
  creator: "SENTINEL-TS",
  title: "SENTINEL-TS — Project Synopsis",
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 420, hanging: 220 } } } },
        ],
      },
      {
        reference: "numbers",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 440, hanging: 240 } } } },
        ],
      },
    ],
  },
  styles: { default: { document: { run: { font: "Calibri", size: 21 } } } },
  sections: [
    {
      properties: { page: { margin: { top: 1000, right: 1000, bottom: 1000, left: 1000 } } },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({ text: "SENTINEL-TS · Project Synopsis · ", size: 16, color: "9AA5B0" }),
                new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "9AA5B0" }),
              ],
            }),
          ],
        }),
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(process.argv[2] || "SENTINEL-TS_Synopsis.docx", buf);
  console.log("written");
});
