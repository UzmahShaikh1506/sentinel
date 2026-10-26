# SENTINEL-TS

**Hybrid time-series anomaly intelligence for cloud infrastructure telemetry**

A forecasting and anomaly-detection system for server metrics. It compares
seven approaches on real Amazon CloudWatch data — including a two-stage hybrid
that corrects a linear forecast with a nonlinear residual learner — and serves
the winner through an operations console that turns raw flags into incidents
with recommended actions.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify the environment and bundled data
python run.py --check

# 3. Launch the console
python run.py
```

Then open **http://127.0.0.1:8000**.

Nothing else is required. There is no database, no build step, no Node
toolchain, and no network access needed at runtime — the console's chart
engine is hand-written and the dataset ships with the project.

### Other ways to run it

```bash
python run.py --port 9000                       # different port
python run.py --cli ec2_cpu_utilization_5f5533  # one analysis in the terminal
python run.py --benchmark                       # all services, aggregated
python -m pytest                                # the test suite (99 tests)
```

---

## What problem this solves

Cloud services emit continuous telemetry, and incidents appear in that stream
before customers notice. Fixed thresholds miss them for a specific and
expensive reason: **server load is strongly diurnal**. A ceiling calibrated for
the daily peak is blind to an outage at 04:00 when load legitimately collapses;
one calibrated for the overnight trough fires every weekday morning. Teams
respond by widening thresholds until the noise stops, at which point the
detector's real-world recall is zero.

SENTINEL-TS models what each metric *should* be at each moment, then alerts on
the gap. The threshold moves with time of day instead of ignoring it.

**Intended users.** Site-reliability and platform engineers who own service
health, and the on-call rota that must act on whatever gets escalated.

---

## Dataset

| | |
|---|---|
| **Source** | [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) — `realAWSCloudwatch` family |
| **Nature** | Genuine Amazon CloudWatch production server telemetry |
| **Frequency** | 5 minutes native; resampled to 15 minutes by default |
| **Duration** | ≈ 14 days per series (4 032 raw observations) |
| **Series** | 6 services — EC2 CPU, RDS CPU, EC2 network ingress, ELB request count, EC2 disk writes, auto-scaling-group load |
| **Labels** | Human-curated anomaly **windows** (11 across the corpus), not point labels |
| **Licence** | AGPL-3.0 (NAB); data redistributed unmodified under `data/raw/` |

Window labels are the realistic framing: an operator cares that an incident was
caught inside its window, not that one particular five-minute bucket was
flagged.

You can also upload your own CSV through the console. Any file with a timestamp
column and a numeric value column works; column names are detected rather than
assumed.

---

## The models

Seven approaches, fitted on identical chronological splits.

| Model | Family | Forecasts? | Why it is in the comparison |
|---|---|:--:|---|
| **Seasonal Naive** | baseline | ✓ | The reference MASE is defined against. Any model with MASE ≥ 1 has not earned its complexity. |
| **Rolling Z-Score (3σ)** | statistical | — | What most monitoring stacks actually ship. Has no notion of time-of-day, so it quantifies the cost of ignoring seasonality. |
| **Holt-Winters ETS** | classical | ✓ | Additive error, damped additive trend, additive daily seasonality — the structure the EDA actually finds. |
| **SARIMAX + Fourier** | classical | ✓ | ARMA dynamics over harmonic seasonal regressors, order chosen by AIC search. |
| **Isolation Forest** | machine learning | — | Assumes no generative model, so it catches shape anomalies a residual threshold misses. |
| **Hybrid SARIMAX + GBR** | hybrid | ✓ | The main contribution — see below. |
| **Hybrid SARIMAX + Neural** | hybrid | ✓ | Ablation: same architecture, different residual learner. Isolates whether the gain is architectural. |

### Why not a textbook SARIMA?

At 15-minute buckets the daily period is **m = 96**. A
`SARIMA(p,d,q)(P,D,Q)[96]` would need 96 seasonal lags estimated from roughly
900 training observations — not estimable. Projecting the daily cycle onto four
Fourier harmonic pairs captures the same periodic shape in **eight
coefficients** and leaves the ARMA component to model only the short-memory
dynamics it is good at. This is the standard remedy for long seasonal periods.

### The hybrid architecture

```
y_t ──► [ Stage 1: SARIMAX + Fourier ] ──► L̂_t        linear forecast
                  │
                  ▼
           e_t = y_t − L̂_t                            linear residual
                  │
                  ▼
         [ Stage 2: nonlinear learner ] ──► N̂_t       residual correction
                  │
                  ▼
           ŷ_t = L̂_t + N̂_t                            hybrid forecast
                  │
                  ▼
           r_t = y_t − ŷ_t                            hybrid residual
                  │
      ┌───────────┴────────────┐
      ▼                        ▼
robust z of r_t        Isolation Forest over
(train-calibrated)     residual-shape features
      └───────────┬────────────┘
                  ▼
   weighted σ-equivalent score ──► debounce ──► alert
```

Each stage exists because a *measured* weakness of the previous one demanded
it:

1. **Linear stage.** SARIMAX extracts autocorrelation and the daily cycle.
2. **Nonlinear stage.** Its residuals are not white — burst clustering,
   asymmetric recovery after a spike, weekday effects that are not pure
   harmonics. A learner trained on lagged residuals plus calendar context
   absorbs that structure. The console reports exactly how much residual
   variance this removed; a figure near zero would be evidence the hybrid is
   unnecessary on that series, which is a legitimate finding rather than a
   failure.
3. **Dual detection.** A magnitude threshold is sharp but blind to shape: a slow
   ramp that never exceeds σ stays invisible. An Isolation Forest over residual
   *behaviour* (local volatility, momentum, persistence) sees the ramp. The two
   are fused in σ-equivalent units — not by rank, which would destroy the
   interpretable threshold. The shape channel is capped so it can tip a
   borderline reading over the line but can never raise an alert alone.
4. **Debounce.** Single-bucket exceedances are overwhelmingly noise. Requiring
   a short dwell converts most false alarms into nothing, at the cost of one
   bucket of latency. Both sides of that trade are measured and reported.

Following Zhang's hybrid methodology: a series is the sum of a linear component
a well-specified ARMA can capture and a nonlinear one it structurally cannot.
Fitting one model to both forces a compromise; fitting them in sequence lets
each do what it is good at.

---

## How it is evaluated

**Forecasting** — MAE, RMSE, MAPE, sMAPE, MASE, R². MASE matters here because
byte and request counts legitimately idle near zero, where MAPE is undefined or
explodes.

**Detection** — point-wise *and* point-adjusted precision/recall/F1, plus
detection latency and false alarms per day.

- *Point-adjusted F1* is the headline. If any bucket inside a real incident
  window is flagged, the incident was caught — one page is enough to put a human
  on it.
- *Point-wise F1* is reported alongside, because point-adjusted F1 alone would
  hide alert fatigue.
- *False alarms per day* decides adoption. Perfect recall with ten false pages a
  day gets muted within a week.

**Composite ranking** — PA-F1 0.45 · segment recall 0.25 · point-wise F1 0.15 ·
latency 0.10 · alert burden 0.05. Alert burden carries negative weight so no
model can win by flagging everything.

**Fleet benchmark** — a single series carries only one or two labelled windows
in its hold-out, too thin to rank models on. The benchmark runs the identical
pipeline across every labelled service and aggregates two ways: a macro average
(each service weighted equally) and pooled counts (each observation weighted
equally). Where the two disagree, no model dominates, and the summary says so.

### Guarantees about leakage

- Splits are strictly chronological; random splitting would leak future
  information through the series' own autocorrelation.
- Model parameters are estimated on training data only. Hold-out predictions
  come from re-running the filter with those parameters frozen.
- Residual scales and alert thresholds are calibrated on training residuals,
  never on the window being judged.
- Every machine-learning feature at time *t* is built from `shift(≥1)`. A unit
  test perturbs `y_t` and asserts the feature row at *t* is unchanged.

An earlier version of the rolling z-score fell back to a *global* MAD when a
window degenerated — a subtle leak of later information into earlier scores.
`test_rolling_zscore_is_causal` caught it; the fallback is now expanding
(past-only).

---

## Project layout

```
sentinel-ts/
├── run.py                   Launcher: server, CLI, benchmark, --check
├── requirements.txt
├── pytest.ini / conftest.py
│
├── data/
│   ├── raw/                 Six NAB realAWSCloudwatch series (CSV)
│   ├── labels/              Ground-truth anomaly windows (JSON)
│   └── uploads/             User-supplied CSVs land here
│
├── sentinel/                Analysis package — no web code
│   ├── config.py            Paths, defaults, SLO thresholds, RunConfig
│   ├── datasets.py          Catalogue, loading, label windows, uploads
│   ├── preprocessing.py     Cleaning, resampling, gap filling, robust stats
│   ├── eda.py               STL, ADF/KPSS, ACF/PACF, seasonality, distribution
│   ├── metrics.py           Forecast + detection scorecards, ranking
│   ├── incidents.py         Incident synthesis, playbooks, capacity outlook
│   ├── pipeline.py          End-to-end orchestration
│   ├── benchmark.py         Cross-dataset aggregation
│   ├── jobs.py              Background job runner with progress
│   ├── api.py               FastAPI routes (routing only, no analysis)
│   └── models/
│       ├── base.py          Detector contract, Fourier/calendar features
│       ├── baselines.py     Seasonal naive, rolling z-score
│       ├── classical.py     Holt-Winters ETS, SARIMAX + Fourier
│       ├── ml.py            Isolation Forest, residual learners
│       └── hybrid.py        The two-stage hybrid
│
├── frontend/                Console — no build step, no CDN
│   ├── index.html
│   ├── styles.css
│   ├── charts.js            Hand-written canvas chart engine
│   └── app.js
│
└── tests/                   99 tests
```

The analysis package contains no web code and the API contains no analysis.
That separation is what lets the science be tested without a server running.

---

## API

Runs are asynchronous. A full comparison takes tens of seconds and a benchmark
takes minutes; holding an HTTP connection open for either invites proxy
timeouts and tells the client nothing until it finishes. Submit, poll, collect.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness and capability probe |
| `GET` | `/api/datasets` | Catalogue, including uploads |
| `GET` | `/api/datasets/{id}/preview` | Thinned raw series for plotting |
| `GET` | `/api/models` | Model registry with stated rationales |
| `POST` | `/api/datasets/upload` | Register a user CSV |
| `POST` | `/api/run` | Queue an analysis → `202` + job id |
| `POST` | `/api/benchmark` | Queue the cross-dataset benchmark |
| `GET` | `/api/jobs/{id}` | Status, stage and progress |
| `GET` | `/api/jobs/{id}/result` | Full result bundle |
| `GET` | `/api/jobs/{id}/export` | Result as a downloadable JSON file |

Interactive docs at `/docs` while the server runs.

---

## The console

Seven views, all driven by the pipeline's own output — nothing is recomputed
client-side.

- **Mission Control** — pick a service, tune parameters, watch the real
  pipeline progress (every percentage comes from the pipeline, not a timer).
- **Overview** — selected model, incident recovery, capacity outlook, and a
  plain-language verdict that names it when the margin over the baseline is
  slim.
- **Data & EDA** — preprocessing audit trail, STL decomposition, ACF/PACF with
  Bartlett bands, reconciled ADF/KPSS verdict, hour-by-weekday heatmap,
  distribution.
- **Model Comparison** — leaderboard, score and RMSE bars, overlaid forecasts,
  per-model parameters, AIC order search and Ljung-Box diagnostics.
- **Detection** — score against threshold per model, and a raster showing where
  every model fires against ground truth.
- **Alerts** — incidents with severity, classified shape (spike, dip, level
  shift, ramp, volatility), deviation from forecast, and an operator playbook.
- **Fleet Benchmark** — aggregate results across the whole corpus.

---

## Known limitations

These are stated rather than hidden; several are discussed in the console
itself.

1. **Labelled windows are scarce.** Eleven windows across six series, one or
   two per hold-out. The fleet benchmark mitigates this but cannot remove it.
   Differences of a few thousandths in composite score are not meaningful.
2. **Reported latency is inflated by the labelling convention.** NAB windows
   open *before* the visible excursion so that early detection scores well. A
   latency of several hours against the window start can still be prompt
   against the point a human would have noticed. Unadjusted here.
3. **Univariate.** Each metric is modelled alone. Real incidents correlate
   across CPU, memory and latency simultaneously; a VAR or multivariate model
   would exploit that. The syllabus VAR/ECM material is the natural extension.
4. **No concept-drift handling.** Parameters are fitted once. A production
   deployment needs scheduled refitting, and a deliberate level shift (a
   deployment) will keep paging until the baseline is retrained.
5. **The hybrid's forward forecast is linear-only.** The nonlinear correction
   is a function of recent residuals, which do not exist beyond the data.
   Extrapolating it would compound error and hand the operator a falsely
   confident curve.
6. **Isolation Forest cannot report all-clear.** Its threshold is capped at the
   contamination quantile, so it always flags roughly that fraction.
7. **Rule-based incident classification.** Transparent and auditable, but a
   fixed rule set. It cannot recognise a shape nobody encoded.
8. **Single-process, in-memory jobs.** Results do not survive a restart. Fine
   for one analyst; a real deployment needs a job store and a worker pool.

## Possible extensions

- Multivariate detection via VAR / cointegration across correlated metrics
- GARCH for volatility-regime anomalies, where the variance rather than the
  mean is the signal
- Online/incremental refitting with drift detection
- Streaming ingestion from Prometheus or CloudWatch instead of CSV
- Integration with a real paging system, with alert feedback closing the loop
  on threshold tuning

---

## Optional: enabling the LSTM stage

The hybrid's residual learner is pluggable. Gradient boosting is the default;
installing PyTorch activates a true LSTM path automatically:

```bash
pip install torch
```

Without it, `ResidualLearner` detects the absence, falls back to a feed-forward
network, and the console reports that it did. Nothing silently mislabels an MLP
as an LSTM.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | `pip install -r requirements.txt`, then `python run.py --check` |
| `Address already in use` | `python run.py --port 9000` |
| Console loads but shows "Could not reach the analysis engine" | The API did not start; check the terminal for a traceback |
| Upload rejected | Needs a parseable timestamp column and a numeric value column; max 8 MB |
| A run takes a long time | Expected: ~20–40 s for seven models. Coarsen the frequency to `30min` or `1h`, or deselect models |
| Charts look empty | The chart engine needs `<canvas>` support; use any current browser |
