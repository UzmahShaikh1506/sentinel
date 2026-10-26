# SENTINEL-TS — Viva preparation

Every group member must be able to answer all of these. The guidelines state
explicitly that individual contributions are assessed during the demonstration
and viva, and that all members must be prepared to explain the dataset,
preprocessing, models, evaluation and application workflow.

---

## A. Dataset

**Q. What is your dataset and where did it come from?**
The `realAWSCloudwatch` family of the Numenta Anomaly Benchmark — genuine
Amazon CloudWatch telemetry from production AWS infrastructure. Six series,
5-minute sampling, about 14 days each (4,032 raw observations per series),
two columns: timestamp and value. Eleven human-curated anomaly windows across
the corpus. Published on GitHub under AGPL-3.0.

**Q. Why real data instead of synthetic?**
Synthetic data contains exactly the structure its author put there, so a model
that assumes that structure always appears to succeed. Real telemetry has flat
idle stretches, mixed noise regimes, and anomalies of several different shapes,
and it punishes assumptions that do not hold. Our hybrid's biggest design flaw
was only exposed by real data on a service we had not tuned against.

**Q. Why are the labels windows rather than points?**
Because that is how incidents work. An engineer cares that the incident was
detected inside its window, not that one specific five-minute bucket was
flagged. It also drives our choice of point-adjusted F1 as the headline metric.

**Q. Any problem with the labelling you should disclose?**
Yes. NAB windows deliberately open *before* the visible excursion so that early
detection is rewarded. That makes our reported detection latency systematically
pessimistic — several hours against the window start can still be prompt
against the point a human would have noticed. We did not adjust for it.

**Q. How many anomalies actually get evaluated?**
Eleven windows exist, but at a 30% chronological hold-out only **five** fall
inside a test set. On two services the single window lies entirely in the
training portion, so detection cannot be scored there at all. Our aggregate
rests on 5 windows across 4 services, and we say so in the report. This is the
main limit on how much confidence the ranking deserves.

---

## B. Preprocessing

**Q. Walk me through your preprocessing.**
Duplicate timestamps collapsed by mean (not dropped — agent double-reporting is
common and both readings are informative). Non-finite readings masked. Resample
from 5 to 15 minutes by bucket mean. Gap filling split by gap length. Outliers
flagged but never removed.

**Q. Why resample to 15 minutes?**
Two reasons. At 5 minutes the daily seasonal period is m = 288; at 15 minutes
it is m = 96, which is what makes the model estimable at all (see D). It also
cuts 4,032 points to 1,345, keeping a seven-model comparison inside interactive
latency. The frequency is user-selectable from 5 minutes to 1 hour — it is a
default, not a constraint.

**Q. Why fill short and long gaps differently?**
A five-minute gap and a four-hour outage are different phenomena. Gaps up to 6
buckets get time-interpolated. Longer gaps are reconstructed from the seasonal
profile — the median at that position in the daily cycle. Interpolating a
straight line across a multi-hour gap, in a region where the real signal has a
pronounced daily shape, would create something that itself reads as an anomaly
to every detector downstream.

**Q. Why don't you remove outliers?**
Because in anomaly detection the extremes *are* the signal. Removing them
removes the phenomenon under study. This is the opposite of standard
forecasting practice and it is deliberate. We flag them for display and carry
them forward unchanged; we make the *statistics* robust instead, using
median/MAD rather than mean/standard deviation.

**Q. What is MAD and why 1.4826?**
Median absolute deviation: the median of |x − median(x)|. Multiplying by 1.4826
(which is 1/Φ⁻¹(0.75)) makes it a consistent estimator of σ for Gaussian data,
so it is comparable to a standard deviation but has a 50% breakdown point
instead of 0%. A single huge value cannot move it.

---

## C. EDA

**Q. Why STL rather than classical decomposition?**
Classical decomposition centres a moving average, which a single large spike
distorts badly. STL fits locally weighted regressions and is robust to exactly
the extremes we are hunting. Using a method the anomalies break would
compromise every downstream estimate.

**Q. What did the decomposition find?**
On the EC2 CPU series: trend 93.0% of variance, seasonal only 1.7%, remainder
9.0%. Seasonal strength 0.116, trend strength 0.909.

**Q. That contradicts your motivation. Doesn't it?**
Partly, and we report it rather than hiding it. On *that* series the daily
cycle is weak and slow level drift dominates. Other series in the corpus — the
load-balancer request count especially — show the strong diurnal profile the
problem statement describes. It is also part of why a seasonality-blind
baseline does as well as it does, which we discuss in Section 8.1.

**Q. Why run both ADF and KPSS?**
They have opposite nulls. ADF's null is that a unit root is present
(non-stationary); KPSS's null is stationarity around a level. Declaring
stationarity from one test alone is a standard error. Only agreement licenses a
differencing decision, and disagreement is itself informative.

**Q. What did they say, and what did you do about it?**
ADF statistic −0.473, p = 0.897 → cannot reject the unit root. KPSS statistic
5.437, p ≤ 0.01 → rejects stationarity. Both agree the level wanders, so the
verdict is non-stationary and the recommended differencing order is **d = 1**.
That value is passed directly into the SARIMAX candidate set.

**Q. How did you pick p and q?**
From ACF and PACF with Bartlett bands of ±1.96/√n = ±0.053. Both cut off after
lag 3, proposing p = q = 3. An independent AIC search over six candidate orders
then selected ARIMA(3,1,3) — the same specification. The graphical diagnostics
and the information criterion agreeing is a useful consistency check.

---

## D. Models

**Q. Why not just use SARIMA?**
Arithmetic, not taste. At 15-minute buckets m = 96, so SARIMA(p,d,q)(P,D,Q)[96]
would need seasonal terms at lags that are multiples of 96, estimated from 941
training observations. The state-space representation grows with the seasonal
period: estimation becomes unstable and fitting becomes too slow for an
interactive app. Dropping to hourly buckets would fix m = 24 but throws away
three quarters of the temporal resolution — and detection latency is measured
in buckets.

**Q. So what did you do instead?**
Harmonic regression. Project the daily cycle onto K = 4 Fourier pairs and feed
them to SARIMAX as exogenous regressors. The daily shape is captured in 8
coefficients rather than 96 seasonal lags, and the ARMA part models only the
short-memory dynamics it is good at. This is the standard remedy for long
seasonal periods.

**Q. Explain your hybrid.**
Two stages. Stage 1 is SARIMAX + Fourier, giving a linear forecast L̂ₜ. Its
residual eₜ = yₜ − L̂ₜ is not white — we checked, Ljung-Box at lag 24 gives
p ≈ 0.000. Stage 2 is a gradient-boosted tree that predicts eₜ from the 12
preceding residuals plus calendar features, giving a correction N̂ₜ. The hybrid
forecast is ŷₜ = L̂ₜ + N̂ₜ. Detection then fuses two channels over the hybrid
residual: a robust z-score for magnitude (weight 0.70) and an Isolation Forest
over residual *shape* (weight 0.30), both in σ-equivalent units, followed by a
two-bucket debounce.

**Q. Why hybridise at all? Justify it with evidence.**
Because the Ljung-Box test failed — there was structure the linear model had
not extracted. Stage 2 then removed **70.3%** of the residual variance the
linear model left behind. That is the direct measurement of what hybridisation
bought. The console reports this figure for every run; a value near zero would
be evidence the hybrid is unnecessary for that series, which we would report as
a legitimate finding.

**Q. What is the theoretical basis?**
Zhang (2003): a series decomposes into a linear component a well-specified ARMA
can capture and a nonlinear one it structurally cannot. Fitting one model to
both forces a compromise; fitting them in sequence lets each do what it is good
at.

**Q. Tell me about a bug you found in your own model.** *(expect this one)*
The most important one. An unconstrained residual learner predicts eₜ from its
own recent lags. During a **sustained** anomaly, residuals are large but highly
persistent — which makes them easy to forecast. The learner predicted them
accurately, the correction cancelled them, the hybrid residual collapsed to
near zero, and the detector went **silent exactly while the incident was
happening**. On the ELB request-count service the hybrid scored 0.050 composite
and recovered 0 of 2 windows while every other model recovered both.

We only found it because we benchmarked across all six services instead of one.
The fix follows from what stage 2 is *for* — tightening ordinary residuals, not
tracking excursions: we winsorise the lagged inputs and clamp the correction at
2 training sigmas. During a real excursion the correction saturates and the
hybrid degrades gracefully to its linear stage, which detects the event. Macro
composite went 0.530 → 0.707, point-adjusted F1 0.452 → 0.687, windows
recovered 4/5 → 5/5. A regression test now asserts the hybrid alerts through
more than half of a sustained synthetic excursion.

**Q. Why cap the shape channel?**
Two reasons. Design: with weight 0.30 and a cap of 6σ it contributes at most
1.8, below any sane threshold, so shape can tip a borderline magnitude reading
over the line but can never raise an alert alone — the cap *enforces* the stated
design rather than merely describing it. Empirics: without it, a very regular
training window gives the forest a razor-thin manifold, every hold-out point
scores as strange, and the hybrid flagged 42% of a synthetic hold-out.

**Q. Why a debounce, and what does it cost?**
Single-bucket exceedances are overwhelmingly noise. Requiring two consecutive
buckets converts a large share of false alarms into nothing, at the cost of one
bucket of detection latency. We measure and report both sides rather than
assuming the trade is favourable.

**Q. Is your LSTM a real LSTM?**
Only if PyTorch is installed. The residual learner is pluggable: gradient
boosting by default, a feed-forward network as an ablation, and a true LSTM
path that activates automatically when torch is present. When it is not, the
system falls back to the feed-forward network **and says so in the console**.
We deliberately do not mislabel an MLP as an LSTM.

**Q. What is Isolation Forest doing and what is its weakness?**
It isolates points by random axis-aligned splits; anomalies need fewer splits
and sit at shallower average depth. No assumption of normality, linearity or
stationarity, so it can catch shape anomalies a univariate residual threshold
misses. Weaknesses: it produces no forecast, so no capacity-planning value; its
scores are hard to explain to an on-call engineer; and because we threshold at
the contamination quantile it can never report all-clear — it always flags
roughly that fraction.

---

## E. Evaluation

**Q. Why MASE as well as MAPE?**
Request counts and byte counts in this corpus legitimately idle at or near zero
overnight. MAPE is undefined at zero and explodes near it. MASE is scaled by
the in-sample seasonal-naive error, so below 1 means you beat the naive
benchmark and above 1 means you did not. Our forecasters sit at 0.44–0.48
against seasonal naive's 1.30.

**Q. What is point-adjusted F1 and why is it your headline?**
If any point inside a ground-truth window is flagged, the whole window is
credited. It reflects how an on-call rota actually works — one page is enough
to engage a human. It is applied only to true segments, so it never
manufactures true positives outside them.

**Q. Then why report point-wise F1 at all?**
Because point-adjusted F1 alone would flatter a detector that fires
constantly. Reporting both means neither failure mode hides. Our point-wise F1
values are low (0.16–0.48) and that is expected, not a defect: NAB windows are
wide and no detector fires continuously across a six-hour window, nor should it.

**Q. Why do you weight false alarms negatively?**
Because a detector with perfect recall and ten false pages a day gets muted
within a week, at which point its real-world recall is zero. Alert burden is
what decides adoption, so it belongs in the score.

**Q. Your hybrid lost to a simple z-score. Explain.** *(expect this one)*
Correct — rolling z-score 0.787 macro composite, hybrid 0.707. Three honest
reasons. (1) The composite weights are ours: PA-F1 and segment recall carry
0.70, so once every model recovers all five windows the ranking is decided
largely by point-wise agreement across wide windows, which has little
operational meaning. (2) The evidence base is five windows across four
services — not enough to separate models differing by 0.08. (3) On the series
we examined in detail the daily cycle is weak, and a trailing-window detector
tracks a slowly drifting level perfectly well.

What we would add: the hybrid is the **quietest detector that still recovers
every labelled window** — 1.36 false alarms/day against the baseline's 2.52,
and it wins outright on two of six services. Under an alert-burden-sensitive
weighting it is preferable. We report the ranking as found rather than tuning
the weights until our model wins.

**Q. Why not just retune the weights so the hybrid wins?**
Because that is fitting the evaluation to the conclusion. The weights encode a
stance about what the product is for, and we fixed them before reading the
results.

**Q. Why a fleet benchmark?**
One series gives one or two windows in its hold-out; with one window segment
recall is 0 or 1 and the leaderboard turns on a handful of points. We run the
identical pipeline across every labelled service and aggregate two ways — macro
(each service equal) and pooled (each observation equal). They answer different
questions, and a model leading on one but not the other is unstable. Here they
agree, which we report as the leader being stable to the aggregation choice.

**Q. Why don't you average RMSE across services?**
Because the services are measured in percent, bytes and request counts.
Averaging their RMSEs averages incommensurable quantities. We report RMSE
per-service and use MASE, which is scale-free, for the aggregate.

---

## F. Leakage

**Q. How do you prevent leakage?**
Four guarantees. (1) Splits are strictly chronological — random splitting leaks
future information through the series' own autocorrelation. (2) Parameters are
estimated on training data only; hold-out predictions come from re-running the
filter with parameters frozen. (3) Residual scales and thresholds are
calibrated on training residuals, never on the window being judged. (4) Every
ML feature at time t is built from `shift(≥1)` — a test perturbs yₜ and asserts
the feature row at t is unchanged.

**Q. Did you ever actually have a leak?**
Yes, and a test caught it. An earlier rolling z-score fell back to a **global**
MAD when a window degenerated, and back-filled during warm-up. Both quietly
admitted later information into earlier scores. The test injects a spike at
position 200 and asserts no score before position 190 changes; it failed with
discrepancies of order 10⁻³. The fallbacks are now expanding — past-only — and
warm-up is scored as zero rather than filled from the future.

**Q. Isn't using `contamination` on the test window a form of leakage?**
No, but it is worth being precise. Contamination is a user-supplied prior on
the alert *rate*; it carries no information about *where* the anomalies are. We
take the stricter of the training quantile and the scored-window quantile. The
real cost is that the detector can never report all-clear, and we state that as
a limitation rather than hiding it.

---

## G. Application

**Q. Why is this not just a notebook?**
Three layers: a Python analysis package with no web code, a FastAPI service
with no analysis, and a browser console. That separation lets the entire
scientific pipeline be tested without a server running. The user selects a
service, sets eight parameters, uploads their own CSV if they want, triggers a
run, watches real progress, and explores seven analysis views.

**Q. Why are runs asynchronous?**
A seven-model comparison takes 20–40 seconds and a benchmark takes minutes.
Holding an HTTP connection open invites a proxy timeout and tells the client
nothing until it finishes. Submit → poll → collect. It is also what makes the
progress bar honest: every percentage is published by the pipeline itself, not
interpolated by a timer.

**Q. Why did you write your own chart engine?**
No external stylesheets, fonts or scripts are loaded, so it runs on a lab
machine with no network. The plots also need marks general-purpose libraries do
not express cleanly — ground-truth window shading, anomaly markers over a
confidence band, ACF stems with Bartlett bands — and the reveal animation needs
frame-level control. About 700 lines of canvas code.

**Q. How is it tested?**
101 Python tests across preprocessing, metrics, models and API, plus a headless
console smoke test (27 checks) that drives a real run through the UI in jsdom.
Three real defects were found by tests rather than inspection: the causality
leak, the unbounded shape channel, and a chart-engine crash where the base
constructor called `draw()` before any subclass had initialised its fields —
which broke the first paint of every chart and was invisible to the Python
suite entirely.

**Q. Show me one use case end to end.**
Select `api-gateway-prod-01`, keep defaults, press Run. The pipeline cleans
4,032 readings to 1,345 buckets, runs the EDA, splits 941/404, fits seven
models, and selects Hybrid SARIMAX + GBR at composite 0.860. Four incidents are
raised, all four confirmed by ground truth, zero false alarms/day. The top one:
**critical, volatility change, peak 54.90% against an expected 39.78% (+38%),
sustained 165 minutes from 2014-02-24 21:15.** The playbook tells the engineer
to look for contention rather than load — noisy neighbours, GC or lock pauses,
a flapping dependency. The capacity outlook shows the forecast staying below
the 85% ceiling across 24 hours including the upper 95% bound, so no capacity
action is indicated.

**Q. Why is incident classification rule-based rather than a model?**
An engineer paged at 03:00 needs to know *why* within seconds. A short list of
audited rules is defensible in a way a second black box is not. The cost is
that it cannot recognise a shape nobody encoded, and we list that as a
limitation.

**Q. Why is the hybrid's forward forecast linear-only?**
The nonlinear correction is a function of recent residuals, which do not exist
beyond the end of the data. Feeding its own predictions back as inputs would
compound error with no accuracy gain and hand the operator a falsely confident
curve. The console states this rather than showing a curve of unclear basis.

---

## H. Limitations and future work

**Q. Biggest weakness?**
The evidence base: five labelled windows across four services. Rankings have
wide uncertainty and differences of a few thousandths are noise. Second is that
the system is univariate — real incidents correlate across CPU, memory and
latency simultaneously, and we score each metric independently.

**Q. What would you do next?**
Multivariate detection via VAR and cointegration, so one incident visible in
three metrics is scored once with higher confidence rather than three times
independently. GARCH for volatility-regime anomalies, where the variance rather
than the mean is the signal — we currently detect that shape heuristically.
Online refitting with drift detection, so a deployment that shifts the baseline
retrains it instead of paging forever. Streaming ingestion from Prometheus or
CloudWatch to make it deployable rather than demonstrable.

**Q. What did you learn?**
That cross-validation across *services*, not just across time, is what finds
real defects. Three substantive flaws in our hybrid were invisible on a single
series and surfaced only under cross-dataset or adversarial testing. Each was
found, diagnosed, fixed and regression-tested — and documented rather than
quietly corrected.

---

## Quick numbers to have memorised

| | |
|---|---|
| Raw observations per series | 4,032 (5-minute sampling, ~14 days) |
| After resampling to 15 min | 1,345 |
| Train / test split | 941 / 404 (30% chronological) |
| Seasonal period | m = 96 buckets/day |
| Fourier harmonics | K = 4 → 8 coefficients |
| Selected SARIMAX order | (3, 1, 3), AIC 2,616.99 |
| ADF / KPSS | −0.473 (p 0.897) / 5.437 (p ≤ 0.01) → d = 1 |
| Ljung-Box lag 24 | p ≈ 0.000 → residual structure remains |
| Residual variance removed by stage 2 | 70.3% |
| Correction clamp / shape cap / debounce | 2σ / 6σ / 2 buckets |
| Best single-service model | Hybrid SARIMAX + GBR, composite 0.860 |
| Best fleet model | Rolling Z-Score, macro composite 0.787 |
| Hybrid fleet position | 4th at 0.707; quietest full-recall detector (1.36 FA/day) |
| Windows evaluated | 5 of 11 (the rest fall in training) |
| Tests | 101 Python + 27 console checks |
