# Screenshot checklist

Deliverable 7 asks for "screenshots of the application and its outputs." These
have to be captured on your own machine — the console runs on `localhost` and
cannot be photographed from here.

## How to capture

```bash
python run.py                 # then open http://127.0.0.1:8000
```

Press **Run analysis** once with the defaults so every view has data in it,
then work down this list. Full-window captures at a reasonable browser width
(1400 px or wider) look best; the layout reflows below about 1180 px.

macOS: `Cmd+Shift+4`, then Space, then click the window.
Windows: `Win+Shift+S`.

## The shots to take

| # | View | What must be visible | Maps to |
|---|---|---|---|
| 1 | Mission Control | Service list with sparklines, parameter panel, series preview with the ground-truth window shaded | §6.4, §6.5 |
| 2 | Mission Control, mid-run | Pipeline stages with one active and the progress bar partway | §6.2.1 |
| 3 | Overview | Verdict banner, four metric tiles, observed-vs-expected chart with the normal band and raised alerts | §5.2, §7 |
| 4 | Overview, lower | Forward capacity outlook with its 95% interval, and the alert profile panel | §7.4 |
| 5 | Data & EDA, upper | Audit-trail tiles and the preprocessing table | §2.5 |
| 6 | Data & EDA, decomposition | All four STL panels with the variance-share chips | §3.1 |
| 7 | Data & EDA, ACF/PACF | Both stem plots with Bartlett bands | §3.3 |
| 8 | Data & EDA, stationarity | Both test cards and the reconciled verdict banner | §3.2 |
| 9 | Data & EDA, seasonality | Hour-by-weekday heatmap and the distribution histogram | §3.4, §3.5 |
| 10 | Model Comparison | The full leaderboard with the winner row highlighted | §5.2 |
| 11 | Model Comparison, charts | Composite-score bars and the overlaid forecast comparison | §5.1.3 |
| 12 | Model Comparison, hybrid card | Hybrid parameters, the AIC order search table, and the variance-reduction note | §4.3, §4.4.2 |
| 13 | Detection | Score-against-threshold chart for the hybrid, with alerts marked | §4.4.4 |
| 14 | Detection, raster | The per-model firing raster against ground truth | §5.2 |
| 15 | Alerts | Incident cards, with at least one expanded to show its playbook | §7.2, §7.3 |
| 16 | Fleet Benchmark | Aggregate leaderboard and per-service winners | §5.3, §5.4 |
| 17 | API docs | `http://127.0.0.1:8000/docs` | §6.3 |
| 18 | Terminal | `python -m pytest` showing 101 passed | §6.6 |

## Two worth staging deliberately

**Upload flow.** Export any series to CSV, upload it through Mission Control,
and capture the result. It demonstrates that the application accepts user input
rather than only shipping fixed data — and that unlabelled series correctly
report detection metrics as unavailable instead of inventing a score.

**A failing configuration.** Set the threshold slider to 1.5σ and run. The
alert count climbs sharply and false alarms per day with it. This is the
cleanest single demonstration of the alert-burden argument in §5.1.3, and it is
a good thing to have on hand if an examiner asks why the threshold matters.
