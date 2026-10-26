# Dataset documentation

Required deliverable 2: "Dataset and its source documentation."

## Source

| | |
|---|---|
| **Name** | Numenta Anomaly Benchmark (NAB), `realAWSCloudwatch` family |
| **Repository** | https://github.com/numenta/NAB |
| **Publication** | Ahmad, S., Lavin, A., Purdy, S. and Agha, Z. (2017). *Unsupervised real-time anomaly detection for streaming data.* Neurocomputing, 262, 134–147. |
| **Licence** | AGPL-3.0 |
| **Redistribution** | Series are included unmodified under `data/raw/`; labels under `data/labels/combined_windows.json` |
| **Retrieved** | October 2026 from `raw.githubusercontent.com/numenta/NAB/master` |

The `realAWSCloudwatch` series are genuine Amazon CloudWatch measurements from
production Amazon Web Services infrastructure. They are not simulated.

## Structure

Each file is a two-column CSV:

| Column | Type | Description |
|---|---|---|
| `timestamp` | ISO 8601 datetime | Measurement time, UTC, regular 5-minute grid |
| `value` | float | The metric reading, in the unit given below |

## Series

| File | Metric | Unit | Observations | Span | Windows |
|---|---|---|---:|---|---:|
| `ec2_cpu_utilization_5f5533.csv` | EC2 instance CPU utilisation | % | 4,032 | 2014-02-14 → 2014-02-28 | 2 |
| `rds_cpu_utilization_cc0c53.csv` | RDS database CPU utilisation | % | 4,032 | 2014-02-14 → 2014-02-28 | 2 |
| `ec2_network_in_257a54.csv` | EC2 inbound network volume | bytes | 4,032 | 2014-04-10 → 2014-04-24 | 1 |
| `elb_request_count_8c0756.csv` | Load balancer request count | req/5 min | 4,032 | 2014-04-10 → 2014-04-24 | 2 |
| `ec2_disk_write_bytes_1ef3de.csv` | EC2 disk write throughput | bytes | 4,730 | 2014-02-14 → 2014-03-02 | 1 |
| `grok_asg_anomaly.csv` | Auto-scaling group aggregate load | units | 4,621 | 2014-01-15 → 2014-01-31 | 3 |

**Total: 25,477 observations, 11 labelled anomaly windows.**

## Frequency and duration

- **Native frequency:** 5 minutes (288 observations/day).
- **Duration:** approximately 14 days per series, 16 days for the two longer ones.
- **Analysis frequency:** resampled to 15 minutes by default (96 buckets/day),
  user-selectable between 5 minutes and 1 hour. See report §2.4.3 for why.

## Ground truth

`combined_windows.json` maps each series to a list of `[start, end]` timestamp
pairs delimiting human-curated anomaly windows. Example:

```json
"realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv": [
  ["2014-02-18 16:02:00.000000", "2014-02-19 08:42:00.000000"],
  ["2014-02-24 10:17:00.000000", "2014-02-25 02:57:00.000000"]
]
```

Two properties of this labelling affect every reported figure and are discussed
in the report rather than worked around:

1. **Windows, not points.** An incident is credited as detected if any bucket
   inside its window is flagged. This drives the use of point-adjusted F1.
2. **Windows open early.** They begin *before* the visibly anomalous behaviour,
   by design, so that early detection is rewarded. Measured detection latency
   against the window start is therefore systematically pessimistic.

A third property constrains the evaluation: at a 30% chronological hold-out,
only **5 of the 11 windows** fall inside a test set. On `ec2_network_in` and
`ec2_disk_write_bytes` the single window lies entirely within the training
portion, so detection cannot be scored on those services at all.

## Data quality

Measured by the pipeline's audit trail at 15-minute buckets:

| Series | Coverage | Duplicates | Non-finite | Gaps filled | Extremes flagged |
|---|---:|---:|---:|---:|---:|
| `ec2_cpu_utilization_5f5533` | 100.0% | 0 | 0 | 0 | 38 (2.8%) |

The bundled series are complete — no gaps, duplicates or invalid readings. The
gap-filling and deduplication machinery exists for user-uploaded series and for
corpora with real outages; on this data it is not exercised. That is itself
useful, because it means the model comparison is not confounded by imputation
artefacts.

## Using your own data

Any CSV with a timestamp column and a numeric value column can be uploaded
through the console. Column names are discovered rather than assumed — a file
naming its columns `date` and `reading` loads without edits. Uploaded series
carry no ground-truth labels, so the system reports detection metrics as
unavailable rather than fabricating a score, and ranks models by forecast error
and alert volume instead.
