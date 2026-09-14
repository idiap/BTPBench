# Verification Experiment

The verification pipeline compares every unique pair in the configured sample
set. Same-subject pairs are genuine comparisons and different-subject pairs are
impostor comparisons; the resulting scores support 1:1 biometric performance
analysis.

## Running the Pipeline

```bash
uv run btpbench verification pipeline \
  -s config/system_config.yaml \
  -e config/experiment_config.yaml \
  [-o ./output_dir] [--override]
```

| Argument | Description |
|---|---|
| `-s` | Path to the system configuration file (required). |
| `-e` | Path to the experiment configuration file (required). |
| `-o` | Output directory (optional, overrides value from experiment config). |
| `--override` | Overwrite existing results. |

### Pipeline Workflow

1. Extract features from all samples using the baseline algorithm.
2. **Unprotected matching:** compare all samples pairwise, output scores.
3. **Protected matching** (if BTP algorithms are configured): for each BTP algorithm, protect all templates, compare pairwise, and output scores.

### Output

Score files are saved as `verification-<baseline>.csv` (unprotected) and `verification-<btp_config>-<baseline>.csv` (protected).

## Computing Metrics

```bash
uv run btpbench verification metrics \
  -s verification_score1.csv [-s verification_score2.csv ...] \
  -l "Label1" [-l "Label2" ...] \
  -o verification_metrics.csv \
  [-f 0.01 -f 0.1]
```

| Argument | Description |
|---|---|
| `-s`, `--score-file` | Verification score file(s) (repeatable, required). |
| `-l`, `--label` | Label for each score file (repeatable and required; the number of labels must match the number of `-s` arguments). |
| `-o` | Output CSV file (required). |
| `-f` | FMR values at which to compute metrics (repeatable, e.g., `0.01`, `0.001`). |

### Metrics Produced

| Metric | Description |
|---|---|
| FTA | Failure to Acquire rate. |
| FMR @ threshold | False Match Rate at specified threshold(s). |
| FNMR @ threshold | False Non-Match Rate at specified threshold(s). |

## Generating Plots (DET Curves)

```bash
uv run btpbench verification plots \
  -f verification_score.csv [-f verification_score2.csv ...] \
  [-l "Label1" -l "Label2" ...] \
  -o det_curves.png \
  [-t "My Plot"]
```

| Argument | Description |
|---|---|
| `-f` | Score file(s) to plot (repeatable, required). |
| `-l` | Custom labels (optional). |
| `-o` | Output PNG file (required). |
| `-t` | Plot title (default: `"DET curves"`). |

The plot shows Detection Error Tradeoff (DET) curves: FMR vs FNMR on
normal-deviate (probit) axes, with ticks labeled as percentages.

## Experiment Configuration

The verification pipeline uses the [common fields](experiment_config.md#common-fields) only. It does not use `protocols` or `splits` — it operates on the `verification_samples` file defined in the system configuration for the selected database.

See [Protocol Format and Custom Databases](protocols.md#verification) for the
sample-list schema and guidance on creating genuine and impostor pairs.

For BTP algorithm configuration, see [experiment_config.md — BTP Algorithms](experiment_config.md#btp-algorithms).
