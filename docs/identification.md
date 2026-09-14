# Identification Experiment

The identification pipeline performs 1:N template matching — each probe is compared against all references to determine identity.

## Running the Pipeline

```bash
uv run btpbench identification pipeline \
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

1. Extract features from reference and probe templates using the specified baseline algorithm.
2. **Unprotected matching:** compare all probes to all references, output scores.
3. **Protected matching** (if BTP algorithms are configured): for each BTP algorithm, protect templates, compare, and output scores.

### Output

Score files are saved as `scores-<protocol>-<split>-<baseline>.csv` (unprotected) and `scores-<protocol>-<split>-<btp_config>-<baseline>.csv` (protected).

## Computing Metrics

```bash
uv run btpbench identification metrics \
  -f score1.csv [-f score2.csv ...] \
  -l "Label1" [-l "Label2" ...] \
  -o metrics.csv \
  [-n 20] [-d 0.1 -d 0.3 -d 0.5] [-p -1] \
  [--large-file-threshold-mb 1024] [--chunk-size 500000] \
  [--rank1-cache-dir ./rank1-cache]
```

| Argument | Description |
|---|---|
| `-f` | Score file(s) to analyze (repeatable, required). |
| `-l` | Label(s) for each score file (required, must match number of `-f`). |
| `-o` | Output CSV file (required). |
| `-n` | Number of resampling iterations (default: `20`). |
| `-d` | Decimation proportions (repeatable, default: `0.1`, `0.3`, `0.5`, meaning 10%, 30%, and 50%). |
| `-p` | Score precision; `-1` uses full precision (default: `-1`). |
| `--large-file-threshold-mb` | Stream score files at least this large instead of loading the full CSV in memory (default: `1024`; use `0` to stream every file). |
| `--chunk-size` | Number of CSV rows read per chunk in streaming mode (default: `500000`). |
| `--rank1-cache-dir` | Optional directory for reusable rank-1 summaries generated in streaming mode. |

### Metrics Produced

| Metric | Description |
|---|---|
| FTA | Failure to Acquire rate. |
| TPIR | True Positive Identification Rate at Rank 1 at the threshold selected for a 1% FPIR target, for each decimation level. Mean and standard deviation come from resampling. |
| FPIR | Achieved False Positive Identification Rate at Rank 1 at that threshold. Mean and standard deviation come from resampling. |
| Threshold | Decision threshold selected for a 1% FPIR target. Mean and standard deviation come from resampling. |

Metrics are computed at each decimation level (e.g., 10%, 30%, 50% of the
gallery). The target FPIR is fixed at 1% and is not configurable from the CLI.

## Generating Plots (DIR Curves)

```bash
uv run btpbench identification plots \
  -f score1.csv [-f score2.csv ...] \
  [-l "Label1" -l "Label2" ...] \
  -o dir_curves.png \
  [-t "My Plot"] [-n 20] [-d 0.1] \
  [--large-file-threshold-mb 1024] [--chunk-size 500000] \
  [--rank1-cache-dir ./rank1-cache]
```

| Argument | Description |
|---|---|
| `-f` | Score file(s) to plot (repeatable, required). |
| `-l` | Custom labels (optional). |
| `-o` | Output PNG file (required). |
| `-t` | Plot title (default: `"DIR curves"`). |
| `-n` | Number of resampling iterations (default: `20`). |
| `-d` | Gallery decimation proportion (default: `0.1`). |
| `--large-file-threshold-mb` | Stream score files at least this large instead of loading the full CSV in memory (default: `1024`; use `0` to stream every file). |
| `--chunk-size` | Number of CSV rows read per chunk in streaming mode (default: `500000`). |
| `--rank1-cache-dir` | Optional directory for reusable rank-1 summaries generated in streaming mode. |

The plot shows Detection and Identification Rate (DIR) curves: TPIR vs FPIR on a log scale, with reference lines at FPIR=1% and TPIR=95%.

## Experiment Configuration

The identification pipeline uses the [common fields](experiment_config.md#common-fields) plus the [identification fields](experiment_config.md#identification-fields) (`protocols`, `splits`).

See [Protocol Format and Custom Databases](protocols.md#identification) for the
enrollment/probe directory layout and CSV requirements.

For BTP algorithm configuration, see [experiment_config.md — BTP Algorithms](experiment_config.md#btp-algorithms).
