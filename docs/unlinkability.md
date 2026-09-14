# Unlinkability Experiment

The unlinkability pipeline evaluates whether protected templates produced from
the same subject can be linked when each sample is protected with a different
key. It produces two protected score files per BTP configuration:

- **Mated scores:** comparisons between protected templates from the same subject.
- **Non-mated scores:** comparisons between protected templates from different subjects.

The two score files can then be used to analyze how separable same-subject and
different-subject protected templates are under the selected key bucket.

## Running the Pipeline

```bash
uv run btpbench unlinkability pipeline \
  -s config/system_config.yaml \
  -e config/experiment_config.yaml \
  -k keys.json \
  -b "-0.9" \
  [--samples-per-subject 60] \
  [--non-mated-samples-per-subject 10] \
  [--seed 42] \
  [-o ./output_dir] [--override]
```

| Argument | Description |
|---|---|
| `-s` | Path to the system configuration file (required). |
| `-e` | Path to the experiment configuration file (required). |
| `-k` | Path to a [key-explorer](key_selection.md#key-explorer) JSON file containing a top-level `keys` mapping (required). |
| `-b` | Bucket name to use from the keys file, e.g. `"-0.9"` (required). |
| `--samples-per-subject` | Number of unlinkability samples and key systems to keep per subject (default: `60`). A simple BTP needs this many bucket keys. A combined BTP constructs each system by sampling `nb_algs` distinct bucket keys and may reuse key values between systems. |
| `--non-mated-samples-per-subject` | Number of protected templates sampled per subject for the non-mated score file (default: `10`). This value must be lower than or equal to `--samples-per-subject`. |
| `--seed` | Optional random seed for reproducible wrapper key-system construction and non-mated template sampling. |
| `-o` | Output directory (optional, overrides value from experiment config). |
| `--override` | Overwrite existing score files. |

## Pipeline Workflow

1. Load system and experiment configuration.
2. Instantiate the dataset and load samples from the database `unlink_samples`
   file by calling `dataset.samples(verification=False)`.
3. Load keys from the requested bucket in the key-explorer JSON file.
4. Build `--samples-per-subject` key systems. For a simple BTP these are the first requested bucket keys. For a combined BTP, each system contains `nb_algs` distinct values sampled from the bucket.
5. Keep the first `--samples-per-subject` unlinkability samples for each
   subject. The pipeline raises an error if any subject has fewer samples.
6. Pair the selected samples with the constructed key systems.
7. Extract unprotected biometric features for all selected samples.
8. For each BTP configuration:
   - Protect sample `i` from every subject with key system `i`, so each selected
     sample of a subject receives a different scalar key or wrapper key list.
   - Compute all same-subject pairwise comparisons. These are the mated scores.
   - Randomly sample `--non-mated-samples-per-subject` protected templates per
     subject.
   - Compute all cross-subject comparisons between the sampled protected
     templates. These are the non-mated scores.

For `S` subjects, `N = --samples-per-subject`, and
`M = --non-mated-samples-per-subject`, each BTP configuration produces:

| Score type | Number of comparisons |
|---|---|
| Mated | `S * N * (N - 1) / 2` |
| Non-mated | `S * (S - 1) / 2 * M * M` |

These are template-pair counts and CSV row counts because every supported BTP
comparison returns one scalar score.

## Output

For each BTP configuration, two score files are produced:

```text
unlinkability-<database>-<btp_config>-<baseline>-keys<bucket_tag>-<samples_per_subject>-mated.csv
unlinkability-<database>-<btp_config>-<baseline>-keys<bucket_tag>-<samples_per_subject>-non-mated-<non_mated_samples_per_subject>.csv
```

The `bucket_tag` is derived from the bucket name by replacing `.` with `d` and
`-` with `m`. For example, bucket `"-0.9"` becomes `m0d9`.

The score CSVs use the standard score format:

| Column | Description |
|---|---|
| `probe_template_id` | Template ID of the probe template. |
| `probe_subject_id` | Subject ID of the probe template. |
| `bio_ref_template_id` | Template ID of the reference template. |
| `bio_ref_subject_id` | Subject ID of the reference template. |
| `score` | Protected comparison score. |
| `bio_ref_<metadata>` | Metadata columns copied from the reference template. |
| `probe_<metadata>` | Metadata columns copied from the probe template. |

The mated file contains only same-subject pairs. The non-mated file contains
only different-subject pairs.

## Plotting

The plotting command visualizes the two unlinkability score files as probability
density curves and overlays the local unlinkability metric `D(s)` on a secondary
y-axis. It also reports the global score `Dsys` in the figure title.

```bash
uv run btpbench unlinkability plots \
  -m unlinkability-<database>-<btp_config>-<baseline>-keys<bucket_tag>-<samples_per_subject>-mated.csv \
  -n unlinkability-<database>-<btp_config>-<baseline>-keys<bucket_tag>-<samples_per_subject>-non-mated-<non_mated_samples_per_subject>.csv \
  [--optimized-mated-score-file diversity.csv] \
  -t "Facenet (overlap = 2)" \
  -o unlinkability.png
```

| Argument | Description |
|---|---|
| `-m` | Mated unlinkability score CSV produced by the pipeline (required). |
| `-n` | Non-mated unlinkability score CSV produced by the pipeline (required). |
| `--optimized-mated-score-file` | Optional diversity CSV containing exactly one criterion's `*_clique_keys` and `*_clique_scores` columns. It overlays the maximum-clique mated-score density. Legacy `selected_keys`/`selected_key_scores` CSVs are also accepted. When either score bound is automatic, these scores also participate in choosing the metric range. |
| `-t` | Plot title (required). |
| `-o` | Output file (required). Use `.html` or `.htm` for an interactive Plotly figure; other extensions produce a static Matplotlib figure. |
| `--mated-label` | Legend label for the mated distribution (default: `Mated`). |
| `--non-mated-label` | Legend label for the non-mated distribution (default: `Non-Mated`). |
| `--bins` | Number of histogram bins used for density estimation (default: `512`; minimum: `10`). |
| `--smooth-sigma` | Non-negative Gaussian smoothing sigma in histogram-bin units (default: `2.0`). |
| `--metric-bins` | Number of bins used to compute `D(s)` and `Dsys` (default: `10`; minimum: `2`). |
| `--omega` | Positive prior ratio used in the unlinkability metric (default: `1.0`). |
| `--d-local-threshold` | Mark contiguous regions where `D(s)` is at or below this value (default: `0.0`; range: `[0, 1]`). Alias: `--d-local-match-value`. |
| `--x-min` | Optional lower bound for the histogram and unlinkability-metric bins; changing it can change `D(s)` and `Dsys`. |
| `--x-max` | Optional upper bound for the histogram and unlinkability-metric bins; changing it can change `D(s)` and `Dsys`. |

The optimized overlay is obtained from the clique-based diversity pipeline; it is not generated by the unlinkability score pipeline itself. Run diversity with `--real-sample-set unlinkability` and one unlinkability-derived criterion, then pass the resulting diversity CSV here. The `AVERAGE` row is ignored because its clique key and score cells are blank. If either `--x-min` or `--x-max` is omitted, optimized scores can expand the automatically selected metric range and thereby change the reported `D(s)` and `Dsys`, in addition to adding the overlay.

`D(s)` follows the local unlinkability metric from Gomez-Barrero et al. For each
score bin `s`, let `p_mated(s)` be the mated score density and
`p_non_mated(s)` be the non-mated score density. With prior ratio `omega`, the
likelihood ratio is:

```text
LR(s) = p_mated(s) / p_non_mated(s)
```

The local metric is:

```text
D(s) = 2 * (omega * LR(s) / (1 + omega * LR(s))) - 1
```

and it is clipped to `0` wherever `omega * LR(s) <= 1`. If the non-mated density
is zero in a bin, `D(s)` is set to `1`. Intuitively, `D(s) = 0` means that the
score bin does not provide mated-linking evidence under this metric, while values
closer to `1` indicate stronger local linkability.

`Dsys` is the mated-density-weighted integral of `D(s)`:

```text
Dsys = integral D(s) * p_mated(s) ds
```

It summarizes the local curve into one value in `[0, 1]`. Lower values indicate
better unlinkability; higher values indicate that mated and non-mated protected
scores are easier to distinguish.

Example:

```bash
uv run btpbench unlinkability plots \
  -m work_dir/unlinkability_results/unlinkability-icarb-normalized_polyprotect_usr_3_5_50-edgeface-keysm0d9-60-mated.csv \
  -n work_dir/unlinkability_results/unlinkability-icarb-normalized_polyprotect_usr_3_5_50-edgeface-keysm0d9-60-non-mated-10.csv \
  -t "EdgeFace (overlap = 3)" \
  -o work_dir/unlinkability_results/icarb_edgeface_unlinkability.png
```

## Protocol Requirements

The selected database must define an `unlink_samples` CSV file in the system
configuration. See
[Protocol Format and Custom Databases](protocols.md#unlinkability) for the
shared schema and protocol-authoring workflow.

```yaml
databases:
  icarb:
    proto_dir: ./protocols/iCarB-Face/
    verification_samples: ./protocols/iCarB-Face/icarb-verification.csv
    unlink_samples: ./protocols/iCarB-Face/icarb-unlink.csv
    dataset_dir: /path/to/iCarB-Face/
    extension: .avi
    video: true
```

The unlinkability CSV must contain the standard sample columns:

```csv
path,subject_id,template_id,...
```

Additional columns are treated as metadata and are copied into the score files.
For video databases, `frame_idx` selects a position as a percentage of the video
length: `0` reads the first frame, while a positive value such as `30` seeks to
approximately 30% of the video.

`template_id` should uniquely identify each selected sample. This is especially
important when the same video is repeated with different `frame_idx` values,
because feature caching uses the subject and template IDs. A practical format is
to append the requested frame percentage to the template ID:

```csv
path,subject_id,template_id,variation_id,env,accessory_id,frame_idx
1/1_1_0_1,1,1_1_0_1_f0,1,indoor,0,0
1/1_1_0_1,1,1_1_0_1_f30,1,indoor,0,30
```

## Example

```bash
uv run btpbench unlinkability pipeline \
  -s config/system_config.yaml \
  -e config/experiment_config.yaml \
  -k work_dir/key_explorer_results.json \
  -b "-0.9" \
  --samples-per-subject 60 \
  --non-mated-samples-per-subject 10 \
  --seed 42 \
  -o work_dir/unlinkability_results/
```

With 200 subjects, this produces per BTP configuration:

- `200 * 60 * 59 / 2 = 354000` mated comparisons.
- `200 * 199 / 2 * 10 * 10 = 1990000` non-mated comparisons.

## Experiment Configuration

The unlinkability pipeline uses the [common fields](experiment_config.md#common-fields)
and requires at least one BTP algorithm in
[experiment_config.md - BTP Algorithms](experiment_config.md#btp-algorithms).
It does not use `protocols` or `splits`; it operates on the `unlink_samples`
file defined in the system configuration for the selected database.
