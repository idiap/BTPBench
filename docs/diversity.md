# Diversity

The diversity pipeline models each subject's protected templates as an undirected graph. Each protected template is a node, and two nodes are connected when their comparison qualifies as non-matching under an FMR-derived threshold or an inclusive score range. The result is an exact maximum clique: the largest set of protected templates in which every pair is non-matching.

In the default verification flow, the nodes come from one subject template protected by each key in the bucket. A larger maximum clique means more keys can be used together without producing colliding protected templates, which is desirable for template renewability.

## Pipeline

```bash
uv run btpbench diversity pipeline \
  -s config/system_config.yaml \
  -e config/experiment_config.yaml \
  -k keys.json \
  -b "-0.9" \
  [-p protected_scores.csv] \
  [-f 0.0001] [-f 0.001] [-f 0.01] \
  [--score-range MIN MAX] \
  [--unlinkability-mated-score-file mated.csv] \
  [--unlinkability-non-mated-score-file non-mated.csv] \
  [--metric-bins 10] [--omega 1.0] \
  [--d-local-threshold 0.0] \
  [--x-min -2.0] [--x-max 0.0] \
  [--seed 42] \
  [--random-vectors 0] [--vector-dim 512] \
  [--real-sample-set verification|unlinkability] \
  [--n-subjects -1] \
  [-o ./output_dir] [--override]
```

| Argument | Description |
|---|---|
| `-s` | Path to the system configuration file (required). |
| `-e` | Path to the experiment configuration file (required). |
| `-k` | Path to a [key-explorer](key_selection.md#key-explorer) JSON file containing a top-level `keys` mapping (required). |
| `-b` | Bucket name to use from the keys file, e.g. `"-0.9"` (required). |
| `-p` | Path to a protected score file used to compute FMR thresholds. Required when FMR thresholds are used. |
| `-f` | FMR value(s) for matching thresholds. Can be specified multiple times. When no FMR, manual score range, or unlinkability score pair is passed, the default FMRs are `0.0001`, `0.001`, and `0.01` (0.01%, 0.1%, 1%). |
| `--score-range MIN MAX` | Inclusive accepted score range. A comparison qualifies when `MIN <= score <= MAX`. Repeating it adds alternative accepted ranges, so a score qualifies when it lies in any provided range. Can be combined with `-f`. |
| `--unlinkability-mated-score-file` | Mated unlinkability score CSV used with the non-mated CSV to derive accepted `D(s) <= C` score ranges. |
| `--unlinkability-non-mated-score-file` | Non-mated unlinkability score CSV used with the mated CSV. The two unlinkability score files must be passed together. |
| `--metric-bins` | Number of bins used to compute unlinkability `D(s)` score ranges (default: `10`; minimum: `2`). |
| `--omega` | Positive prior ratio used in the unlinkability metric (default: `1.0`). |
| `--d-local-threshold` | Local threshold `C`; contiguous `D(s) <= C` regions become accepted diversity score ranges (default: `0.0`; range: `[0, 1]`). |
| `--x-min`, `--x-max` | Optional score limits used to construct unlinkability metric bins. Pass the same values as the unlinkability plot command to derive identical regions. |
| `--seed` | Optional random seed used for random input vectors and for constructing combined key systems. |
| `--random-vectors` | If `> 0`, skip dataset loading / feature extraction and use this many synthetic subjects whose templates are uniform random vectors in `[-1, 1]` (default: `0`, i.e. real features). |
| `--vector-dim` | Dimensionality of the random input vectors (default: `512`, only used when `--random-vectors > 0`). |
| `--real-sample-set` | Real-template sample set to use: `verification` keeps the current one-template-per-subject diversity flow, while `unlinkability` protects different same-subject unlinkability templates with different keys. Cannot be combined with random vectors. |
| `--n-subjects` | Process only the first `N` subjects in dataset order. Use `-1` to process every subject (default: `-1`). The limit is applied before feature extraction, protection, pairwise scoring, and clique search. |
| `-o` | Output directory (optional, overrides value from experiment config). |
| `--override` | Overwrite existing results. |

### Workflow

1. Load system and experiment configuration.
2. Load keys from the specified bucket in a key-explorer JSON file. The system-specific key-evaluation pipeline consumes this format but does not produce a key JSON file itself.
3. Build the threshold criteria: compute requested FMR thresholds from a protected score file using `bob.measure.far_threshold`, combine manually requested inclusive score ranges into one accepted union, and/or derive an accepted union from contiguous `D(s) <= C` regions in unlinkability mated and non-mated score files. The derived path uses the same metric calculation as `btpbench unlinkability plots`.
4. Build the per-subject template set:
   - **Verification real features** (default): extract features from the verification samples and keep one template per subject.
   - **Unlinkability real features** (`--real-sample-set unlinkability`): extract all unlinkability samples. For each subject, protect template index `i` with key index `i` instead of protecting one template with every bucket key. The selected key/template count is capped by whichever is smaller: the bucket size or the highest number of usable unlinkability templates available for a subject.
   - **Random vectors** (`--random-vectors N`): skip dataset loading and feature extraction; generate `N` synthetic subjects whose templates are uniform random vectors of dimension `--vector-dim` in `[-1, 1]`. Useful to check that the diversity of a BTP scheme is intrinsic to the scheme/keys and not driven by the input feature distribution. The output `database` tag becomes `random<N>d<dim>`.
5. For each BTP algorithm and each subject:
   a. Protect the subject template set with its selected keys.
   b. Compute all pairwise comparison scores between the subject's protected templates (`n * (n - 1) / 2` comparisons for that subject's actual node count `n`). In unlinkability mode, subjects can have different node counts, and failed protections are omitted.
   c. For each criterion, create an edge for every qualifying comparison: scores strictly below an FMR threshold or scores inside an inclusive score range.
   d. Use NetworkX's exact branch-and-bound maximum-weight-clique algorithm with unit node weights to find one maximum clique.
   e. Store the clique size, its keys, and the already-computed pairwise scores inside the clique. Keys and scores are compact JSON fields.

Subjects are analyzed independently in a process pool using the experiment configuration's `num_processes` value. Each worker creates one BTP algorithm instance, then performs both pairwise scoring and maximum-clique search for its assigned subjects. Output row order remains the original subject order.

Finding an exact maximum clique is NP-hard. NetworkX provides the implementation, but runtime can still grow quickly for large or difficult graphs.

The clique implementation requires one scalar score per comparison, which every
supported BTP provides.

### Output

- **Diversity CSV:** `diversity-<database>-<algorithm>-<baseline>-keys<bucket_tag>-<n_keys>.csv`

  Unlinkability real-template runs add `-unlinkability` after `<database>` so they do not overwrite a verification run with the same selected key count.

  Limited-subject runs add `-subjects<N>` after the database/sample-set tag so they do not overwrite an all-subject run.

  When score ranges are derived from unlinkability score files, the output also
  adds `-threshold<threshold_tag>` before `.csv`, for example
  `-threshold0d5.csv`. The bucket and local decision threshold are file-name
  metadata only; they are not added as CSV columns.

  Contains one row per subject followed by an `AVERAGE` row with the following columns:

  | Column | Description |
  |---|---|
  | `subject_id` | Subject identifier, or `AVERAGE` for the final summary row. |
  | `<criterion>_clique_size` | Size of the exact maximum clique for this criterion. The `AVERAGE` row contains the mean across subjects. Criterion tags include `fmr_<value>`, `score_range_<min>_<max>`, and the multi-range form `score_ranges_<min1>_<max1>__<min2>_<max2>`. |
  | `<criterion>_clique_keys` | Keys corresponding to the reported maximum-clique nodes, serialized as a compact JSON array in one CSV cell, for example `"[10,11,12]"`. This cell is blank in the `AVERAGE` row. |
  | `<criterion>_clique_scores` | Pairwise scores between nodes in the reported maximum clique, in upper-triangle order and serialized as compact JSON. This supplies the optional optimized-mated distribution used by unlinkability plots. This cell is blank in the `AVERAGE` row. |

`btpbench unlinkability plots --optimized-mated-score-file` accepts a diversity CSV when it contains exactly one `*_clique_scores` criterion. This is the normal shape of a diversity run derived from one pair of unlinkability score files. Diversity CSVs containing several criteria are ambiguous and are rejected by that option.

## Plotting a Diversity Result

The `plots` command creates violin plots for each detected criterion in a
supported per-subject diversity schema (`*_clique_size`, or one of the legacy
schemas when no clique-size columns exist). The pipeline's final `AVERAGE` row
is excluded; the plot shows each criterion's distribution, range, median, and
mean.

```bash
uv run btpbench diversity plots \
  -f work_dir/diversity_results/diversity-result.csv \
  -o work_dir/diversity_results/diversity-violin.png \
  [-t "Diversity distribution across subjects"]
```

| Argument | Description |
|---|---|
| `-f`, `--result-file` | One diversity result CSV (required). |
| `-o`, `--output-file` | Output image filename (required). |
| `-t`, `--title` | Plot title (default: `Diversity distribution across subjects`). |

## Summarizing FMR-Based Results

The `summary` command searches one or more experiment/result directories for
FMR-based diversity CSVs and reports the number of subjects plus the mean,
minimum, and maximum diversity. It currently recognizes BioHash, PolyProtect,
and combined PolyProtect filename formats.

```bash
uv run btpbench diversity summary \
  -i work_dir/run1 [-i work_dir/run2 ...] \
  -o work_dir/diversity-summary.csv \
  [-f 0.01 -f 0.001]
```

`-i` is repeatable, and `-f` selects repeatable FMR fractions (defaults: `0.01`
and `0.001`). The output argument is a prefix: the command writes
`diversity-summary-biohash.csv`, `diversity-summary-polyprotect.csv`, and
`diversity-summary-combined-polyprotect.csv`.

## Summarizing Unlinkability-Derived Results

Use `unlinkability-summary` for diversity runs produced with
`--real-sample-set unlinkability` and unlinkability-derived score ranges. It
groups results by algorithm family, key bucket, local threshold, and score-range
criterion.

```bash
uv run btpbench diversity unlinkability-summary \
  -i work_dir/run1 [-i work_dir/run2 ...] \
  -o work_dir/unlinkability-diversity-summary.csv
```

Like `summary`, `-i` is repeatable and `-o` is a prefix used to create one CSV
for each supported algorithm family.

## Pipeline Example

```bash
uv run btpbench diversity pipeline \
  -s config/system_config.yaml \
  -e config/experiment_config.yaml \
  -k work_dir/key_explorer_results.json \
  -b "-0.9" \
  -p work_dir/protected_verification_scores.csv \
  -f 0.0001 -f 0.001 -f 0.01 \
  --seed 42 \
  -o work_dir/diversity_results/
```

This will produce a CSV like:

```
subject_id,fmr_0.01_clique_size,fmr_0.01_clique_keys,fmr_0.01_clique_scores
001,3,"[10,11,12]","[-1.2,-1.1,-1.0]"
002,2,"[20,22]",[-0.9]
AVERAGE,2.5,,
```
