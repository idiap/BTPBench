# Key Selection

The key-selection commands select a separate key for each user, explore the
key space on synthetic vectors, or validate sampled system keys against
recognition protocols. System-specific inversion resistance is evaluated by
the [irreversibility pipeline](irreversibility.md), which supports configured
keys, key-explorer buckets, and random key distributions.

## User-Specific Key Selection

Selects a suitable key for each subject by testing randomly generated candidate
keys against an inversion-resistance threshold. The threshold can be derived
from a verification score file, supplied directly, or set to `0.0` by
`--random` for a lenient random-key baseline.

```bash
uv run btpbench keyselection pipeline_user \
  -s config/system_config.yaml \
  -e config/experiment_config.yaml \
  -v verification_scores.csv \
  -f 0.01 \
  [-o ./output_dir] [--override] [--random]
```

| Argument | Description |
|---|---|
| `-s` | Path to the system configuration file (required). |
| `-e` | Path to the experiment configuration file (required). |
| `-v`, `--verification-file` | Verification score file used to compute an FMR threshold. Required unless `-t` or `--random` is used. |
| `-f` | Target FMR used to derive the inversion-resistance threshold (default: `0.01`). |
| `-t` | Manual threshold value instead of an FMR-derived threshold (optional). |
| `-o` | Output directory (optional, overrides value from experiment config). |
| `--override` | Overwrite existing results. |
| `--random` | Use threshold `0.0` for a lenient random-key baseline. Candidate keys are still generated and evaluated by the normal key-selection routine. |

### Workflow

1. Extract features from the verification samples and keep one template per
   subject for selection.
2. Compute the reference feature distribution used by template inversion.
3. Derive the decision threshold from `-v` and `-f`, use the manual `-t`
   threshold, or use `0.0` with `--random`.
4. For each user-specific BTP algorithm and subject, repeatedly draw a random
   integer candidate key, protect and invert the template, and compare the
   inversion with the original. A finite inversion score must be strictly below
   the threshold for the candidate to qualify; with `multiple_guesses`, every
   generated guess must satisfy this test. A failed inversion does not reject
   the candidate.
5. Store the selected subject-to-key mapping in a JSON file.

The configured BTP must have `system_specific: false`. There is no
entropy-based selector in this workflow: candidate keys are random in every
mode, and `--random` changes only the threshold and output tag.

### Output

- **Key file:** `keys_select_<fmr_tag>-<ks_tag>-<inversion_tag>-<database>-<algorithm>-<baseline>.json`
  Contains a mapping of subject IDs to selected keys: `{"subject_id": key, ...}`.
- **Selection diagnostic scores:** `ref_keys_select_<fmr_tag>-<ks_tag>-<inversion_tag>-<database>-<algorithm>-<baseline>.csv`

Where `<fmr_tag>` is `fmr<int(fmr * 10000)>` (for example, `fmr100`
for `0.01`), `random` when using `--random`, or
`thresh<int(threshold * 10000)>` when using `-t` (for example,
`thresh-7000` for `-t -0.7`).

### Evaluate the selected keys

Key selection does not constitute an independent security evaluation. To
measure inversion resistance, use the generated JSON file with the
[irreversibility pipeline](irreversibility.md) and an independent attack-target
protocol:

```yaml
sampling_mode: configured
n_attack_trials: 10
attack_seed: 42

btps:
  algs:
    - type: polyprotect
      system_specific: false
      key_dictionary_file: ./results/keys_select_....json
      # Use the same BTP parameters as during key selection.
```

Every subject in `for_irreversibility.csv` must occur in the selected-key
dictionary. Use normal verification or identification pipelines with the same
dictionary to evaluate recognition utility separately.

## Key Explorer

The key explorer finds inversion-resistant keys for random vectors across
multiple thresholds. Unlike the data-backed workflows, it generates synthetic
vectors and searches for keys whose inverted templates have a negative
cosine-distance score at or below the configured threshold. Lower scores
represent larger cosine distance from the original vector in this command.

This is useful for studying the key space of a BTP algorithm independently of any specific dataset.

```bash
uv run btpbench keyselection key_explorer \
  -e config/experiment_config.yaml \
  -t -0.7 -t -0.8 -t -0.9 \
  [-o ./output_dir] \
  [--time-limit 172800] \
  [--batch-size 100] \
  [--vector-dim 512] \
  [--n-dist-samples 1000] \
  [--bad] \
  [--override]
```

| Argument | Description |
|---|---|
| `-e` | Path to the experiment configuration file (required). |
| `-o` | Output directory (optional, overrides value from experiment config). |
| `-t` | Threshold value(s). Can be specified multiple times (e.g. `-t -0.75 -t -1.0`). Required. |
| `--time-limit` | Time limit in seconds (default: `172800`, i.e. 48 hours). |
| `--batch-size` | Number of random vectors per batch (default: `100`). |
| `--vector-dim` | Dimensionality of random vectors (default: `512`). |
| `--n-dist-samples` | Number of random samples for computing the reference distribution (default: `1000`). |
| `--bad` | Search for bad keys instead: key selection uses threshold `0`, and keys whose scores are greater than or equal to each requested threshold are bucketed. These higher scores correspond to inversions closer to the original vector. The output filename is prefixed with `bad_`. |
| `--override` | Overwrite existing output files. |

### Workflow

1. Parse `output_dir`, `compliant`, `num_processes`, and `btps` from the
   experiment configuration; no system configuration, database, baseline, or
   detector is used.
2. For each BTP algorithm:
   a. Generate random vectors to compute a reference distribution.
   b. Open corresponding output file.
   c. Loop in batches until the time limit:
      - Generate a batch of random vectors.
      - For each vector, use parallel key selection (with the most lenient threshold) to find a key.
      - In normal mode, place scores at or below each threshold into that bucket; in `--bad` mode, use the reverse comparison.
      - Enforce global key uniqueness — duplicate keys are discarded.
      - Place each found key into all qualifying threshold buckets.
      - Periodically save results to disk (crash resilience).
3. Save final results.

Every top-level BTP entry used by the key explorer must set
`system_specific: false`, because the explorer invokes the user-specific
candidate-key search. For a `combined` wrapper, this flag belongs on the outer
entry.

### Output

- **Result file:** `key_explorer_<thresholds>-<ks_tag>-<inversion_tag>-<algorithm>.json`

The JSON file contains:

```json
{
  "metadata": {
    "thresholds": [-0.7, -0.8, -0.9],
    "vectors_processed": 5000,
    "elapsed_seconds": 172200,
    "keys_per_threshold": {"-0.7": 2, "-0.8": 1, "-0.9": 0}
  },
  "keys": {
    "-0.7": {"0": 12345, "1": 67890},
    "-0.8": {"0": 12345},
    "-0.9": {}
  }
}
```

Each threshold bucket maps vector IDs to the unique key found for that vector.

## System-Key Validation

Two commands validate sampled system keys without running key selection again:

```bash
uv run btpbench keyselection validate_sys \
  -s config/system_config.yaml \
  -e config/experiment_config.yaml \
  [-o ./output_dir] [--override]

uv run btpbench keyselection validate_sys_verification \
  -s config/system_config.yaml \
  -e config/experiment_config.yaml \
  [-o ./output_dir] [--override]
```

`validate_sys` evaluates identification protocols and therefore requires `protocols` and `splits`. `validate_sys_verification` evaluates the database's verification sample set. Both use `sampling_mode`, `n_keys`, and, for key-file sampling, `keys_file` and `keys_bucket` from the experiment configuration.
Both commands also honor `key_sampling_seed`, so they evaluate the same sampled
key systems as the irreversibility pipeline when given the same configuration.

They write one score CSV per BTP configuration:

- `validate_sys`: `sys_keys_identification-<database>-<algorithm>-<baseline>-<n_keys><keys_or_distribution_tag>.csv`
- `validate_sys_verification`: `sys_keys_verification-<database>-<algorithm>-<baseline>-<n_keys><keys_or_distribution_tag>.csv`

## Experiment Configuration

The user-specific pipeline uses the
[common fields](experiment_config.md#common-fields) plus the
[key-selection fields](experiment_config.md#key-selection-fields).
System-key validation requires `n_keys` and, in key-file mode, `keys_file` and
`keys_bucket`. The key explorer uses only `output_dir`, `compliant`,
`num_processes`, and `btps`. For system-key inversion evaluation, use the
[irreversibility configuration](experiment_config.md#irreversibility-key-fields).

BTP algorithms should be configured with
[key-selection parameters](experiment_config.md#key-selection-parameters)
(`ks_method`, and a positive `ks_n_elements` when using
`multiple_guesses`) and
[inversion parameters](experiment_config.md#inversion-parameters) (`method`,
`num_guesses`, `precision`).

For full configuration reference and examples, see [experiment_config.md](experiment_config.md).
