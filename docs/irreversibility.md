# Irreversibility Experiment

The irreversibility pipeline evaluates how resistant protected templates are to
inversion attacks. It attempts to reconstruct each original biometric template
from its protected form and measures whether the reconstruction would be
accepted by the unprotected biometric system.

The attack receives the key stored with the protected template. Results
therefore represent a conservative **key-disclosure** threat model rather than
security obtained by keeping the key secret.

## Running the Pipeline

```bash
uv run btpbench irreversibility pipeline \
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

1. Load background samples from
   `<proto_dir>/irreversibility/for_distribution.csv` and attack targets from
   `<proto_dir>/irreversibility/for_irreversibility.csv`.
2. Extract features from both sets and use only the background templates to
   estimate the distribution from which inversion guesses are drawn.
3. Resolve one or more key systems from the configured key source.
4. For each BTP algorithm, key system, and attack target:
   - protect the target once;
   - perform `n_attack_trials` complete inversion attempts;
   - compare every reconstruction only with its corresponding original target;
   - write the score and attack provenance.

The pipeline writes raw scores only. Solution, match, and inversion-success
rates are computed later by the `irreversibility metrics` command.

`n_attack_trials` and the BTP's `num_guesses` serve different purposes:

- `n_attack_trials` repeats the entire attack to estimate its success
  probability;
- `num_guesses` is the number of starting guesses attempted inside one attack.

Trial randomness is derived from `attack_seed`, the trial number, target, and
key. Runs with the same inputs and seeds are reproducible, while different
trials receive different random streams.

### Key sources

| `sampling_mode` | Required fields | Behavior |
|---|---|---|
| omitted or `configured` | BTP key configuration | A system-specific BTP uses its configured `key`. A user-specific BTP uses `key_dictionary_file`, or the legacy subject-ID key when no dictionary is configured. One key system is evaluated. |
| `keys` | `n_keys`, `keys_file`, `keys_bucket` | Samples `n_keys` key systems from a key-explorer bucket. System-specific keys are shared by all targets; user-specific keys are assigned per subject. |
| `distribution` | `n_keys` | Samples `n_keys` integer key systems from `[0, 1000000)`. System-specific keys are shared; user-specific keys are sampled per subject. |

`key_sampling_seed` controls bucket/distribution sampling and defaults to `42`.
For a combined BTP, one key system contains the required number of component
keys automatically.

The key-explorer `keys_bucket` selects key seeds. A BTP's `dist_tag` instead
selects a PolyProtect coefficient distribution; it is not a key source.

### Output

Score files are saved as:

```text
irreversibility-<key-source>-systems<n>-trials<n>-<inversion-config>-<database>-<algorithm>-<baseline>.csv
```

In addition to the standard score columns and protocol metadata, every row
contains prefixed copies of:

| Field | Meaning |
|---|---|
| `attack_trial` | Zero-based attack-trial index. |
| `attack_solved` | Whether inversion produced a reconstructed template. |
| `key` | Scalar or component-key list used for this target. |
| `key_system` | Zero-based sampled key-system index. |
| `key_source` | `configured`, `subject-id`, `dictionary:...`, `bucket:...`, or `random-distribution`. |
| `key_scope` | `user` or `system`. |

## Computing Metrics

Irreversibility metrics require both an **identification score file** (to compute FAR thresholds) and one or more **irreversibility score files**.

```bash
uv run btpbench irreversibility metrics \
  -s id_scores.csv \
  -i irr_scores1.csv [-i irr_scores2.csv ...] \
  -l "Label1" [-l "Label2" ...] \
  -o irr_metrics.csv \
  [-f 0.01 -f 0.001]
```

| Argument | Description |
|---|---|
| `-s` | Identification score file, used to compute FAR thresholds (required). |
| `-i` | Irreversibility score file(s) (repeatable, required). |
| `-l` | Label for each irreversibility file (required; the number of labels must match the number of `-i` arguments). |
| `-o` | Output CSV file (required). |
| `-f` | FAR values at which to compute match/success rates (repeatable, optional). Without it, the output contains the solution rate only. |

### Metrics Produced

| Metric | Description |
|---|---|
| Solution rate | Solved attacks divided by all target/key/trial attempts. |
| Match rate | Accepted reconstructions divided by solved attacks. |
| Inversion success rate | Accepted reconstructions divided by all attack attempts. |
| TMR | Template Match Rate at specified FAR values. |

Thresholds are derived from the identification score file and applied to each irreversibility score file.
Rates in the metrics CSV are stored as fractions in `[0, 1]`; the plotting
command displays inversion-success rates as percentages.

## Generating Plots

```bash
uv run btpbench irreversibility plots \
  -s id_scores.csv \
  -i irr_scores1.csv [-i irr_scores2.csv ...] \
  -l "Label1" [-l "Label2" ...] \
  -o irr_plot.png \
  [-t "My Plot"] [-x 0.1]
```

| Argument | Description |
|---|---|
| `-s` | Identification score file (required). |
| `-i` | Irreversibility score file(s) (repeatable, required). |
| `-l` | Label(s) for each file (required, must match number of `-i`). |
| `-o` | Output PNG file (required). |
| `-t` | Plot title (default: `"Inversion success rate vs FMR"`). |
| `-x` | FMR x-axis limit (default: `0.1`, i.e., 10%). |

The plot shows inversion success rate vs FMR on a logarithmic x-axis, with one
curve for each irreversibility score file.

## Experiment Configuration

The irreversibility pipeline uses the
[common fields](experiment_config.md#common-fields) and requires at least one
BTP algorithm. It ignores `protocols` and `splits`; instead, the selected
database's `proto_dir` must contain the two fixed irreversibility input files
listed in the workflow above. Inversion behavior can be customized with
[inversion parameters](experiment_config.md#inversion-parameters), whose
documented defaults are used when they are omitted.

A minimal configured-key experiment adds:

```yaml
n_attack_trials: 10
attack_seed: 42

btps:
  algs:
    - type: polyprotect
      system_specific: true
      key: 39
```

To evaluate user-specific keys produced by `keyselection pipeline_user`, keep
the same BTP parameters used during selection and configure its JSON output as
the key dictionary:

```yaml
n_attack_trials: 10
attack_seed: 42

btps:
  algs:
    - type: polyprotect
      system_specific: false
      key_dictionary_file: ./results/keys_select_....json
```

Every attack-target subject must have an entry in the dictionary. The
irreversibility targets should be independent from the templates used to
select those keys.

To evaluate a key-explorer bucket instead, add:

```yaml
n_attack_trials: 10
attack_seed: 42
sampling_mode: keys
n_keys: 100
keys_file: ./results/key_explorer/key_explorer-output.json
keys_bucket: "-0.9"
key_sampling_seed: 42
```

The former `n_validations` field remains accepted as a compatibility fallback,
but new configurations should use `n_attack_trials`.

See [Protocol Format and Custom Databases](protocols.md#irreversibility) for the
role and schema of the distribution and inversion sample sets.

For BTP algorithm configuration, see [experiment_config.md — BTP Algorithms](experiment_config.md#btp-algorithms).
