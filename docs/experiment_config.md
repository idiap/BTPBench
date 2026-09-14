# Experiment Configuration

The experiment configuration file is a YAML file that defines the parameters for a specific experiment run. Different experiments use different subsets of fields. This page documents all available fields, their types, defaults, and which experiments use them. Relative paths are resolved from the process's current working directory, not from the directory containing the YAML file.

## Common Fields

These fields are used by the dataset-backed pipelines: identification, verification, irreversibility, user/system key evaluation, diversity, unlinkability, and distribution. The key explorer is the exception described below.

```yaml
output_dir: ./results/soteria       # Where to save results
save: true                          # Save intermediate templates
compliant: true                     # Warnings (true) or errors (false)
num_processes: 10                   # Number of parallel workers
bio_alg: edgeface                   # Baseline algorithm
database: soteria                   # Dataset name
detector: mediapipe                 # Face detector
```

| Field | Type | Required | Description |
|---|---|---|---|
| `output_dir` | string | Yes | Output directory for results. |
| `save` | bool | Yes | Whether to save intermediate templates to disk. |
| `compliant` | bool | Yes | If `true`, missing/failed samples trigger warnings. If `false`, they trigger errors. |
| `num_processes` | int | Yes | Number of parallel worker processes. Set to roughly your number of CPU cores. |
| `bio_alg` | string | Yes | Baseline algorithm. Must be listed in the system config `baselines`. |
| `database` | string | Yes | Dataset name. Must be listed in the system config `databases`. |
| `detector` | string | Yes | Face detector: `mediapipe` or `mtcnn`. |

<a id="identification--irreversibility-fields"></a>

## Identification Fields

These fields are used by the **identification** pipeline only.

```yaml
protocols:
  - grandtest
splits: eval
```

| Field | Type | Required | Description |
|---|---|---|---|
| `protocols` | string, list of strings, or `"all"` | Yes | Evaluation protocol name(s). Use `all` to run every protocol discovered below the database's `proto_dir`. |
| `splits` | string, list of strings, or `"all"` | Yes | Split name(s) to run. Names come from the protocol-directory layout (commonly `dev` and `eval`); use `all` for every discovered split. |

The irreversibility pipeline ignores `protocols` and `splits`. It always loads
`irreversibility/for_distribution.csv` and
`irreversibility/for_irreversibility.csv` below the selected database's
`proto_dir` from the system configuration.

## Inversion Evaluation Fields

These fields control repeated attacks in the irreversibility pipeline.

```yaml
n_attack_trials: 10
attack_seed: 42
key_sampling_seed: 42
```

| Field | Type | Default | Description |
|---|---|---|---|
| `n_attack_trials` | int | `10` | Number of complete, independent inversion attempts for every target/key combination. Must be positive. |
| `attack_seed` | int or `null` | `42` | Base seed used to derive reproducible per-target/per-trial inversion randomness. Use `null` for nondeterministic attacks. |
| `key_sampling_seed` | int or `null` | `42` | Seed used to select bucket/distribution keys. Use `null` for nondeterministic selection. |

`num_guesses` in a BTP configuration is separate: it controls starting guesses
inside one attack trial. The legacy `n_validations` field remains a fallback
when `n_attack_trials` is absent.

## Irreversibility Key Fields

With no `sampling_mode`, irreversibility evaluates the keys configured on each
BTP. Set `sampling_mode` to evaluate sampled key systems instead:

```yaml
sampling_mode: keys
n_keys: 100
keys_file: ./results/key_explorer/key_explorer-output.json
keys_bucket: "-0.9"
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `sampling_mode` | string | No | `configured` | `configured` uses each BTP's system key or user dictionary; `keys` samples an explorer bucket; `distribution` samples integer seeds from `[0, 1000000)`. |
| `n_keys` | int | For sampled modes | `1` | Number of system-wide keys or user-specific key assignments to evaluate. |
| `keys_file` | string | When `sampling_mode: keys` | — | Key-explorer JSON file. |
| `keys_bucket` | string | When `sampling_mode: keys` | — | Bucket selected from `keys_file`, for example `"-0.9"`. |

## Key Selection Fields

These fields are used by the user-specific **key selection** pipeline.

### User-Specific (`keyselection pipeline_user`)

```yaml
key_sampling_seed: 42
```

| Field | Type | Default | Description |
|---|---|---|---|
| `key_sampling_seed` | int or `null` | `42` | Base seed for reproducible per-subject candidate-key generation. |

## BTP Algorithms

BTP algorithms are configured in the `btps.algs` list. Multiple algorithms can
be specified and are evaluated one by one.

Omitting the entire `btps` section is supported for unprotected-only
identification, verification, and distribution-plot runs. BTP-dependent
commands—including irreversibility, key selection, diversity, and
unlinkability—need at least one entry in `btps.algs` to produce their intended
results. Do not leave an empty `btps` mapping as a substitute for omitting it;
several commands treat the presence of the section as enabling protected
processing and then access `btps.algs`.

```yaml
btps:
  algs:
    - type: polyprotect
      overlap: 3
      nb_coef: 5
      coef_range: 50
    - type: biohash
      num_bits: 128
```

### Common BTP Parameters

These parameters are parsed by the common BTP base class. They are all optional
and have defaults, but the key-distribution fields affect only PolyProtect, as
described below.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `system_specific` | bool | `false` | If `true`, a single key is used for all subjects. If `false`, keys are per-subject. |
| `normalize_input` | bool | `true` | Normalize the input feature vector before protection. |
| `key` | int | `42` | System-wide key. Only used when `system_specific: true`. |
| `key_dictionary_file` | string | `null` | Path to a JSON file mapping subject IDs to keys. Only used when `system_specific: false`. |
| `key_distribution_file` | string | `null` | Optional coefficient-distribution JSON used by PolyProtect. Other BTP types may load the file but do not use it to construct their secret. |
| `dist_tag` | string | `all` | Top-level distribution selected from `key_distribution_file`; used by PolyProtect only. |

A PolyProtect key-distribution file uses the JSON shape emitted by
[`plots polyprotect_coeff --output-json`](plots.md#polyprotect-coefficient-plot).
The selected `dist_tag` must name a top-level object with entries `"1"` through
`"<nb_coef>"`; each entry contains equally sized coefficient values and sampling
probabilities:

```json
{
  "-0.7": {
    "1": {"values": [-3, 2], "probabilities": [0.4, 0.6]},
    "2": {"values": [-1, 4], "probabilities": [0.7, 0.3]},
    "3": {"values": [-2, 3], "probabilities": [0.5, 0.5]}
  }
}
```

For this example, configure `dist_tag: "-0.7"` and `nb_coef: 3`. When the
PolyProtect transform is inside a `combined` wrapper, put
`key_distribution_file` and `dist_tag` in `algs_config` so the inner transform
receives them.

For a `combined` wrapper, parameters that control key assignment—such as
`system_specific`, `key`, and
`key_dictionary_file`—belong on the outer wrapper entry. `algs_config`
contains the inner transform's parameters. An inner `system_specific` value can
still be supplied when its system/user tag is desired in the generated
algorithm label, but an inner `key` does not set the wrapper's active key.

#### Inversion Parameters

Used by the **irreversibility** and **key selection** pipelines to configure template inversion.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | string | `"minimize_cos"` | Inversion method. Options: `minimize_cos`, `minimize_l2`, `minimize_hamming`, `minimize_surrogate`, `minimize_hard_constraints`, `root_cos`, `root_l2`, `root_hamming`, `lsq_cos`, `lsq_l2`, `cmaes_cos`, `cmaes_l2`, `cmaes_hamming`. |
| `num_guesses` | int | `100` (`1` for CMA-ES, `10` for surrogate/hard constraints) | Number of different initial guesses for each inversion attempt. |
| `precision` | int | `3` | Decimal precision for computing unprotected template distribution. |
| `cma_sigma` | float | `0.1` | Initial CMA-ES sampling standard deviation. Must be greater than zero. |
| `cma_maxiter` | int or null | `100` | Maximum CMA-ES generations per initial guess. Use `null` for the library default. |
| `cma_popsize` | int or null | `null` | CMA-ES population size. Use `null` for the dimension-dependent library default. |
| `cma_seed` | int or null | `42` | Random seed for reproducible CMA-ES runs. Use `null` for a time-based seed. |
| `surrogate_temperature` | float | `0.1` | Logistic-loss temperature used by `minimize_surrogate`. |
| `surrogate_regularization` | float | `0.001` | L2 penalty anchoring the solution to its initial guess. |
| `surrogate_maxiter` | int | `500` | Maximum L-BFGS-B iterations for each surrogate inversion attempt. |
| `hard_constraint_epsilon` | float | `1e-10` | Minimum signed margin enforced by `minimize_hard_constraints`. |
| `hard_constraint_maxiter` | int | `500` | Maximum SLSQP iterations for each hard-constrained inversion attempt. |
| `hard_constraint_ftol` | float | `1e-9` | SLSQP convergence tolerance for hard-constrained inversion. |

The CMA-ES defaults target normalized, high-dimensional BioHash inputs. CMA-ES
uses one initial guess by default, the binary surrogate and hard-constrained
methods use 10, and the other inversion methods use 100.
During shared irreversibility/key-validation runs, the derived per-trial
`attack_seed` overrides `cma_seed` so that CMA-ES trials are distinct and the
whole experiment remains reproducible. `cma_seed` remains the default for
direct API calls that do not supply an inversion seed.

`minimize_surrogate` and `minimize_hard_constraints` require a binary BTP output,
currently BioHash or PolyProtect with `binarize: true`.

#### Key Selection Parameters

Used by the **key selection** pipelines.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ks_method` | string | `"legacy"` | Key selection method: `legacy` or `multiple_guesses`. |
| `ks_n_elements` | positive int or null | `null` | Number of guesses for `multiple_guesses`. It is required and must be a positive integer when `ks_method: multiple_guesses`; it is unused by `legacy`. |

### BioHash

```yaml
- type: biohash
  num_bits: 128
  binarize: true
  num_features: 512
  system_specific: true
  key: 42
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_bits` | int | `64` | Number of output bits (e.g., 64, 128, 256). |
| `binarize` | bool | `true` | Whether to binarize the output. |
| `num_features` | int | `512` | Number of features in the input vector. |

### PolyProtect

```yaml
- type: polyprotect
  overlap: 2
  nb_coef: 4
  coef_range: 10
  binarize: false
  system_specific: true
  key: 39
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `overlap` | int | `2` | Number of input positions shared by adjacent polynomial windows; use `0 <= overlap < nb_coef`. |
| `nb_coef` | int | `4` | Number of coefficients per segment. |
| `coef_range` | int | `10` | Range of coefficient values (excluding 0). |
| `binarize` | bool | `false` | Whether to binarize the PolyProtected output before comparison. |

### Combined

Multiple instances of the same algorithm with outputs combined into a single template.

```yaml
- type: combined
  inner_type: polyprotect
  nb_algs: 5
  normalized: true
  system_specific: true
  key: 34
  algs_config:
    overlap: 2
    nb_coef: 4
    coef_range: 10
    system_specific: true
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `inner_type` | string | Yes | Inner algorithm type: `biohash` or `polyprotect`. |
| `nb_algs` | int | Yes | Number of instances to combine (must be > 1). |
| `normalized` | bool | Yes | Whether to normalize the combined output. |
| `algs_config` | dict | Yes | Configuration for the inner algorithm (same fields as the corresponding type above). |

## Full Examples

### Identification

```yaml
output_dir: ./results/soteria
save: true
compliant: true
num_processes: 10
bio_alg: edgeface
database: soteria
detector: mediapipe

protocols:
  - grandtest
splits: eval

btps:
  algs:
    - type: combined
      inner_type: polyprotect
      normalized: true
      system_specific: true
      key: 34
      nb_algs: 5
      algs_config:
        overlap: 2
        nb_coef: 4
        coef_range: 10
        system_specific: true
    - type: biohash
      num_bits: 128
      binarize: true
      system_specific: true
      key: 42
```

For an irreversibility run, use the same common and BTP fields, add the
[inversion evaluation](#inversion-evaluation-fields) and desired
[key-source](#irreversibility-key-fields) settings, and omit `protocols` and
`splits`. The pipeline uses the fixed irreversibility CSVs described under
[Identification Fields](#identification-fields).

### Verification / Distribution Plots

```yaml
output_dir: ./results/soteria
save: true
compliant: true
num_processes: 10
bio_alg: edgeface
database: soteria
detector: mediapipe

btps:
  algs:
    - type: polyprotect
      overlap: 2
      nb_coef: 4
      coef_range: 10
      system_specific: true
      key: 39
```

### User-Specific Key Selection

```yaml
output_dir: ./results/key_selection
save: true
compliant: true
num_processes: 10
bio_alg: edgeface
database: soteria
detector: mediapipe

key_sampling_seed: 42

btps:
  algs:
    - type: polyprotect
      system_specific: false
      overlap: 3
      nb_coef: 5
      coef_range: 50
      num_guesses: 100
      precision: 3
      method: minimize_cos
      ks_method: legacy
```

### System-Specific Irreversibility Evaluation

Use this configuration with `btpbench irreversibility pipeline`.

```yaml
output_dir: ./results/irreversibility
save: true
compliant: true
num_processes: 15
bio_alg: edgeface
database: soteria
detector: mediapipe

n_attack_trials: 10
attack_seed: 42
key_sampling_seed: 42
sampling_mode: keys
n_keys: 100
keys_file: ./results/key_explorer/key_explorer_-0.7_-0.8_-0.9-legacy_None-minimize_cos-3-100-normalized_polyprotect_usr_3_5_50.json
keys_bucket: "-0.7"

btps:
  algs:
    - type: polyprotect
      overlap: 2
      nb_coef: 5
      coef_range: 50
      system_specific: true
      num_guesses: 100
      normalize_input: true
      precision: 3
      method: minimize_cos
```

### Key Explorer

The key explorer does not use a system configuration, real database, baseline, detector, or saved templates. It requires only `output_dir`, `compliant`, `num_processes`, and `btps`.

Each top-level BTP entry must use `system_specific: false`, because the key
explorer runs the user-specific candidate-key search. For a `combined` wrapper,
place that flag on the outer wrapper entry.

```yaml
output_dir: ./results/key_explorer
compliant: true
num_processes: 15

btps:
  algs:
    - type: polyprotect
      overlap: 3
      nb_coef: 5
      coef_range: 50
      system_specific: false
      num_guesses: 100
      normalize_input: true
      precision: 3
      method: minimize_cos
      ks_method: legacy
```

### Diversity

The diversity pipeline uses only the common fields and the `btps` section. All diversity-specific parameters (keys file, bucket, protected score file, FMR values, score ranges) are passed as CLI arguments — see [docs/diversity.md](diversity.md).

The pipeline explicitly supplies each selected key when protecting a template, so `system_specific` does not control key assignment in this workflow. Use the setting that gives the intended algorithm/output label; active diversity launchers use `system_specific: false`.

```yaml
output_dir: ./results/diversity
save: true
compliant: true
num_processes: 10
bio_alg: edgeface
database: soteria
detector: mediapipe

btps:
  algs:
    - type: combined
      inner_type: polyprotect
      normalized: true
      system_specific: false
      nb_algs: 5
      algs_config:
        overlap: 2
        nb_coef: 5
        coef_range: 40
        system_specific: false
```
