# System Configuration

The system configuration declares the biometric baselines that experiments may select and maps database names to their protocol and raw-data locations. The repository provides an example at `config/system_config.yaml`.

The example uses `/path/to/...` placeholders for raw datasets. Replace them
with your local paths before running an experiment. Relative paths are resolved
from the process's current working directory, not from the directory containing
the YAML file; the examples below assume commands are run from the repository
root.

## Structure

### Baselines

```yaml
baselines:
  - iresnet100
  - iresnet50
  - edgeface
  - edgefacexs
  - facenet
  - random
```

`baselines` is a configuration whitelist: an experiment's `bio_alg` must appear here. Implementations are registered separately by `get_baseline_dict()` in `src/btpbench/algorithms.py`, so a baseline must be present in both places to be usable. Adding a name to the YAML does not implement or register it. The `random` baseline generates random embeddings for testing purposes.

### Protected Baselines

```yaml
protected_baselines:
  - biohash
  - polyprotect
  - combined
```

This list describes the registered Biometric Template Protection (BTP) algorithms, but is currently informational: the runtime does not read `protected_baselines` when validating or constructing a BTP algorithm. BTP implementations are selected from `get_protected_baseline_dict()` in `src/btpbench/algorithms.py`. Editing this YAML list therefore does not enable or disable an implementation. When adding a BTP implementation, update the code registry and keep this informational list synchronized where appropriate.

### Databases

Each database entry contains:

```yaml
databases:
  soteria:
    proto_dir: ./protocols/soteria
    verification_samples: ./protocols/soteria/soteria-verification.csv
    unlink_samples: ./protocols/soteria/soteria-bf-no-mask-samples.csv
    dataset_dir: /path/to/soteria
    extension: ""
    video: true
```

The provided configuration contains protocol bundles for `soteria`, `icarb`,
`mobio`, and `multipie`, plus `test`, a small protocol fixture and configuration
example. The corresponding raw media are not distributed with this project.

The fields have the following meanings:

| Field | Description |
|---|---|
| `proto_dir` | Root directory containing evaluation and irreversibility protocol CSVs. |
| `verification_samples` | Sample list used by verification and the default diversity flow. |
| `unlink_samples` | Sample list used by unlinkability and unlinkability-template diversity runs. |
| `dataset_dir` | Root directory containing the raw image or video files; these files are not included in this repository. |
| `extension` | Text appended verbatim to every CSV `path`; use an empty string when paths already contain their suffix. |
| `video` | Selects the video loader when `true` and the image loader when `false`. |

All four path fields are interpreted relative to the current working directory when they are not absolute. Both sample-list CSVs are read when a dataset is initialized, so `verification_samples` and `unlink_samples` must both point to readable files even when a particular command uses only one of them.

### Protocol files

All protocol CSVs share the required columns `path`, `subject_id`, and
`template_id`. Identification uses paired `for_enrolling.csv` and
`for_probing.csv` files below protocol and split directories, while
irreversibility uses two fixed files below `irreversibility/`.

See [Protocol Format and Custom Databases](protocols.md) for the complete CSV
schema, directory layout, path and video-frame behavior, evaluation-set
semantics, metadata constraints, worked example, and validation checklist.
