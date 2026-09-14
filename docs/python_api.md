# Python API

BTPBench can be used without the CLI. The package root does not
re-export its classes, so import objects from the modules shown below.

## Data model and loading

`btpbench.sample.Sample` stores one media item:

| Attribute | Meaning |
|---|---|
| `subject_id` | Identity label. |
| `template_id` | Identifier used for caching and score output. |
| `path` | Source image or video path. |
| `metadata` | Additional protocol columns. |
| `data` | Zero-argument callable that loads the RGB array lazily. |

The built-in loaders are `cv_loader()` for images and `cv_loader_video()` for
videos. Both return RGB NumPy arrays. A positive video `frame_idx` is treated as
a percentage of the video length; `random_frame=True` selects a random frame.

`btpbench.dataloader.Dataset` combines the protocol directory, raw
dataset directory, and the two sample lists used by verification and
unlinkability:

```python
from pathlib import Path

from btpbench.dataloader import Dataset, cv_loader

dataset = Dataset(
    protocols_dir=Path("./protocols/example"),
    dataset_dir=Path("/path/to/example"),
    verification_samples_file=Path("./protocols/example/example-verification.csv"),
    unlinkability_samples_file=Path("./protocols/example/example-unlinkability.csv"),
    extension=".png",
    compliant=True,
    load_f=cv_loader,
)
```

The main dataset methods are:

| Method | Result |
|---|---|
| `samples(n_subjects=-1, verification=True)` | Verification samples, or unlinkability samples when `verification=False`. |
| `metadata_names(verification=True)` | Extra CSV column names for the selected sample list. |
| `protocols()` | Discovered identification protocol names. |
| `protocol_splits(name)` | Split names available for a protocol. |
| `load_protocol(name, split_name="eval")` | A `Dataloader` with `references()` and `probes()` generators. |
| `load_irreversibility()` | A `Dataloader` for the two fixed irreversibility CSVs. |

With `compliant=True`, missing media files are logged and skipped. With
`compliant=False`, a missing file raises `FileNotFoundError`.

See [Protocol Format and Custom Databases](protocols.md) for the complete input
schema and a loader-based smoke test for a new database.

## Face-recognition baselines

`btpbench.algorithms.get_baseline_dict()` returns the baseline registry:

| Registry name | Class |
|---|---|
| `iresnet100` | `IResNet100` |
| `iresnet50` | `IResNet50` |
| `edgeface` | `EdgeFaceBase` |
| `edgefacexs` | `EdgeFaceXS` |
| `facenet` | `FaceNet` |
| `random` | `Random` (testing only) |

The model classes implement the `BaselineAlg` interface:

```python
processed = baseline.preprocessor(sample)
template = baseline.feature_extraction(processed)
score = baseline.compare(reference_template, probe_template)
```

Model constructors accept a working directory and a `save` flag. The face
models also accept `detector="mediapipe"` or `detector="mtcnn"`. When `save` is
true, `feature_extraction()` caches templates below the working directory.
EdgeFace and FaceNet may download their upstream model parameters on first use.

`btpbench.baselines.Template` represents an unprotected feature vector.
Its `get_template()`, `save()`, and `load()` methods provide array access and
NumPy-based persistence. `BaselineAlg.compare()` returns negative cosine
distance; larger scores therefore indicate greater similarity.

## Biometric template protection

`btpbench.algorithms.get_protected_baseline_dict()` returns the BTP
registry:

| Registry name | Implementation |
|---|---|
| `biohash` | BioHash |
| `polyprotect` | PolyProtect |
| `combined` | Combine independent instances into one template |

BTP classes receive their algorithm-specific values in a `config` dictionary.
The [experiment configuration guide](experiment_config.md#btp-algorithms)
documents every supported key and default.

```python
from pathlib import Path

import numpy as np

from btpbench.baselines import Template
from btpbench.btps.biohash import BioHash

reference = Template("subject-1", "reference", np.ones(4))
probe = Template("subject-1", "probe", np.array([1.0, 0.9, 1.1, 1.0]))

algorithm = BioHash(
    Path("./work"),
    config={
        "num_bits": 4,
        "num_features": 4,
        "system_specific": True,
        "key": 42,
    },
)
protected_reference = algorithm.protect(reference)
protected_probe = algorithm.protect(probe)
score = algorithm.compare(protected_reference, protected_probe)
```

`protect(template, key=None)` returns a `ProtectedTemplate`. An explicit key
overrides configured key selection for that call. Binary configurations compare
with negative Hamming distance; other configurations compare with negative
cosine distance. Every supported BTP comparison returns one scalar score.

`invert(protected_template, template_distribution, dont_save=False, seed=None)`
reconstructs an unprotected `Template`. Supplying `seed` makes both the initial
guess sampling and CMA-ES randomness reproducible. Use a distinct derived seed
for every independent attack trial; the CLI does this automatically.

## Score files

`btpbench.scorewriter.CSVScoreWriter` writes the common score schema
used by the CLI:

```python
from pathlib import Path

from btpbench.scorewriter import CSVScoreWriter

writer = CSVScoreWriter(Path("scores.csv"), metadata_names=[])
try:
    writer.write_score(score, protected_reference, protected_probe)
finally:
    writer.close()
```

Always call `close()`: it flushes buffered rows and atomically renames the
temporary file to the requested output path. `write_score()` accepts one scalar
score per comparison; score lists are rejected.

The lower-level metric and plotting functions are available from
`btpbench.metrics`. For complete evaluation workflows, the CLI groups
documented in the [README](../README.md#experiments) provide configuration
validation, parallel execution, consistent filenames, and reporting.
