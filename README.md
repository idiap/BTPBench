# BTPBench

BTPBench is a Python library and command-line application for
evaluating face-recognition systems and biometric template protection (BTP)
algorithms. It provides reproducible workflows for identification,
verification, irreversibility, key selection, diversity, and unlinkability.

The repository contains protocol definitions, but not the biometric datasets.
You must obtain each dataset under its own terms and point the system
configuration at your local copy.

## Installation

BTPBench requires Python 3.12 and Git LFS. Install and initialize Git
LFS before cloning so that the bundled model files are checked out correctly:

```bash
git lfs install
git clone https://gitlab.idiap.ch/primeaid/btpbench.git
cd btpbench
```

The recommended installer is [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

For development tools, install the `dev` extra:

```bash
uv sync --extra dev
```

You can also install the project with another PEP 517-compatible installer.

The IResNet and MediaPipe parameters are stored with Git LFS. EdgeFace and FaceNet fetch
their upstream model parameters automatically on first use and therefore
require network access at that point.

## Face-recognition and face-detection models and licenses

Everything needed to run these models is shipped with BTPBench or its installed
dependencies, or fetched automatically on first use, so no extra setup actions
are required. The following table covers the upstream implementations and
pretrained parameters used by BTPBench. Implementation and checkpoint terms are
listed separately where they differ.

| Model | Configuration key | Parameter source | Upstream license and weight terms |
|---|---|---|---|
| iResNet50 | `iresnet50` | Bundled [pytorch-insightface](https://github.com/nizhib/pytorch-insightface) checkpoint | Implementation: [MIT](https://github.com/nizhib/pytorch-insightface/blob/main/LICENSE); weights: [non-commercial research use only](LICENSES/LicenseRef-InsightFace-NonCommercial-Model.txt) |
| iResNet100 | `iresnet100` | Bundled [pytorch-insightface](https://github.com/nizhib/pytorch-insightface) checkpoint | Implementation: [MIT](https://github.com/nizhib/pytorch-insightface/blob/main/LICENSE); weights: [non-commercial research use only](LICENSES/LicenseRef-InsightFace-NonCommercial-Model.txt) |
| EdgeFaceBase | `edgeface` | Downloaded [EdgeFace-Base](https://huggingface.co/Idiap/EdgeFace-Base) checkpoint | Model and weights: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| EdgeFaceXS | `edgefacexs` | Downloaded [EdgeFace-XS-GAMMA-06](https://huggingface.co/Idiap/EdgeFace-XS-GAMMA) checkpoint | Model and weights: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| FaceNet | `facenet` | Downloaded [facenet-pytorch VGGFace2](https://github.com/timesler/facenet-pytorch#pretrained-models) checkpoint, ported from David Sandberg's [20180402-114759](https://github.com/davidsandberg/facenet#pre-trained-models) model | Implementations: [MIT (PyTorch port)](https://github.com/timesler/facenet-pytorch/blob/master/LICENSE.md) and [MIT (TensorFlow source)](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md) |
| MediaPipe Face Detector (BlazeFace short-range) | `detector: mediapipe` | Bundled [MediaPipe BlazeFace short-range](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md#face-detector) `detector.tflite` model | Upstream implementation and [BTPBench asset record](REUSE.toml): [Apache-2.0](LICENSES/Apache-2.0.txt) |

## Quick start

Follow the
[user-specific quickstart](quickstart/README.md)
for one complete, step-by-step experiment. It measures baseline protected
verification, selects inversion-resistant user keys, repeats security and
recognition measurements, then compares random and selected keys for
unlinkability and diversity. Every score, metric table, and plot is generated
locally by the documented commands.

For a custom dataset or experiment, start from
[`config/system_config.yaml`](config/system_config.yaml)
and
[`config/experiment_config.yaml`](config/experiment_config.yaml),
then use the detailed guides below. Every command has local help, for example:

```bash
uv run btpbench verification pipeline --help
```

## Protocols and custom databases

Protocol CSVs map local images or videos to subject and template IDs and define
the samples used by each evaluation. The
[protocol authoring guide](docs/protocols.md)
documents the shared CSV schema, identification directory layout, image and
video path rules, verification and unlinkability sample lists,
irreversibility files, and a validation workflow for adding a new database.

## Experiments

| Experiment | Purpose | Guide |
|---|---|---|
| Identification | 1:N template matching and DIR analysis | [Identification](docs/identification.md) |
| Verification | Pairwise biometric comparison and DET analysis | [Verification](docs/verification.md) |
| Irreversibility | Resistance of protected templates to inversion attacks | [Irreversibility](docs/irreversibility.md) |
| Key selection | User-specific and system-specific BTP key evaluation | [Key selection](docs/key_selection.md) |
| Diversity | Mutually non-matching protected-template sets | [Diversity](docs/diversity.md) |
| Unlinkability | Linkability analysis across differently protected templates | [Unlinkability](docs/unlinkability.md) |
| Standalone plots | Distribution, histogram, and PolyProtect visualizations | [Plots](docs/plots.md) |

See [system configuration](docs/system_config.md)
and [experiment configuration](docs/experiment_config.md)
for the complete YAML reference. The
[Python API guide](docs/python_api.md)
covers direct library use.

## Development

Run the test suite and repository checks with:

```bash
uv run pytest
uv run pre-commit run --all-files
uv run reuse lint
```

To install the Git hooks locally, run `uv run pre-commit install` once.

## Acknowledgments

We acknowledge the authors of the following methods, code, and evaluation
framework used in BTPBench:

- **PolyProtect:** the original method described in the
  [PolyProtect paper](https://ieeexplore.ieee.org/abstract/document/9670462)
  and the authors' [original implementation](https://gitlab.idiap.ch/bob/bob.paper.polyprotect_2021).
- **BioHashing:** the original method described in the
  [BioHashing paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320304001876).
  We also acknowledge the authors of
  [bob.chapter.fingerveins_biohashing](https://gitlab.idiap.ch/bob/bob.chapter.fingerveins_biohashing)
  for their separate BioHashing implementation.
- **Unlinkability evaluation:** the framework proposed by Gómez-Barrero et al.
  in their [unlinkability framework paper](https://ieeexplore.ieee.org/abstract/document/8241848).

## Licensing

Project-authored code and documentation are licensed under the
[Research Only and Non Commercial License](LICENSES/LicenseRef-primeaid-NC.txt).
See
[`THIRDPARTY.md`](THIRDPARTY.md)
for dependency licenses and the `LICENSES/`
directory for license texts.

Bundled and downloaded third-party components retain their upstream terms.
Review the model-specific terms above before redistributing the package or
using the weights.
