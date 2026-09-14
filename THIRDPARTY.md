# Third-party software

## Runtime dependencies

| Dependency | Declared version | Use | License | Upstream |
|---|---:|---|---|---|
| bob-measure | `>=6.1.1,<7` | Biometric performance metrics | BSD-3-Clause | [bob.measure](https://gitlab.idiap.ch/bob/bob.measure) |
| Click | `>=8.1.7,<9` | Command-line interface | BSD-3-Clause | [pallets/click](https://github.com/pallets/click) |
| cma | `>=4.4.0,<5` | CMA-ES optimization | BSD-3-Clause | [CMA-ES/pycma](https://github.com/CMA-ES/pycma) |
| facenet-pytorch | `>=2.6.0,<3` | FaceNet and MTCNN models | MIT | [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch) |
| Matplotlib | `>=3.10,<4` | Static plots | [Matplotlib License](https://matplotlib.org/stable/project/license.html) (PSF-based) | [matplotlib/matplotlib](https://github.com/matplotlib/matplotlib) |
| MediaPipe | `>=0.10.18,<1` | Face detection | Apache-2.0 | [google/mediapipe](https://github.com/google/mediapipe) |
| NetworkX | `>=3.4,<4` | Diversity graph analysis | BSD-3-Clause | [networkx/networkx](https://github.com/networkx/networkx) |
| NumPy | `>=1.26.4,<2` | Numerical arrays and operations | BSD-3-Clause | [numpy/numpy](https://github.com/numpy/numpy) |
| opencv-contrib-python | `>=4.10.0.84,<4.12` | Image and video I/O and processing | Apache-2.0 | [opencv/opencv-python](https://github.com/opencv/opencv-python) |
| pandas | `>=2.2.3,<3` | Protocol and score tables | BSD-3-Clause | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) |
| Plotly | `>=5.24.1,<7` | Interactive unlinkability plots | MIT | [plotly/plotly.py](https://github.com/plotly/plotly.py) |
| PyYAML | `>=6.0.2,<7` | Configuration parsing | MIT | [yaml/pyyaml](https://github.com/yaml/pyyaml) |
| scikit-learn | `>=1.8.0,<2` | t-SNE distribution plots | BSD-3-Clause | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) |
| SciPy | `>=1.14.1,<2` | Optimization, distances, and statistics | BSD-3-Clause | [scipy/scipy](https://github.com/scipy/scipy) |
| timm | `>=1.0.12,<2` | EdgeFace model support | Apache-2.0 | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) |
| PyTorch | `>=2.2.2,<2.3` | Neural-network inference | BSD-3-Clause | [pytorch/pytorch](https://github.com/pytorch/pytorch) |

## Build and development dependencies

| Dependency | Declared version | Use | License | Upstream |
|---|---:|---|---|---|
| uv-build | `>=0.12.5,<0.13.0` | PEP 517 build backend | MIT OR Apache-2.0 | [astral-sh/uv](https://github.com/astral-sh/uv) |
| pre-commit | `>=4.6.2,<5` | Development hook runner | MIT | [pre-commit/pre-commit](https://github.com/pre-commit/pre-commit) |
| pytest | `>=9.1.1,<10` | Test runner | MIT | [pytest-dev/pytest](https://github.com/pytest-dev/pytest) |
| pytest-cov | `>=7.1,<8` | Test coverage | MIT | [pytest-dev/pytest-cov](https://github.com/pytest-dev/pytest-cov) |
| REUSE | `>=6.2,<7` | License-compliance checks | Apache-2.0 AND CC0-1.0 AND CC-BY-SA-4.0 | [fsfe/reuse-tool](https://github.com/fsfe/reuse-tool) |

The following tools are fetched by pre-commit at the revisions in
`.pre-commit-config.yaml`:

| Tool | Configured revision | License | Upstream |
|---|---:|---|---|
| Ruff pre-commit | `v0.16.1` | MIT | [astral-sh/ruff-pre-commit](https://github.com/astral-sh/ruff-pre-commit) |
| mypy | `v2.3.0` | MIT | [python/mypy](https://github.com/python/mypy) |
| pre-commit-hooks | `v6.0.0` | MIT | [pre-commit/pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks) |
