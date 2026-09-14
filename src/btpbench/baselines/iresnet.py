# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import importlib.resources

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from btpbench.baselines._torch import TorchEmbeddingBaseline
from btpbench.baselines.iresnet_base import iresnet50, iresnet100
from btpbench.preprocessor import create_detector


class IResNetModel(TorchEmbeddingBaseline):
    """Base class for IResNet model."""

    def __init__(
        self,
        loader: Callable[[str], Any],
        weight_file: Path,
        work_dir: Path,
        save: bool,
        detector: str = "mediapipe",
    ):
        super().__init__(work_dir, save)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = loader(str(weight_file))
        self._model.to(self._device)
        self._model.eval()

        self._detector = create_detector(detector)


class IResNet100(IResNetModel):
    """IResNet100 model."""

    def __init__(
        self,
        work_dir: Path,
        save: bool,
        detector: str = "mediapipe",
    ):
        weight_file = Path(
            str(
                importlib.resources.files("btpbench.baselines.weights").joinpath(
                    "iresnet100-73e07ba7.pth"
                )
            )
        )

        if not weight_file.exists():
            raise FileNotFoundError(f"No file `{str(weight_file)}`")

        super().__init__(iresnet100, weight_file, work_dir, save, detector)


class IResNet50(IResNetModel):
    """IResNet50 model."""

    def __init__(
        self,
        work_dir: Path,
        save: bool,
        detector: str = "mediapipe",
    ):
        weight_file = Path(
            str(
                importlib.resources.files("btpbench.baselines.weights").joinpath(
                    "iresnet50-7f187506.pth"
                )
            )
        )

        if not weight_file.exists():
            raise FileNotFoundError(f"No file `{str(weight_file)}`")

        super().__init__(iresnet50, weight_file, work_dir, save, detector)
