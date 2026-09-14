# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import threading

from collections import OrderedDict
from pathlib import Path

import timm
import torch

from torch import nn

from btpbench.baselines._checkpoint import load_verified_torch_checkpoint
from btpbench.baselines._torch import TorchEmbeddingBaseline
from btpbench.preprocessor import create_detector

_MODEL_LOAD_LOCK = threading.Lock()

_ARCHITECTURES = {
    "edgeface_base": ("edgenext_base", None),
    "edgeface_xs_gamma_06": ("edgenext_x_small", 0.6),
}


def _replace_linear_with_low_rank(module: nn.Module, rank_ratio: float) -> None:
    """Replace linear layers with the factorization used by EdgeFace-GAMMA."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and "head" not in name:
            rank = max(
                2,
                int(min(child.in_features, child.out_features) * rank_ratio),
            )
            replacement = nn.Sequential(
                OrderedDict(
                    (
                        (
                            "linear1",
                            nn.Linear(child.in_features, rank, bias=False),
                        ),
                        (
                            "linear2",
                            nn.Linear(
                                rank,
                                child.out_features,
                                bias=child.bias is not None,
                            ),
                        ),
                    )
                )
            )
            setattr(module, name, replacement)
        else:
            _replace_linear_with_low_rank(child, rank_ratio)


def _create_edgeface_model(model: str) -> nn.Module:
    """Build the architecture expected by an EdgeFace checkpoint."""
    architecture, rank_ratio = _ARCHITECTURES[model]
    backbone = timm.create_model(architecture)
    backbone.reset_classifier(512)

    edgeface = nn.Sequential(OrderedDict((("model", backbone),)))
    if rank_ratio is not None:
        _replace_linear_with_low_rank(edgeface, rank_ratio)
    return edgeface


class EdgeFace(TorchEmbeddingBaseline):
    """Base class for EdgeFace model."""

    CHECKPOINTS = {
        "edgeface_base": (
            (
                "https://huggingface.co/Idiap/EdgeFace-Base/resolve/"
                "2944a63d6a73efb3754c553e119b57fe7e54a179/"
                "edgeface_base.pt"
            ),
            "95861c09b22810136f43ec98845e7f09bfc3c43f5a804984a7bd2eac20abc30c",
        ),
        "edgeface_xs_gamma_06": (
            (
                "https://huggingface.co/Idiap/EdgeFace-XS-GAMMA/resolve/"
                "735c1b59bdc798260e56f12533fddfe8a8c4c568/"
                "edgeface_xs_gamma_06.pt"
            ),
            "5ae7504cd9aee0a5d52c2115fd2eb66b0985dd1730f40134b5854e0cb658ce16",
        ),
    }

    def __init__(
        self,
        model: str,
        work_dir: Path,
        save: bool,
        detector: str = "mediapipe",
    ):
        super().__init__(work_dir, save)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        with _MODEL_LOAD_LOCK:
            self._model = _create_edgeface_model(model)
            self._model.load_state_dict(self._load_checkpoint(model))
        self._model.to(self._device)
        self._model.eval()

        self._detector = create_detector(detector)

    @classmethod
    def _load_checkpoint(cls, model: str):
        """Download and load a checkpoint only after SHA-256 verification."""
        checkpoint_url, expected_hash = cls.CHECKPOINTS[model]
        checkpoint_file = Path(torch.hub.get_dir()) / "checkpoints" / f"{model}.pt"
        return load_verified_torch_checkpoint(
            checkpoint_url,
            expected_hash,
            checkpoint_file,
        )


class EdgeFaceBase(EdgeFace):
    """EdgeFace base model."""

    def __init__(
        self,
        work_dir: Path,
        save: bool,
        detector: str = "mediapipe",
    ):
        model = "edgeface_base"
        super().__init__(model, work_dir, save, detector)


class EdgeFaceXS(EdgeFace):
    """Extra Small Edge Face model."""

    def __init__(
        self,
        work_dir: Path,
        save: bool,
        detector: str = "mediapipe",
    ):
        model = "edgeface_xs_gamma_06"
        super().__init__(model, work_dir, save, detector)
