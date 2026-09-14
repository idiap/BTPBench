# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import threading

from pathlib import Path

import torch

from facenet_pytorch import InceptionResnetV1

from btpbench.baselines._checkpoint import load_verified_torch_checkpoint
from btpbench.baselines._torch import TorchEmbeddingBaseline
from btpbench.preprocessor import create_detector

_MODEL_LOAD_LOCK = threading.Lock()


class FaceNet(TorchEmbeddingBaseline):
    """FaceNet embedding model using the VGGFace2 checkpoint."""

    CHECKPOINT_URL = (
        "https://github.com/timesler/facenet-pytorch/releases/download/"
        "v2.2.9/20180402-114759-vggface2.pt"
    )
    CHECKPOINT_SHA256 = (
        "281cebca8662831adb987a874bdcb36e73f5b1c6dc5ee5878f305e985625d99b"
    )

    def __init__(
        self,
        work_dir: Path,
        save: bool,
        detector: str = "mediapipe",
    ):
        super().__init__(work_dir, save)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        with _MODEL_LOAD_LOCK:
            self._model = InceptionResnetV1(pretrained=None)
            state_dict = self._load_checkpoint()
            # Classification weights are not used when producing embeddings.
            state_dict.pop("logits.weight")
            state_dict.pop("logits.bias")
            self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

        self._detector = create_detector(
            detector,
            (160, 160),
            [(46, 107), (46, 53)],
        )

    @classmethod
    def _load_checkpoint(cls):
        """Download and load the VGGFace2 checkpoint after hash verification."""
        checkpoint_file = (
            Path(torch.hub.get_dir()).parent
            / "checkpoints"
            / "20180402-114759-vggface2.pt"
        )
        return load_verified_torch_checkpoint(
            cls.CHECKPOINT_URL,
            cls.CHECKPOINT_SHA256,
            checkpoint_file,
        )
