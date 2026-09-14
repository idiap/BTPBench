# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

from collections.abc import Callable

import torch

from btpbench.baselines import BaselineAlg, Template
from btpbench.preprocessor import chanel_first, normalize
from btpbench.sample import Sample


class TorchEmbeddingBaseline(BaselineAlg):
    """Shared inference and preprocessing for PyTorch face embeddings."""

    _model: torch.nn.Module
    _device: str
    _detector: Callable[[Sample], Sample]

    def _feature_extraction(self, sample: Sample) -> Template:
        """Extract one flattened embedding without tracking gradients."""
        tensor = torch.Tensor(sample.data())
        with torch.no_grad():
            embedding = (
                self._model(tensor.to(self._device)).cpu().detach().numpy().flatten()
            )
        return Template(sample.subject_id, sample.template_id, embedding)

    def preprocessor(self, sample: Sample) -> Sample:
        """Detect, normalize, and reorder one face sample for model inference."""
        sample = self._detector(sample)
        sample = normalize(sample)
        return chanel_first(sample)
