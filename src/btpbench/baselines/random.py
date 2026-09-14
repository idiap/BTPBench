# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

from pathlib import Path

import numpy

from btpbench.baselines import BaselineAlg, Template
from btpbench.sample import Sample


class Random(BaselineAlg):
    """Random FR model."""

    def __init__(
        self,
        work_dir: Path,
        save: bool,
        detector: str = "mediapipe",
        min_val: float = -1,
        max_val: float = 1,
    ):
        super().__init__(work_dir, save)

        self._min_val = min_val
        self._max_val = max_val

        self._generator = numpy.random.default_rng()

    def _feature_extraction(self, sample: Sample) -> Template:
        """Only perform feature extraction."""

        template = self._generator.uniform(
            low=self._min_val, high=self._max_val, size=512
        )  # Assuming feature vector of size 512
        return Template(sample.subject_id, sample.template_id, template)

    def preprocessor(self, sample: Sample) -> Sample:
        """Apply precosessors to work with the model."""
        return sample
