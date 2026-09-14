# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy

from scipy.spatial.distance import cdist

from btpbench.sample import Sample


def negative_cosine_distance(ref: "Template", probe: "Template") -> float:
    """Return the negative cosine distance between two templates."""
    ref_template = ref.get_template().reshape((1, -1))
    probe_template = probe.get_template().reshape((1, -1))
    return -1.0 * cdist(ref_template, probe_template, metric="cosine").squeeze()


class Template:
    """Biometric template."""

    def __init__(
        self,
        subject_id: str,
        template_id: str,
        template: numpy.ndarray,
        metadata: dict[str, Any] | None = None,
    ):
        self.subject_id = subject_id
        self.template_id = template_id
        self._template = template
        self.metadata = {} if metadata is None else metadata

    def save(self, path: Path):
        """Save a template on the disk."""

        if self._template is None:
            return

        path.touch(exist_ok=True)
        numpy.save(path, self._template)

    def get_template(self) -> numpy.ndarray:
        return self._template

    @staticmethod
    def load(subject_id: str, template_id: str, path: Path) -> tuple[bool, Any]:
        """Load a template from the disk."""
        if not path.exists():
            return False, None

        arr = numpy.load(str(path))
        return True, Template(subject_id, template_id, arr)


class BaselineAlg(ABC):
    """Baseline FR algorithm template class."""

    def __init__(self, work_dir: Path, save: bool = False):
        """Build new object.

        Args:
            work_dir `Path`: Algorithm working directory.
            save `bool`: Flag to save templates.
        """
        self._work_dir = work_dir
        self._save = save
        if not work_dir.exists():
            work_dir.mkdir(parents=True, exist_ok=True)

    def feature_extraction(self, sample: Sample) -> Template:
        """Perform feature extraction with backup management.

        Sample's metadata are automatically passed to Template's
        metadata.
        """
        filename = f"{sample.subject_id}_{sample.template_id}.npy"
        filename = filename.replace("/", "_")
        filename = filename.replace("\\", "_")

        # Only load from disk if save is true
        exist = False
        if self._save:
            template_file = self._work_dir / str(sample.subject_id) / filename
            exist, template = Template.load(
                sample.subject_id, sample.template_id, template_file
            )

        if not exist:
            template = self._feature_extraction(sample)

            template.metadata = sample.metadata
            if self._save:
                if not template_file.parent.exists():
                    template_file.parent.mkdir(exist_ok=True)

                template.save(template_file)
        else:
            template.metadata = sample.metadata

        return template

    @abstractmethod
    def _feature_extraction(self, sample: Sample) -> Template:
        """Only perform feature extraction."""
        pass

    @abstractmethod
    def preprocessor(self, sample: Sample) -> Sample:
        pass

    def compare(self, ref: Template, probe: Template) -> float:
        """Compute distance. Default to cosine similarity."""
        return negative_cosine_distance(ref, probe)
