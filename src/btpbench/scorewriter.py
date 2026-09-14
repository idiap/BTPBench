# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import csv

from numbers import Real
from pathlib import Path
from typing import Any

from btpbench.baselines import Template


class CSVScoreWriter:
    """Helper class to generate a score file."""

    def __init__(
        self, score_file_path: Path, metadata_names: list[str], buffer_size: int = 1000
    ):
        fixed_headers: list[str] = [
            "probe_template_id",
            "probe_subject_id",
            "bio_ref_template_id",
            "bio_ref_subject_id",
            "score",
        ]
        self._score_file_path = score_file_path
        self._score_file_path_temp = score_file_path.parent / (
            score_file_path.name + ".temporary"
        )

        # If temporary files exist we remove it.
        if self._score_file_path_temp.exists():
            self._score_file_path_temp.write_text("")

        self._metadata_names = metadata_names

        self._header = fixed_headers
        self._header += ["bio_ref_" + m for m in metadata_names]
        self._header += ["probe_" + m for m in metadata_names]

        self._buffer_size = buffer_size
        self._buffer: list[Any] = []

        self._score_file = Path.open(self._score_file_path_temp, "w", newline="")
        self._score_writer = csv.writer(self._score_file)
        self._score_writer.writerow(self._header)

    def close(self):
        """Close CSV file."""

        if self._buffer:
            self._score_writer.writerows(self._buffer)

        self._score_file.close()
        self._score_file_path_temp.rename(self._score_file_path)

    def write_score(
        self,
        score: float,
        ref_template: Template,
        probe_template: Template,
    ) -> None:
        """Write one scalar score to file."""
        if not isinstance(score, Real):
            raise TypeError("score must be a scalar number")

        self._write_score_atomic(score, ref_template, probe_template)

    def _write_score_atomic(
        self, score: float, ref_template: Template, probe_template: Template
    ):
        """Write single score in file."""

        self._buffer.append(
            [
                probe_template.template_id,
                probe_template.subject_id,
                ref_template.template_id,
                ref_template.subject_id,
                score,
            ]
            + [ref_template.metadata.get(m, "") for m in self._metadata_names]
            + [probe_template.metadata.get(m, "") for m in self._metadata_names]
        )

        if len(self._buffer) > self._buffer_size:
            self._score_writer.writerows(self._buffer)
            self._buffer.clear()
