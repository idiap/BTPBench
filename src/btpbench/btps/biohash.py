# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-FileContributor: Vedrana Krivokuća Hahn <vedrana.krivokuca@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

from functools import cache
from pathlib import Path
from typing import Any

import numpy

from btpbench.btps import BaselineBTP


class BioHash(BaselineBTP):
    """BioHash BTP algorithm."""

    _default_binarize = True
    _supports_binary_inversion = True

    def __init__(
        self,
        work_dir: Path,
        save: bool = False,
        config: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(work_dir, save, config, **kwargs)
        self._num_bits = self._config.get("num_bits", 64)
        self._num_features = self._config.get("num_features", 512)

        self._use_algorithm_work_dir()

    def get_alg_name(self) -> str:
        """Return algorithm name."""
        system_specific_tag = "sys" if self._system_specific else "usr"
        binarize_tag = "binary" if self._binarize else "real"
        return f"{self._normalize_tag}_biohash_{system_specific_tag}_{binarize_tag}_{self._num_bits}"

    @cache
    def _orthonormal_matrix(self, key: int, n_features: int) -> numpy.ndarray:
        """Generate (and cache) a n_features × num_bits orthonormal matrix keyed by seed."""
        rng = numpy.random.default_rng(key)
        rand_mat = rng.random((n_features, self._num_bits))
        orth_mat, _ = numpy.linalg.qr(rand_mat, mode="reduced")
        return orth_mat

    def get_secret(self, key: int) -> dict:
        return {"orth_mat": self._orthonormal_matrix(key, self._num_features)}

    def _scores(self, feature_vector: numpy.ndarray, key: int | list[int]):
        key = self._single_key(key)
        return feature_vector @ self.get_secret(key)["orth_mat"]

    def _protect(
        self, feature_vector: numpy.ndarray, key: int | list[int]
    ) -> numpy.ndarray:
        """Create a BioHash."""
        return self._scores(feature_vector, key)

    def _binary_scores_and_jacobian(
        self,
        feature_vector: numpy.ndarray,
        key: int | list[int],
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        key = self._single_key(key)
        orth_mat = self.get_secret(key)["orth_mat"]
        return feature_vector @ orth_mat, orth_mat.T
