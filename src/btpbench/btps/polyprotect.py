# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-FileContributor: Vedrana Krivokuća Hahn <vedrana.krivokuca@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import logging
import math

from functools import cache
from pathlib import Path
from typing import Any

import numpy

from btpbench.btps import BaselineBTP

logger = logging.getLogger(__name__)


class PolyProtect(BaselineBTP):
    """PolyProtect BTP algorithm."""

    _supports_binary_inversion = True

    def __init__(
        self,
        work_dir: Path,
        save: bool = False,
        config: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(work_dir, save, config, **kwargs)

        self._overlap = self._config.get("overlap", 2)
        self._nb_coef = self._config.get("nb_coef", 4)
        self._c_range = self._config.get("coef_range", 10)

        self._use_algorithm_work_dir()

        # Calculate coefficient range (excluding 0):
        self._neg_range = numpy.arange(-1 * self._c_range, 0)
        self._pos_range = numpy.arange(1, self._c_range + 1)
        self._whole_range = numpy.concatenate([self._neg_range, self._pos_range])

        self._step_size = self._nb_coef - self._overlap

    def get_alg_name(self) -> str:
        """Return algorithm name."""
        system_specific_tag = "sys" if self._system_specific else "usr"
        binarize_tag = "_binary" if self._binarize else ""
        return f"{self._normalize_tag}_polyprotect_{system_specific_tag}{binarize_tag}_{self._overlap}_{self._nb_coef}_{self._c_range}"

    @cache
    def get_secret(self, key: int) -> dict:
        generator = numpy.random.default_rng(key)
        logger.info(f"Seed: {key}")
        if self._key_distribution is None:
            # Randomly generate m unique coefficients and exponents:
            coefficients = generator.permutation(self._whole_range)[
                0 : self._nb_coef
            ]  # randomly permute the whole range and pick the first few m values
            exponents = generator.permutation(range(1, self._nb_coef + 1))[
                0 : self._nb_coef
            ]  # permute the integers in the range [1, m]
        else:
            logger.info("C/E from dist")
            coefficients = [
                generator.choice(
                    self._key_distribution[str(self._key_distribution_tag)][str(i + 1)][
                        "values"
                    ],
                    p=self._key_distribution[str(self._key_distribution_tag)][
                        str(i + 1)
                    ]["probabilities"],
                )
                for i in range(self._nb_coef)
            ]
            exponents = [i + 1 for i in range(self._nb_coef)]

            # Shuffle coefficients and exponents together to maintain their association
            perm = generator.permutation(self._nb_coef)
            coefficients = [coefficients[i] for i in perm]
            exponents = [exponents[i] for i in perm]

        logger.info(
            f"Generated secret for key {key}: coefficients={coefficients}, exponents={exponents}"
        )

        return {"coefficients": coefficients, "exponents": exponents}

    @cache
    def _get_padding(self, feat_vec_size: int) -> int:
        decimal_remainder, _ = math.modf(
            (feat_vec_size - self._nb_coef) / self._step_size
        )

        padding = 0
        if decimal_remainder > 0:
            padding = math.ceil((1 - decimal_remainder) * self._step_size)

        return padding

    @cache
    def _get_starting_indices(self, feat_vec_size: int) -> list[int]:
        return list(range(0, feat_vec_size - self._nb_coef + 1, self._step_size))

    def _scores(
        self,
        feature_vector: numpy.ndarray,
        key: int | list[int],
    ) -> numpy.ndarray:
        key = self._single_key(key)
        secret = self.get_secret(key)

        coefficients = secret["coefficients"]
        exponents = secret["exponents"]

        padding = self._get_padding(len(feature_vector))

        # pad feature_vector by "padding" zeros at the end
        feature_vector = numpy.pad(
            feature_vector, (0, padding), "constant", constant_values=(0, 0)
        )

        starting_indices = self._get_starting_indices(len(feature_vector))

        # Each PolyProtected element will come from m non-overlapping feature_vector elements
        protected_feature_vector = numpy.zeros(len(starting_indices))
        storage_ind = 0
        for ind in starting_indices:
            final_ind = ind + self._nb_coef
            crnt_word = feature_vector[ind:final_ind]
            protected_feature_vector[storage_ind] = sum(
                coefficients[i] * crnt_word[i] ** (exponents[i])
                for i in range(0, self._nb_coef)
            )
            storage_ind = storage_ind + 1

        return protected_feature_vector

    def _binary_scores_and_jacobian(
        self,
        feature_vector: numpy.ndarray,
        key: int | list[int],
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        key = self._single_key(key)
        secret = self.get_secret(key)

        coefficients = secret["coefficients"]
        exponents = secret["exponents"]
        input_size = len(feature_vector)
        padding = self._get_padding(input_size)
        feature_vector = numpy.pad(
            feature_vector, (0, padding), "constant", constant_values=(0, 0)
        )
        starting_indices = self._get_starting_indices(len(feature_vector))

        scores = numpy.zeros(len(starting_indices))
        score_jacobian = numpy.zeros((len(starting_indices), input_size))
        for storage_ind, ind in enumerate(starting_indices):
            final_ind = ind + self._nb_coef
            crnt_word = feature_vector[ind:final_ind]
            scores[storage_ind] = sum(
                coefficients[i] * crnt_word[i] ** (exponents[i])
                for i in range(0, self._nb_coef)
            )

            for i in range(0, self._nb_coef):
                source_ind = ind + i
                if source_ind >= input_size:
                    continue
                score_jacobian[storage_ind, source_ind] += (
                    coefficients[i]
                    * exponents[i]
                    * feature_vector[source_ind] ** (exponents[i] - 1)
                )

        return scores, score_jacobian

    def _protect(
        self,
        feature_vector: numpy.ndarray,
        key: int | list[int],
    ):
        return self._scores(feature_vector, key)
