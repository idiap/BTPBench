# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-FileContributor: Vedrana Krivokuća Hahn <vedrana.krivokuca@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC


from abc import ABC
from pathlib import Path
from typing import Any

import numpy

from btpbench.btps import BaselineBTP, ProtectedTemplate


class MutlipleBTPAlgs(BaselineBTP, ABC):
    """Base class for Combined BTP experiments."""

    def __init__(
        self,
        work_dir: Path,
        save: bool = False,
        config: dict[str, Any] | None = None,
        prot_baseline_dict: dict[str, type] | None = None,
    ):
        super().__init__(work_dir=work_dir, save=save, config=config)
        prot_baseline_dict = {} if prot_baseline_dict is None else prot_baseline_dict
        self._n_algs = self._config["nb_algs"]

        if self._n_algs <= 1:
            raise RuntimeError("`MutlipleBTPAlgs` needs more than 1 BTP algorithm.")

        self._type = self._config["inner_type"]
        if self._type not in prot_baseline_dict:
            raise RuntimeError(f"Invalid btp algs `{self._type}`")

        if self._type == "combined":
            raise RuntimeError("Multiple BTP algorithms are not supported as child!")

        self._algs_config = self._config["algs_config"]

        btp_alg_cls = prot_baseline_dict[self._type]

        self._btps = [
            btp_alg_cls(
                self._work_dir / (self._type + f"_{i}"),
                False,
                self._algs_config,
                offset=i * 300 + self._key_offset,
            )
            for i in range(self._n_algs)
        ]
        self._use_algorithm_work_dir()

    def compare(self, ref: ProtectedTemplate, probe: ProtectedTemplate) -> float:
        """Use last btp alg compare for final comparison."""
        return self._btps[-1].compare(ref, probe)

    def get_secret(self, key: int | list[int]) -> dict:
        return dict()


class CombinedBTPAlgs(MutlipleBTPAlgs):
    """Combined BTP algorithms."""

    def __init__(
        self,
        work_dir: Path,
        save: bool = False,
        config: dict[str, Any] | None = None,
        prot_baseline_dict: dict[str, type] | None = None,
    ):
        super().__init__(work_dir, save, config, prot_baseline_dict)
        self._normalized = self._config["normalized"]

        inner_binarize = self._algs_config.get("binarize", self._type == "biohash")
        self._aggregate_majority = inner_binarize and self._type in {
            "biohash",
            "polyprotect",
        }
        self._binarize = self._type == "biohash" and not self._aggregate_majority
        self._compare_hamming = self._aggregate_majority or self._binarize

    def _protect(
        self, feature_vector: numpy.ndarray, key: int | list[int]
    ) -> numpy.ndarray:
        """Perform feature extraction."""

        if isinstance(key, list):
            if len(key) != self._n_algs:
                raise RuntimeError(
                    f"Number of keys must be equal to number of algorithms ({self._n_algs})"
                )
        else:
            key = [key + btp.get_key_offset() for btp in self._btps]

        protected_templates = []
        for i, btp in enumerate(self._btps):
            inner_template = btp._protect(feature_vector, key[i])  # noqa SLF001
            if self._aggregate_majority:
                inner_template = btp._postprocess_protected_template(  # noqa SLF001
                    inner_template
                )
            protected_templates.append(inner_template)

        prot_template = numpy.sum(
            # Each row represents an individual template
            numpy.vstack(protected_templates),
            axis=0,
        )

        if self._aggregate_majority:
            thresh = self._n_algs / 2
            majority_mask = prot_template > thresh
            tie_mask = prot_template == thresh

            prot_template = numpy.zeros_like(prot_template)
            prot_template[majority_mask] = 1
            # When we have a tie we select randomly a value
            rng = numpy.random.default_rng()
            prot_template[tie_mask] = rng.integers(0, 2, size=tie_mask.sum())

        else:
            if self._normalized:
                prot_template = prot_template / len(self._btps)

        return prot_template

    def get_alg_name(self) -> str:
        """Return algorithm name."""
        return f"{self._normalize_tag}_combined_{self._n_algs}_{self._btps[0].get_alg_name()}"

    def compare(self, ref: ProtectedTemplate, probe: ProtectedTemplate) -> float:
        """Compare combined templates."""
        if self._compare_hamming:
            return BaselineBTP.compare(self, ref, probe)
        return super().compare(ref, probe)
