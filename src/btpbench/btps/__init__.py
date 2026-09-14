# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-FileContributor: Vedrana Krivokuća Hahn <vedrana.krivokuca@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import json
import logging

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import cma
import numpy

from scipy.optimize import least_squares, minimize, root
from scipy.spatial.distance import cdist
from scipy.special import expit

from btpbench.baselines import Template
from btpbench.exceptions import InvalidInputSampleError
from btpbench.utils import (
    Distribution,
    derive_seed,
    generate_guesses_from_distribution,
)

logger = logging.getLogger(__name__)


class ProtectedTemplate(Template):
    """Protected Biometric Templates."""

    def __init__(
        self, subject_id: str, template_id: str, prot_template: numpy.ndarray, keys: Any
    ):
        super().__init__(subject_id, template_id, prot_template)
        self._keys = keys

    def get_keys(self) -> Any:
        return self._keys

    def save(self, path: Path):
        super().save(path)  # Save protected template

    @staticmethod
    def _load_key_from_path(path: Path) -> int | list[int]:
        stem = path.stem
        marker_positions = [
            (stem.rfind("_sys_"), "_sys_"),
            (stem.rfind("_usr_"), "_usr_"),
        ]
        marker_position, marker = max(marker_positions, key=lambda item: item[0])
        if marker_position < 0:
            raise ValueError(f"Could not retrieve key from filename {path.name}")

        key_tag = stem[marker_position + len(marker) :]
        try:
            key_list = [int(x) for x in key_tag.split("_")]
        except ValueError as exc:
            raise ValueError(
                f"Could not retrieve key from filename {path.name}"
            ) from exc

        if len(key_list) == 0:
            raise ValueError(f"Could not retrieve key from filename {path.name}")
        if len(key_list) == 1:
            return key_list[0]
        return key_list

    @staticmethod
    def load(subject_id: str, template_id: str, path: Path) -> tuple[bool, Any]:
        ok, standard_template = Template.load(subject_id, template_id, path)
        if not ok:
            return False, None

        key = ProtectedTemplate._load_key_from_path(path)
        key_tag = (
            str(key)
            if isinstance(key, (int, numpy.integer))
            else "_".join(map(str, key))
        )

        return ok, ProtectedTemplate(
            standard_template.subject_id,
            standard_template.template_id + f"_k{key_tag}",
            standard_template.get_template(),
            key,
        )


class BaselineBTP(ABC):
    """Baseline BTP algorithm template class."""

    _default_binarize = False
    _supports_binary_inversion = False

    def __init__(
        self,
        work_dir: Path,
        save: bool = False,
        config: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Build new object.

        Args:
            work_dir `Path`: Algorithm working directory.
            save `bool`: Flag to save templates.
            config `dict`: Algorithm parameters.
        """
        self._work_dir = work_dir
        self._save = save
        self._config = {} if config is None else config

        # General parameters
        self._system_specific = self._config.get("system_specific", False)
        self._normalize_input = self._config.get("normalize_input", True)
        self._normalize_tag = "normalized" if self._normalize_input else "unnormalized"
        self._binarize = self._config.get("binarize", self._default_binarize)
        self._compare_hamming = self._binarize

        # Key parameters
        self._key = None
        self._key_dictionary = None
        self._key_dictionary_file = None
        self._key_distribution = None
        self._key_offset = int(kwargs.get("offset", 0))
        if self._system_specific:
            self._key = self._config.get("key", 42)
            if isinstance(self._key, (int, numpy.integer)):
                self._key += self._key_offset

        # Inversion parameters
        self._method = self._config.get("method", "minimize_cos")
        uses_cmaes = self._method.startswith("cmaes")
        uses_surrogate = self._method == "minimize_surrogate"
        uses_hard_constraints = self._method == "minimize_hard_constraints"
        if uses_cmaes:
            default_num_guesses = 1
        elif uses_surrogate or uses_hard_constraints:
            default_num_guesses = 10
        else:
            default_num_guesses = 100

        self._num_guesses = self._config.get("num_guesses", default_num_guesses)
        self._precision = self._config.get("precision", 3)
        self._cma_sigma = float(self._config.get("cma_sigma", 0.1))
        self._cma_maxiter = self._config.get("cma_maxiter", 100)
        self._cma_popsize = self._config.get("cma_popsize")
        self._cma_seed = self._config.get("cma_seed", 42)
        self._surrogate_temperature = float(
            self._config.get("surrogate_temperature", 0.1)
        )
        self._surrogate_regularization = float(
            self._config.get("surrogate_regularization", 0.001)
        )
        self._surrogate_maxiter = int(self._config.get("surrogate_maxiter", 500))
        self._hard_constraint_epsilon = float(
            self._config.get("hard_constraint_epsilon", 1e-10)
        )
        self._hard_constraint_maxiter = int(
            self._config.get("hard_constraint_maxiter", 500)
        )
        self._hard_constraint_ftol = float(
            self._config.get("hard_constraint_ftol", 1e-9)
        )

        if uses_cmaes:
            if self._cma_sigma <= 0:
                raise ValueError("cma_sigma must be greater than zero.")
            if self._cma_maxiter is not None and self._cma_maxiter <= 0:
                raise ValueError("cma_maxiter must be greater than zero.")
            if self._cma_popsize is not None and self._cma_popsize <= 1:
                raise ValueError("cma_popsize must be greater than one.")

        if uses_surrogate or uses_hard_constraints:
            if not self._binarize:
                raise ValueError(f"{self._method} requires binarize to be enabled.")
            if not self._supports_binary_inversion:
                raise ValueError(
                    f"{self._method} is not supported by {self.__class__.__name__}."
                )

        if uses_surrogate:
            if self._surrogate_temperature <= 0:
                raise ValueError("surrogate_temperature must be greater than zero.")
            if self._surrogate_regularization < 0:
                raise ValueError(
                    "surrogate_regularization must be greater than or equal to zero."
                )
            if self._surrogate_maxiter <= 0:
                raise ValueError("surrogate_maxiter must be greater than zero.")

        if uses_hard_constraints:
            if self._hard_constraint_epsilon <= 0:
                raise ValueError("hard_constraint_epsilon must be greater than zero.")
            if self._hard_constraint_maxiter <= 0:
                raise ValueError("hard_constraint_maxiter must be greater than zero.")
            if self._hard_constraint_ftol <= 0:
                raise ValueError("hard_constraint_ftol must be greater than zero.")

        # K.S. parameters
        self._ks_method = self._config.get("ks_method", "legacy")
        self._ks_n_elements = self._config.get("ks_n_elements", None)

        self._key_distribution_file = self._config.get("key_distribution_file", None)
        self._key_distribution_tag = self._config.get("dist_tag", "all")
        if self._key_distribution_file is not None:
            if not Path(self._key_distribution_file).exists():
                raise FileNotFoundError(
                    f"Key distribution file {self._key_distribution_file} not found."
                )

            with Path(self._key_distribution_file).open() as f:
                self._key_distribution = json.load(f)

            logger.info(
                f"Loaded key distribution from {self._key_distribution_file} with {len(self._key_distribution)} entries and tag {self._key_distribution_tag}."
            )

        # Try to load key dictionary if not system specific and exists
        if not self._system_specific:
            self._key_dictionary_file = self._config.get("key_dictionary_file", None)
            if self._key_dictionary_file is not None:
                if not Path(self._key_dictionary_file).exists():
                    raise FileNotFoundError(
                        f"Key dictionary file {self._key_dictionary_file} not found."
                    )

                self._key_dictionary = json.loads(
                    Path(self._key_dictionary_file).read_text()
                )

                logger.info(
                    f"Loaded key dictionary from {self._key_dictionary_file} with {len(self._key_dictionary)} entries."
                )

        if not work_dir.exists() and save:
            work_dir.mkdir(parents=True, exist_ok=True)

    def _use_algorithm_work_dir(self) -> None:
        """Append the algorithm name to the working directory and create it."""
        self._work_dir /= self.get_alg_name()
        if self._save:
            self._work_dir.mkdir(parents=True, exist_ok=True)

    def get_key_offset(self) -> int:
        """Return key offset."""
        return self._key_offset

    def is_system_specific(self) -> bool:
        """Return True if the algorithm is system specific."""
        return self._system_specific

    def set_key(self, key: int | list[int]):
        """Set the key for system specific algorithms."""
        if not self._system_specific:
            raise RuntimeError(
                "This method should only be called if alg is system specific."
            )
        self._key = key

    def has_key_dictionary(self) -> bool:
        """Return True if a key dictionary is used."""
        return self._key_dictionary is not None

    def has_key_distribution(self) -> bool:
        """Return True if a key distribution is used."""
        return self._key_distribution is not None

    def get_key_distribution_name(self) -> str | None:
        """Return the key distribution file name."""
        if self._key_distribution_file is None:
            return None

        return f"{Path(self._key_distribution_file).stem}_{self._key_distribution_tag}"

    def get_key_dictionary_name(self) -> str | None:
        """Return the key dictionary file name."""
        if self._key_dictionary_file is None:
            return None

        # Crop key_select from filename
        return "_".join(Path(self._key_dictionary_file).stem.split("_")[2:])

    def invert(
        self,
        protected_template: ProtectedTemplate,
        template_dist: Distribution,
        dont_save: bool = False,
        seed: int | None = None,
    ) -> Template:
        """Invert a protected template, optionally using reproducible randomness."""

        if protected_template.get_template() is None:
            return Template(
                protected_template.subject_id,
                protected_template.template_id,
                None,
                protected_template.metadata,
            )

        key = protected_template.get_keys()
        key_tag = (
            str(key)
            if isinstance(key, (int, numpy.integer))
            else "_".join(map(str, key))
        )

        exist = False
        if self._save and not dont_save:
            system_specific_tag = "sys" if self._system_specific else "usr"
            filename = f"inverted_{self.get_inversion_config_tag()}_{protected_template.subject_id}_{protected_template.template_id}_{system_specific_tag}_{key_tag}.npy"
            filename = filename.replace("/", "_")
            filename = filename.replace("\\", "_")

            template_file = (
                self._work_dir / str(protected_template.subject_id) / filename
            )
            exist, inverted_template = Template.load(
                protected_template.subject_id,
                "inverted_" + protected_template.template_id,
                template_file,
            )

            if exist:
                inverted_template.metadata = protected_template.metadata

        if not exist:
            raw_inverted_template = self._invert(
                protected_template.get_template(),
                key,
                template_dist,
                seed,
            )

            inverted_template = Template(
                protected_template.subject_id,
                "inverted_" + protected_template.template_id,
                raw_inverted_template,
            )

            inverted_template.metadata = protected_template.metadata

            if self._save and not dont_save:
                if not template_file.parent.exists():
                    template_file.parent.mkdir(exist_ok=True)

                inverted_template.save(template_file)

        return inverted_template

    def get_system_secret(self) -> dict:
        if not self._system_specific:
            raise RuntimeError(
                "This method should only be called if alg is system specific."
            )

        return self.get_secret(self._key)

    @abstractmethod
    def get_secret(self, key: int) -> dict:
        """Return keys for the algorithm."""
        pass

    def _single_key(self, key: int | list[int]) -> int:
        if isinstance(key, list):
            if len(key) != 1:
                raise RuntimeError(
                    "Number of keys must be equal to 1 for "
                    f"{self.__class__.__name__}, got {len(key)}"
                )
            key = key[0]
        return key

    @staticmethod
    def _binarize_scores(scores: numpy.ndarray) -> numpy.ndarray:
        thresh = scores.mean()
        return numpy.where(scores > thresh, 1, 0)

    def _postprocess_protected_template(
        self,
        protected_template: numpy.ndarray,
    ) -> numpy.ndarray:
        if not self._binarize:
            return protected_template
        if numpy.all(numpy.isin(protected_template, (0, 1))):
            return protected_template
        return self._binarize_scores(protected_template)

    def _protect_postprocessed(
        self,
        feature_vector: numpy.ndarray,
        key: int | list[int],
    ) -> numpy.ndarray:
        return self._postprocess_protected_template(self._protect(feature_vector, key))

    def _binary_scores_and_jacobian(
        self,
        feature_vector: numpy.ndarray,
        key: int | list[int],
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        raise ValueError(
            f"Binary inversion is not supported by {self.__class__.__name__}."
        )

    def _target_signs(self, protected_template: numpy.ndarray) -> numpy.ndarray:
        target = numpy.asarray(protected_template)
        if target.ndim != 1:
            raise ValueError(
                "Binary inversion requires a one-dimensional protected template, "
                f"got shape {target.shape}."
            )
        if not numpy.all(numpy.isin(target, (0, 1))):
            raise ValueError(
                f"{self.__class__.__name__} inversion requires a binary protected "
                "template."
            )
        return 2.0 * target - 1.0

    def _optimization_vector(
        self,
        feature_vector: numpy.ndarray,
    ) -> tuple[numpy.ndarray, float]:
        optimization_vector = feature_vector
        feature_norm = 1.0
        if self._normalize_input:
            feature_norm = numpy.linalg.norm(feature_vector)
            if feature_norm == 0:
                feature_norm = numpy.finfo(float).eps
            optimization_vector = feature_vector / feature_norm
        return optimization_vector, feature_norm

    def _centered_binary_scores_and_jacobian(
        self,
        feature_vector: numpy.ndarray,
        key: int | list[int],
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        scores, score_jacobian = self._binary_scores_and_jacobian(
            feature_vector,
            key,
        )
        return scores - scores.mean(), score_jacobian - score_jacobian.mean(
            axis=0,
            keepdims=True,
        )

    def _centered_binary_scores_and_input_jacobian(
        self,
        feature_vector: numpy.ndarray,
        key: int | list[int],
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        optimization_vector, feature_norm = self._optimization_vector(feature_vector)
        centered_scores, score_jacobian = self._centered_binary_scores_and_jacobian(
            optimization_vector,
            key,
        )

        if self._normalize_input:
            directional_derivative = score_jacobian @ optimization_vector
            score_jacobian = (
                score_jacobian
                - directional_derivative[:, numpy.newaxis] * optimization_vector
            ) / feature_norm

        return centered_scores, score_jacobian

    @staticmethod
    def _validate_binary_target_shape(
        target_signs: numpy.ndarray,
        centered_scores: numpy.ndarray,
    ) -> None:
        if target_signs.shape != centered_scores.shape:
            raise ValueError(
                "Expected a protected template with shape "
                f"{centered_scores.shape}, got {target_signs.shape}."
            )

    def key_selection_usr(
        self,
        template: Template,
        dist: Distribution,
        thresh: float,
        compare_f: Callable[[Template, Template], float],
        seed: int | None = None,
    ) -> tuple[ProtectedTemplate, Template, float]:
        """Find a suitable key for a given template."""

        if self._system_specific:
            raise RuntimeError(
                "This method should only be called if alg is in user specific mode."
            )

        # Call legacy method if specified
        if self._ks_method == "legacy":
            return self.__key_selection_usr_legacy(
                template,
                dist,
                thresh,
                compare_f,
                seed,
            )

        # Ensure we are in multiple guesses mode
        if self._ks_method != "multiple_guesses":
            raise ValueError(f"Unknown key selection method {self._ks_method}")

        # Store used keys to avoid repetitions
        used_keys: set[int] = set()
        rng = numpy.random.default_rng(seed)
        random_key = rng.integers(0, 2_000_000)

        while len(used_keys) < 2_000_000:
            while random_key in used_keys:
                random_key = rng.integers(0, 2_000_000)

            used_keys.add(random_key)

            # Compute protected vector
            prot_template_arr = self._protect_postprocessed(
                template.get_template(),
                random_key,
            )

            # Generate guesses
            inversion_seed = int(rng.integers(1, 2**32))
            guesses = generate_guesses_from_distribution(
                dist,
                self._ks_n_elements,
                inversion_seed,
            )

            guesses_ok = True

            inverted_template = None
            score = float("nan")

            for guess_index, guess in enumerate(guesses):
                # Try to invert only with current guess
                optimizer_seed = derive_seed(inversion_seed, "guess", guess_index)
                inverted_template_arr = self._invert_atomic(
                    prot_template_arr,
                    random_key,
                    guess,
                    optimizer_seed,
                )

                # Build inverted template object
                inverted_template = Template(
                    template.subject_id,
                    template.template_id,
                    inverted_template_arr,
                    template.metadata,
                )

                # Compare inverted template with original
                score = float("nan")
                if inverted_template.get_template() is not None:
                    score = compare_f(template, inverted_template)

                # If inverted template is too close we reject the key
                if score >= thresh:
                    guesses_ok = False
                    logger.debug(
                        f"Key {random_key} rejected with score {score} >= {thresh} for one of the guesses of subject {template.subject_id}"
                    )
                    break  # No need to check other guesses

            # If all guesses are ok we return the protected template
            if guesses_ok:
                logger.debug(
                    f"Key {random_key} accepted with score < {thresh} for all {self._ks_n_elements} guesses of subject {template.subject_id}"
                )

                # Build protected template object
                # We compute it again to ensure it's correctly saved on the disk if needed
                prot_template = self.protect(template, key=random_key)

                return (prot_template, inverted_template, score)

        # We should never reach this point
        raise RuntimeError(
            "K.S. did not find suitable key after exhausting all options."
        )

    def __key_selection_usr_legacy(
        self,
        template: Template,
        dist: Distribution,
        thresh: float,
        compare_f: Callable[[Template, Template], float],
        seed: int | None = None,
    ) -> tuple[ProtectedTemplate, Template, float]:
        """Legacy k.s. method kept only for reproducibility."""

        used_keys = set()
        rng = numpy.random.default_rng(seed)
        random_key = rng.integers(0, 2_000_000)

        while True:
            while random_key in used_keys:
                random_key = rng.integers(0, 2_000_000)
            used_keys.add(random_key)

            prot_template = self.protect(template, key=random_key)

            inversion_seed = int(rng.integers(1, 2**32))
            inverted_template = self.invert(prot_template, dist, True, inversion_seed)

            # score to nan if inversion failed
            score = float("nan")
            if inverted_template.get_template() is not None:
                score = compare_f(template, inverted_template)

            if score < thresh or numpy.isnan(score):
                # If protected template is suitable
                prot_template.template_id += (
                    "_0"  # To differentiate multiple keys for same template
                )
                return (prot_template, inverted_template, score)

            # If protected template is not suitable, delete saved version if any
            logger.debug(
                f"Key {prot_template.get_keys()} rejected with score {score} >= {thresh}"
            )

            system_specific_tag = "sys" if self._system_specific else "usr"
            filename = f"{prot_template.subject_id}_{prot_template.template_id}_{system_specific_tag}_{prot_template.get_keys()}.npy"
            filename = filename.replace("/", "_")
            filename = filename.replace("\\", "_")

            template_file = self._work_dir / str(template.subject_id) / filename
            template_file.unlink(missing_ok=True)

    def protect(
        self,
        template: Template,
        key: int | list[int] = None,
    ) -> ProtectedTemplate:
        """Perform feature extraction with backup management.

        Template's metadata are automatically passed to ProtectedTemplate's
        metadata.
        """

        if template.get_template() is None:
            raise InvalidInputSampleError("No unprotected template to protect")

        if key is None:
            # If no key is specified, we use the subject id
            # If BTP is system specific we always use the same key

            if self._system_specific:
                key = self._key
            else:
                # Mean we are user specific
                if self._key_dictionary is None:
                    # If no key dictionary, use subject id as key
                    key = int(template.subject_id) + self._key_offset
                elif str(template.subject_id) in self._key_dictionary:
                    key = (
                        int(self._key_dictionary[str(template.subject_id)])
                        + self._key_offset
                    )
                    logger.info(
                        f"Using key {key} from key dictionary for subject {template.subject_id}."
                    )
                else:
                    raise KeyError(
                        f"Subject id {template.subject_id} not found in key dictionary."
                    )
        else:
            logger.info(
                f"Using provided key {key} to protect subject {template.subject_id}"
            )

        exist = False
        key_tag = (
            str(key)
            if isinstance(key, (int, numpy.integer))
            else "_".join(map(str, key))
        )

        if self._save:
            system_specific_tag = "sys" if self._system_specific else "usr"

            filename = f"{template.subject_id}_{template.template_id}_{system_specific_tag}_{key_tag}.npy"
            filename = filename.replace("/", "_")
            filename = filename.replace("\\", "_")

            template_file = self._work_dir / str(template.subject_id) / filename
            exist, prot_template = ProtectedTemplate.load(
                template.subject_id, template.template_id, template_file
            )

            if exist:
                prot_template.metadata = template.metadata

        if not exist:
            feature_vector = template.get_template()

            if self._normalize_input:
                feature_vector = feature_vector / numpy.linalg.norm(feature_vector)

            prot_template_vector = self._protect_postprocessed(feature_vector, key)

            prot_template = ProtectedTemplate(
                template.subject_id,
                template.template_id + f"_k{key_tag}",
                prot_template_vector,
                key,
            )
            prot_template.metadata = template.metadata

            if self._save:
                if not template_file.parent.exists():
                    template_file.parent.mkdir(exist_ok=True)

                prot_template.save(template_file)

        return prot_template

    def _invert_alg_lsq(
        self,
        feature_vector: numpy.ndarray,
        residuals: numpy.ndarray,
        key: int | list[int],
        cosine: bool = True,
    ):
        if self._normalize_input:
            feature_vector = feature_vector / numpy.linalg.norm(feature_vector)

        protected_feature_vector = self._protect_postprocessed(feature_vector, key)

        if cosine:
            res_normalized = residuals / numpy.linalg.norm(residuals)

            prot_fv_norm = numpy.linalg.norm(protected_feature_vector)

            if prot_fv_norm == 0:
                return 1e3 * res_normalized  # Penalize zero vectors

            prot_fv_normalized = protected_feature_vector / prot_fv_norm

            return prot_fv_normalized - res_normalized
        return protected_feature_vector - residuals

    def _invert_alg_minimize(
        self,
        feature_vector: numpy.ndarray,
        residuals: numpy.ndarray,
        key: int | list[int],
        cosine: bool = True,
        hamming: bool = False,
    ):
        if self._normalize_input:
            feature_vector = feature_vector / numpy.linalg.norm(feature_vector)

        protected_feature_vector = self._protect_postprocessed(feature_vector, key)

        if hamming:
            protected_feature_vector = protected_feature_vector.reshape((1, -1))
            residuals = residuals.reshape((1, -1))

            return float(
                cdist(protected_feature_vector, residuals, metric="hamming").squeeze()
            )

        if cosine:
            protected_feature_vector = protected_feature_vector.reshape((1, -1))
            residuals = residuals.reshape((1, -1))

            return float(
                cdist(protected_feature_vector, residuals, metric="cosine").squeeze()
            )
        return numpy.linalg.norm(protected_feature_vector - residuals) ** 2

    def _invert_alg_root(
        self,
        feature_vector: numpy.ndarray,
        residuals: numpy.ndarray,
        key: int | list[int],
        cosine: bool = True,
        hamming: bool = False,
    ):
        if self._normalize_input:
            feature_vector = feature_vector / numpy.linalg.norm(feature_vector)

        protected_feature_vector = self._protect_postprocessed(feature_vector, key)

        result = protected_feature_vector - residuals
        if hamming:
            result = numpy.not_equal(protected_feature_vector, residuals).astype(float)
        elif cosine:
            res_normalized = residuals / numpy.linalg.norm(residuals)

            prot_fv_norm = numpy.linalg.norm(protected_feature_vector)

            if prot_fv_norm == 0:
                return 1e3 * res_normalized  # Penalize zero vectors

            prot_fv_normalized = protected_feature_vector / prot_fv_norm
            result = prot_fv_normalized - res_normalized

        # Scipy's root optimizer requires the same dimensions at input and output
        # With PolyProtect we know that protected is at best shorter than unprotected.
        return numpy.hstack(
            (
                result,
                numpy.zeros(len(feature_vector) - len(result)),
            )
        )

    def _get_cmaes_options(self, seed: int | None = None) -> dict[str, float | int]:
        options: dict[str, float | int] = {
            "ftarget": 0.0,
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
        }
        if self._cma_maxiter is not None:
            options["maxiter"] = int(self._cma_maxiter)
        if self._cma_popsize is not None:
            options["popsize"] = int(self._cma_popsize)
        effective_seed = self._cma_seed if seed is None else seed
        if effective_seed is not None:
            options["seed"] = int(effective_seed)
        return options

    def _invert_alg_cmaes(
        self,
        protected_template: numpy.ndarray,
        key: int | list[int],
        initial_guess: numpy.ndarray,
        seed: int | None = None,
    ) -> numpy.ndarray | None:
        objective = partial(
            self._invert_alg_minimize,
            key=key,
            residuals=protected_template,
            cosine="cos" in self._method,
            hamming="hamming" in self._method,
        )
        inverted_template, optimizer = cma.fmin2(
            objective,
            initial_guess,
            self._cma_sigma,
            options=self._get_cmaes_options(seed),
        )

        if not numpy.isfinite(optimizer.result.fbest):
            return None
        return numpy.asarray(inverted_template)

    def _invert_alg_surrogate(
        self,
        protected_template: numpy.ndarray,
        key: int | list[int],
        initial_guess: numpy.ndarray,
    ) -> numpy.ndarray | None:
        result = minimize(
            self._surrogate_loss_and_gradient,
            initial_guess,
            args=(protected_template, key, initial_guess),
            method="L-BFGS-B",
            jac=True,
            options={"ftol": 1e-9, "maxiter": self._surrogate_maxiter},
        )
        if not numpy.isfinite(result.fun):
            return None
        return result.x

    def _surrogate_loss_and_gradient(
        self,
        feature_vector: numpy.ndarray,
        protected_template: numpy.ndarray,
        key: int | list[int],
        initial_guess: numpy.ndarray,
    ) -> tuple[float, numpy.ndarray]:
        target_signs = self._target_signs(protected_template)
        centered_scores, score_jacobian = (
            self._centered_binary_scores_and_input_jacobian(feature_vector, key)
        )
        self._validate_binary_target_shape(target_signs, centered_scores)

        scaled_negative_margins = (
            -target_signs * centered_scores / self._surrogate_temperature
        )

        loss = numpy.mean(numpy.logaddexp(0.0, scaled_negative_margins))
        score_gradient = (
            -target_signs
            * expit(scaled_negative_margins)
            / (len(target_signs) * self._surrogate_temperature)
        )
        gradient = score_jacobian.T @ score_gradient

        if self._surrogate_regularization:
            difference = feature_vector - initial_guess
            loss += self._surrogate_regularization * numpy.mean(difference**2)
            gradient += (
                2.0 * self._surrogate_regularization * difference / len(feature_vector)
            )

        return float(loss), gradient

    @staticmethod
    def _hard_constraint_objective_and_gradient(
        feature_vector: numpy.ndarray,
        initial_guess: numpy.ndarray,
    ) -> tuple[float, numpy.ndarray]:
        difference = feature_vector - initial_guess
        return 0.5 * float(numpy.dot(difference, difference)), difference

    def _hard_constraint_values(
        self,
        feature_vector: numpy.ndarray,
        protected_template: numpy.ndarray,
        key: int | list[int],
    ) -> numpy.ndarray:
        target_signs = self._target_signs(protected_template)
        centered_scores, _ = self._centered_binary_scores_and_input_jacobian(
            feature_vector,
            key,
        )
        self._validate_binary_target_shape(target_signs, centered_scores)
        return target_signs * centered_scores - self._hard_constraint_epsilon

    def _hard_constraint_jacobian(
        self,
        feature_vector: numpy.ndarray,
        protected_template: numpy.ndarray,
        key: int | list[int],
    ) -> numpy.ndarray:
        target_signs = self._target_signs(protected_template)
        centered_scores, score_jacobian = (
            self._centered_binary_scores_and_input_jacobian(feature_vector, key)
        )
        self._validate_binary_target_shape(target_signs, centered_scores)
        return target_signs[:, numpy.newaxis] * score_jacobian

    def _invert_alg_hard_constraints(
        self,
        protected_template: numpy.ndarray,
        key: int | list[int],
        initial_guess: numpy.ndarray,
    ) -> numpy.ndarray | None:
        result = minimize(
            self._hard_constraint_objective_and_gradient,
            initial_guess,
            args=(initial_guess,),
            method="SLSQP",
            jac=True,
            constraints={
                "type": "ineq",
                "fun": lambda feature_vector: self._hard_constraint_values(
                    feature_vector,
                    protected_template,
                    key,
                ),
                "jac": lambda feature_vector: self._hard_constraint_jacobian(
                    feature_vector,
                    protected_template,
                    key,
                ),
            },
            options={
                "ftol": self._hard_constraint_ftol,
                "maxiter": self._hard_constraint_maxiter,
            },
        )
        if not numpy.isfinite(result.fun):
            return None

        constraint_values = self._hard_constraint_values(
            result.x,
            protected_template,
            key,
        )
        tolerance = max(1e-8, self._hard_constraint_epsilon * 1e-3)
        if numpy.min(constraint_values) < -tolerance:
            return None

        return result.x

    def _invert_atomic(
        self,
        protected_template: numpy.ndarray,
        key: int | list[int],
        initial_guess: numpy.ndarray,
        seed: int | None = None,
    ) -> numpy.ndarray | None:
        if self._method.startswith("cmaes"):
            return self._invert_alg_cmaes(
                protected_template,
                key,
                initial_guess,
                seed,
            )

        if self._method == "minimize_hard_constraints":
            return self._invert_alg_hard_constraints(
                protected_template,
                key,
                initial_guess,
            )

        if self._method == "minimize_surrogate":
            return self._invert_alg_surrogate(
                protected_template,
                key,
                initial_guess,
            )

        if "root" in self._method:
            invert_f = partial(
                self._invert_alg_root,
                key=key,
                residuals=protected_template,
                cosine="cos" in self._method,
                hamming="hamming" in self._method,
            )
            solution = root(
                invert_f, initial_guess, method="lm", options={"ftol": 0.001}
            )
        elif "minimize" in self._method:
            invert_f = partial(
                self._invert_alg_minimize,
                key=key,
                residuals=protected_template,
                cosine="cos" in self._method,
                hamming="hamming" in self._method,
            )
            solution = minimize(
                invert_f,
                initial_guess,
                method="L-BFGS-B",
                options={"ftol": 0.001},
            )
        elif "lsq" in self._method:
            invert_f = partial(
                self._invert_alg_lsq,
                key=key,
                residuals=protected_template,
                cosine="cos" in self._method,
            )
            solution = least_squares(
                invert_f,
                initial_guess,
                method="trf",
                ftol=0.001,
                xtol=0.001,
                gtol=0.001,
            )
        else:
            raise ValueError(f"Unknown method {self._method} for inversion.")

        if solution.success:
            return solution.x

        return None

    def _invert(
        self,
        protected_template: numpy.ndarray,
        key: int | list[int],
        template_dist: Distribution,
        seed: int | None = None,
    ) -> numpy.ndarray | None:
        """Invert protected template."""

        guesses = generate_guesses_from_distribution(
            template_dist,
            self._num_guesses,
            seed,
        )
        best_surrogate_template = None
        best_hamming_distance = numpy.inf

        for guess_idx in range(self._num_guesses):
            guess_set = guesses[guess_idx]
            optimizer_seed = (
                derive_seed(seed, "guess", guess_idx) if seed is not None else None
            )

            inverted_template = self._invert_atomic(
                protected_template,
                key,
                guess_set,
                optimizer_seed,
            )

            if inverted_template is not None:
                if self._method == "minimize_surrogate":
                    hamming_distance = self._invert_alg_minimize(
                        inverted_template,
                        protected_template,
                        key,
                        cosine=False,
                        hamming=True,
                    )
                    if hamming_distance < best_hamming_distance:
                        best_hamming_distance = hamming_distance
                        best_surrogate_template = inverted_template
                    if hamming_distance > 0:
                        continue
                return inverted_template

            logger.warning(f"Inversion failed for guess {guess_idx}")

        if best_surrogate_template is not None:
            return best_surrogate_template

        logger.warning("Inversion failed for all guesses")
        return None

    @abstractmethod
    def _protect(
        self, feature_vector: numpy.ndarray, key: int | list[int]
    ) -> numpy.ndarray:
        """Implement protection algorithm and return the full protected template."""
        pass

    @abstractmethod
    def get_alg_name(self) -> str:
        """Return algorithm name."""
        pass

    def get_key_selection_tag(self) -> str:
        """Return a string representing the key selection configuration."""
        return f"{self._ks_method}_{self._ks_n_elements}"

    def get_inversion_config_tag(self) -> str:
        """Return a string representing the inversion configuration."""
        tag = f"{self._method}-{self._precision}-{self._num_guesses}"
        if self._method.startswith("cmaes"):
            tag += (
                f"-sigma{self._cma_sigma}"
                f"-iter{self._cma_maxiter}"
                f"-pop{self._cma_popsize}"
                f"-seed{self._cma_seed}"
            )
        elif self._method == "minimize_surrogate":
            tag += (
                f"-temp{self._surrogate_temperature}"
                f"-reg{self._surrogate_regularization}"
                f"-iter{self._surrogate_maxiter}"
            )
        elif self._method == "minimize_hard_constraints":
            tag += (
                f"-eps{self._hard_constraint_epsilon}"
                f"-iter{self._hard_constraint_maxiter}"
                f"-ftol{self._hard_constraint_ftol}"
            )
        return tag

    def get_inversion_config(self) -> dict[str, Any]:
        """Return a dict representing the inversion configuration."""
        config = {
            "method": self._method,
            "num_guesses": self._num_guesses,
            "precision": self._precision,
        }
        if self._method.startswith("cmaes"):
            config.update(
                {
                    "cma_sigma": self._cma_sigma,
                    "cma_maxiter": self._cma_maxiter,
                    "cma_popsize": self._cma_popsize,
                    "cma_seed": self._cma_seed,
                }
            )
        elif self._method == "minimize_surrogate":
            config.update(
                {
                    "surrogate_temperature": self._surrogate_temperature,
                    "surrogate_regularization": self._surrogate_regularization,
                    "surrogate_maxiter": self._surrogate_maxiter,
                }
            )
        elif self._method == "minimize_hard_constraints":
            config.update(
                {
                    "hard_constraint_epsilon": self._hard_constraint_epsilon,
                    "hard_constraint_maxiter": self._hard_constraint_maxiter,
                    "hard_constraint_ftol": self._hard_constraint_ftol,
                }
            )
        return config

    def compare(self, ref: ProtectedTemplate, probe: ProtectedTemplate) -> float:
        """Compute distance. Defaults to cosine, or Hamming for binary templates."""

        ref_template = ref.get_template().reshape((1, -1))
        probe_template = probe.get_template().reshape((1, -1))
        metric = "hamming" if self._compare_hamming else "cosine"

        return -1.0 * cdist(ref_template, probe_template, metric=metric).squeeze()
