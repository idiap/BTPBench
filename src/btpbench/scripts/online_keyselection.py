# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

"""Shared support for online, per-sample user key assignment."""

import hashlib
import json
import logging
import math

from collections.abc import Callable, Collection, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import bob.measure
import click
import numpy
import pandas

from btpbench.algorithms import get_baseline_dict, get_protected_baseline_dict
from btpbench.baselines import Template
from btpbench.btps import ProtectedTemplate, _load_candidate_keys
from btpbench.scripts import pipeline_utils
from btpbench.scripts.unlinkability.pipeline import (
    _group_by_subject,
    _select_samples_per_subject,
)
from btpbench.utils import (
    Distribution,
    derive_seed,
    templates_matrix_distribution,
    templates_to_matrix,
)

pipeline_utils.setup_logger()
logger = logging.getLogger(__name__)

_KEY_SPACE_SIZE = 2_000_000
_WORKER: tuple[Any, ...] | None = None


def common_options(command: Callable) -> Callable:
    """Attach the options shared by online key-assignment commands."""
    options = [
        click.option(
            "-s",
            "--system-conf",
            "system_config_file",
            required=True,
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="System configuration file.",
        ),
        click.option(
            "-e",
            "--exp-conf",
            "exp_config_file",
            required=True,
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="Experiment configuration file.",
        ),
        click.option(
            "-k",
            "--keys-file",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help=(
                "Optional candidate-key pool as a JSON integer list, JSON "
                "mapping, or text file containing integers."
            ),
        ),
        click.option(
            "-v",
            "-d",
            "--verification-file",
            "--dedup-file",
            "verification_file",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help=(
                "Unprotected comparison scores used to calibrate the selected-key "
                "threshold. Required unless --random is used."
            ),
        ),
        click.option(
            "-f",
            "--ks-fmr",
            type=click.FloatRange(0, 1, min_open=True, max_open=True),
            default=0.05,
            show_default=True,
            help="Key-selection FMR as a fraction (0.05 means 5%).",
        ),
        click.option(
            "--samples-per-subject",
            type=click.IntRange(min=2),
            default=60,
            show_default=True,
            help="Number of samples and distinct keys to use for each subject.",
        ),
        click.option(
            "--n-subjects",
            type=int,
            default=-1,
            show_default=True,
            help="Use the first N subjects, or -1 for all subjects.",
        ),
        click.option(
            "--random",
            "random_keys",
            is_flag=True,
            help="Assign distinct random keys without inversion-based selection.",
        ),
        click.option(
            "-o",
            "--output-dir",
            type=click.Path(file_okay=False, path_type=Path),
            help="Output directory; defaults to output_dir in the experiment config.",
        ),
        click.option(
            "--override",
            is_flag=True,
            help="Replace existing online outputs.",
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def fmr_threshold(path: Path, fmr: float) -> float:
    """Calibrate key selection on valid unprotected non-mated scores."""
    negatives = []
    columns = {
        "probe_subject_id": str,
        "bio_ref_subject_id": str,
        "score": "float64",
    }
    try:
        for frame in pandas.read_csv(
            path,
            usecols=list(columns),
            dtype=columns,
            chunksize=1_000_000,
        ):
            subject_ids = frame[["probe_subject_id", "bio_ref_subject_id"]]
            if subject_ids.isna().any().any() or any(
                subject_ids[column].str.strip().eq("").any() for column in subject_ids
            ):
                raise ValueError("missing subject IDs")
            if numpy.isinf(frame.score).any():
                raise ValueError("infinite scores")
            non_mated = frame.probe_subject_id != frame.bio_ref_subject_id
            negatives.append(frame.loc[non_mated, "score"].dropna().to_numpy())
    except (ValueError, OSError) as exc:
        raise click.ClickException(
            f"Cannot calibrate key selection from {path}: {exc}"
        ) from exc

    negative = numpy.concatenate(negatives) if negatives else numpy.array([])
    if len(negative) < 2:
        raise click.ClickException(
            f"Need at least two valid non-mated scores in {path}."
        )
    return float(bob.measure.far_threshold(negative, [], fmr))


@dataclass(frozen=True)
class OnlineRun:
    """Validated configuration and algorithms for an online experiment."""

    output_dir: Path
    database: str
    dataset: Any
    baseline: str
    bio_alg_cls: type
    bio_alg_args: tuple
    alg: Any
    alg_cls: type
    alg_args: tuple
    config: dict[str, Any]
    keys_file: Path | None
    keys: tuple[int, ...] | None
    pool_hash: str | None
    verification_file: Path | None
    verification_hash: str | None
    system_config_hash: str
    experiment_config_hash: str
    verification_samples_hash: str
    unlinkability_samples_hash: str
    dataset_dir: str
    detector: str
    ks_fmr: float
    threshold: float | None
    key_sampling_seed: int | None
    random_keys: bool
    n_workers: int
    compliant: bool

    @property
    def assignment(self) -> str:
        """Return a stable label for the key-assignment condition."""
        return "random" if self.random_keys else "selected"

    @property
    def pool_tag(self) -> str:
        """Return the candidate-pool tag used in output names."""
        return "randomspace" if self.pool_hash is None else self.pool_hash[:12]

    def stem(self, samples_per_subject: int, n_subjects: int) -> str:
        """Identify the algorithm, key assignment, pool, and sample count."""
        subject_tag = "" if n_subjects == -1 else f"-subjects{n_subjects}"
        seed_tag = "none" if self.key_sampling_seed is None else self.key_sampling_seed
        condition = self.assignment
        if not self.random_keys:
            fmr_tag = str(self.ks_fmr).replace(".", "d")
            condition += f"-ksfmr{fmr_tag}"
        return (
            f"{self.database}-{self.alg.get_alg_name()}-{self.baseline}-{condition}"
            f"-pool{self.pool_tag}-keyseed{seed_tag}"
            f"-{self.alg.get_inversion_config_tag()}-n{samples_per_subject}{subject_tag}"
        )

    def metadata(self, samples_per_subject: int, n_subjects: int) -> dict[str, Any]:
        """Describe settings that determine the online key assignments."""
        return {
            "schema_version": 1,
            "database": self.database,
            "baseline": self.baseline,
            "detector": self.detector,
            "btp_config": self.config,
            "assignment": self.assignment,
            "ks_fmr": None if self.random_keys else self.ks_fmr,
            "threshold": self.threshold,
            "key_sampling_seed": self.key_sampling_seed,
            "candidate_pool_sha256": self.pool_hash,
            "candidate_count": (
                _KEY_SPACE_SIZE if self.keys is None else len(self.keys)
            ),
            "samples_per_subject": samples_per_subject,
            "n_subjects_requested": n_subjects,
            "reference_sample_set": (None if self.random_keys else "verification"),
            "verification_scores_sha256": self.verification_hash,
            "system_config_sha256": self.system_config_hash,
            "experiment_config_sha256": self.experiment_config_hash,
            "verification_samples_sha256": self.verification_samples_hash,
            "unlinkability_samples_sha256": self.unlinkability_samples_hash,
            "dataset_dir": self.dataset_dir,
        }


def prepare_run(
    system_config_file: Path,
    exp_config_file: Path,
    keys_file: Path | None,
    verification_file: Path | None,
    ks_fmr: float,
    samples_per_subject: int,
    n_subjects: int,
    output_dir: Path | None,
    random_keys: bool,
) -> OnlineRun:
    """Validate online inputs before feature extraction or key assignment."""
    if not math.isfinite(ks_fmr) or not 0 < ks_fmr < 1:
        raise click.BadParameter(
            "Key-selection FMR must be a finite fraction in (0, 1).",
            param_hint="--ks-fmr",
        )
    if n_subjects != -1 and n_subjects < 1:
        raise click.BadParameter(
            "Use -1 or a positive number.",
            param_hint="--n-subjects",
        )
    if not random_keys and verification_file is None:
        raise click.UsageError(
            "-v/--verification-file is required unless --random is used."
        )

    keys = None
    pool_hash = None
    if keys_file is not None:
        try:
            keys = _load_candidate_keys(keys_file)
        except (ValueError, OSError) as exc:
            raise click.ClickException(str(exc)) from exc
        if len(keys) < samples_per_subject:
            raise click.ClickException(
                f"Need {samples_per_subject} distinct candidate keys per subject; "
                f"the pool has {len(keys)}."
            )
        pool_hash = hashlib.sha256(keys_file.read_bytes()).hexdigest()

    system, experiment = pipeline_utils.load_config(
        system_config_file,
        exp_config_file,
    )
    baseline, has_btp, save, compliant, detector, n_workers = (
        pipeline_utils.load_common_parameters(system, experiment)
    )
    btps_config = experiment.get("btps")
    configs = btps_config.get("algs") if isinstance(btps_config, dict) else None
    if (
        not has_btp
        or not isinstance(configs, list)
        or len(configs) != 1
        or not isinstance(configs[0], dict)
    ):
        raise click.ClickException(
            "Configure btps.algs as a list containing exactly one BTP algorithm."
        )

    config = dict(configs[0])
    if not isinstance(config.get("type"), str) or not config["type"].strip():
        raise click.ClickException("The online BTP configuration needs a type.")
    if config.get("type") in {"combined", "chained", "cumulated"}:
        raise click.ClickException(
            "Online key assignment currently supports single-key BTPs such as "
            "PolyProtect."
        )
    if config.get("system_specific", False):
        raise click.ClickException(
            "Online key assignment requires system_specific: false."
        )
    if not random_keys and config.get("ks_method", "legacy") != "legacy":
        raise click.ClickException(
            "Online selected-key assignment requires ks_method: legacy."
        )
    if config.get("key_dictionary_file"):
        raise click.ClickException(
            "Online key assignment supplies keys directly; remove key_dictionary_file."
        )

    pipeline_utils.validate_detector(detector)
    if not isinstance(n_workers, int) or isinstance(n_workers, bool) or n_workers < 1:
        raise click.ClickException("num_processes must be a positive integer.")

    baselines = get_baseline_dict()
    btps = get_protected_baseline_dict()
    if baseline not in baselines:
        raise click.ClickException(f"Unknown baseline: {baseline}")
    if config.get("type") not in btps:
        raise click.ClickException(f"Unknown BTP algorithm: {config.get('type')}")

    threshold = None
    verification_hash = None
    if not random_keys:
        assert verification_file is not None
        threshold = fmr_threshold(verification_file, ks_fmr)
        verification_hash = hashlib.sha256(verification_file.read_bytes()).hexdigest()

    output_dir = output_dir or Path(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    database, dataset = pipeline_utils.create_dataset(system, experiment)
    database_config = system["databases"][database]
    verification_samples_file = Path(database_config["verification_samples"])
    unlinkability_samples_file = Path(database_config["unlink_samples"])

    # Candidate protection must not reuse cached results from an earlier search.
    alg, alg_cls, alg_args, _ = pipeline_utils.build_btp_alg(
        config,
        btps,
        output_dir / "online_keyselection",
        baseline,
        False,
    )
    return OnlineRun(
        output_dir=output_dir,
        database=database,
        dataset=dataset,
        baseline=baseline,
        bio_alg_cls=baselines[baseline],
        bio_alg_args=(output_dir / baseline, save, detector),
        alg=alg,
        alg_cls=alg_cls,
        alg_args=alg_args,
        config=config,
        keys_file=None if keys_file is None else keys_file.resolve(),
        keys=keys,
        pool_hash=pool_hash,
        verification_file=(
            None
            if random_keys or verification_file is None
            else verification_file.resolve()
        ),
        verification_hash=verification_hash,
        system_config_hash=hashlib.sha256(system_config_file.read_bytes()).hexdigest(),
        experiment_config_hash=hashlib.sha256(exp_config_file.read_bytes()).hexdigest(),
        verification_samples_hash=hashlib.sha256(
            verification_samples_file.read_bytes()
        ).hexdigest(),
        unlinkability_samples_hash=hashlib.sha256(
            unlinkability_samples_file.read_bytes()
        ).hexdigest(),
        dataset_dir=str(Path(database_config["dataset_dir"]).resolve()),
        detector=detector,
        ks_fmr=ks_fmr,
        threshold=threshold,
        key_sampling_seed=pipeline_utils.key_sampling_seed(experiment),
        random_keys=random_keys,
        n_workers=n_workers,
        compliant=compliant,
    )


def load_templates(
    run: OnlineRun,
    samples_per_subject: int,
    n_subjects: int,
) -> tuple[
    dict[Any, list[Template]],
    Distribution | None,
    Callable[[Template, Template], float] | None,
]:
    """Extract unlinkability samples and selection reference templates."""
    samples = run.dataset.samples(verification=False)
    if n_subjects != -1:
        kept = list(dict.fromkeys(sample.subject_id for sample in samples))[:n_subjects]
        samples = [sample for sample in samples if sample.subject_id in kept]
    try:
        grouped = _select_samples_per_subject(samples, samples_per_subject)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not grouped:
        raise click.ClickException("No unlinkability samples are available.")
    for subject_id, subject_samples in grouped.items():
        template_ids = [sample.template_id for sample in subject_samples]
        if len(template_ids) != len(set(template_ids)):
            raise click.ClickException(
                f"Subject {subject_id} has duplicate template IDs in the selected "
                "unlinkability samples."
            )

    selected = [sample for group in grouped.values() for sample in group]
    logger.info("Extracting %d unlinkability templates", len(selected))
    templates = pipeline_utils.threaded_feature_extraction(
        selected,
        run.bio_alg_cls,
        run.bio_alg_args,
        run.compliant,
        run.n_workers,
        run.n_workers * 100,
    )
    if len(templates) != len(selected) or any(
        template.get_template() is None for template in templates
    ):
        raise click.ClickException(
            "Online key assignment needs a valid feature template for every "
            "selected sample."
        )

    by_subject = _group_by_subject(templates)
    if set(by_subject) != set(grouped) or any(
        len(group) != samples_per_subject for group in by_subject.values()
    ):
        raise click.ClickException(
            "Feature extraction changed the selected per-subject sample counts."
        )

    if run.random_keys:
        return by_subject, None, None

    logger.info("Extracting verification templates for the inversion distribution")
    reference = pipeline_utils.threaded_feature_extraction(
        run.dataset.samples(verification=True),
        run.bio_alg_cls,
        run.bio_alg_args,
        run.compliant,
        run.n_workers,
        run.n_workers * 100,
    )
    if not any(template.get_template() is not None for template in reference):
        raise click.ClickException(
            "No valid verification templates are available for inversion."
        )
    matrix, _ = templates_to_matrix(
        reference,
        run.alg.get_inversion_config()["precision"],
    )
    distribution = templates_matrix_distribution(matrix)
    compare = run.bio_alg_cls(*run.bio_alg_args).compare
    return by_subject, distribution, compare


def _init_worker(
    alg_cls: type,
    alg_args: tuple,
    keys: tuple[int, ...] | None,
    distribution: Distribution | None,
    threshold: float | None,
    compare: Callable[[Template, Template], float] | None,
    key_sampling_seed: int | None,
    random_keys: bool,
) -> None:
    """Create an isolated BTP instance per online-assignment process."""
    global _WORKER
    _WORKER = (
        alg_cls(*alg_args),
        keys,
        distribution,
        threshold,
        compare,
        key_sampling_seed,
        random_keys,
    )


def _random_key(
    candidates: Sequence[int] | None,
    excluded: Collection[int],
    seed: int | None,
) -> int:
    """Draw one random key while excluding keys already used by the subject."""
    rng = numpy.random.default_rng(seed)
    if candidates is not None:
        available = [key for key in candidates if key not in excluded]
        if not available:
            raise RuntimeError("Candidate-key pool exhausted.")
        return int(available[int(rng.integers(0, len(available)))])

    if len(excluded) >= _KEY_SPACE_SIZE:
        raise RuntimeError("Candidate-key space exhausted.")
    while True:
        key = int(rng.integers(0, _KEY_SPACE_SIZE))
        if key not in excluded:
            return key


def _selection_seed(
    base_seed: int | None,
    assignment: str,
    template: Template,
) -> int | None:
    """Derive a stable seed for one sample's independent key search."""
    if base_seed is None:
        return None
    return derive_seed(
        base_seed,
        f"online-{assignment}",
        template.subject_id,
        template.template_id,
    )


def _select_subject(
    task: tuple[Any, list[Template]],
) -> tuple[Any, list[ProtectedTemplate], list[dict[str, Any]]]:
    """Assign a distinct key to every sample of one subject in sample order."""
    if _WORKER is None:
        raise RuntimeError("Online key-assignment worker is not initialized.")
    (
        alg,
        keys,
        distribution,
        threshold,
        compare,
        key_sampling_seed,
        random_keys,
    ) = _WORKER
    subject_id, templates = task
    used: set[int] = set()
    protected: list[ProtectedTemplate] = []
    audit: list[dict[str, Any]] = []
    assignment = "random" if random_keys else "selected"

    for template in templates:
        seed = _selection_seed(key_sampling_seed, assignment, template)
        started = perf_counter()
        try:
            if random_keys:
                key = _random_key(keys, used, seed)
                result = alg.protect(template, key=key)
                score = None
                n_trials = 1
            else:
                if distribution is None or threshold is None or compare is None:
                    raise RuntimeError(
                        "Selected-key worker is missing inversion inputs."
                    )
                result, _, raw_score, n_trials = alg.key_selection_usr_with_stats(
                    template,
                    distribution,
                    threshold,
                    compare,
                    seed=seed,
                    candidate_keys=keys,
                    excluded_keys=used,
                )
                key = int(result.get_keys())
                score = float(raw_score) if math.isfinite(raw_score) else None
        except RuntimeError as exc:
            raise RuntimeError(
                f"Online {assignment} key assignment failed for subject "
                f"{subject_id}, template {template.template_id}, after assigning "
                f"{len(protected)}/{len(templates)} distinct keys: {exc}"
            ) from exc

        if key in used:
            raise RuntimeError(
                f"Online {assignment} key assignment reused key {key} for subject "
                f"{subject_id}."
            )
        used.add(key)
        protected.append(result)
        audit.append(
            {
                "subject_id": subject_id,
                "template_id": template.template_id,
                "key": key,
                "inversion_score": score,
                "n_trials": n_trials,
                "elapsed_seconds": perf_counter() - started,
            }
        )

    return subject_id, protected, audit


def protect_online(
    run: OnlineRun,
    by_subject: dict[Any, list[Template]],
    distribution: Distribution | None,
    compare: Callable[[Template, Template], float] | None,
) -> tuple[dict[Any, list[ProtectedTemplate]], list[dict[str, Any]]]:
    """Assign keys in parallel across subjects while preserving sample order."""
    tasks = list(by_subject.items())
    initargs = (
        run.alg_cls,
        run.alg_args,
        run.keys,
        distribution,
        run.threshold,
        compare,
        run.key_sampling_seed,
        run.random_keys,
    )
    protected = {}
    audit = []
    logger.info(
        "Online %s assignment: FMR=%s, threshold=%s, %d candidate keys",
        run.assignment,
        None if run.random_keys else run.ks_fmr,
        run.threshold,
        _KEY_SPACE_SIZE if run.keys is None else len(run.keys),
    )
    try:
        if run.n_workers == 1:
            _init_worker(*initargs)
            results = map(_select_subject, tasks)
            for subject, templates, rows in results:
                protected[subject] = templates
                audit.extend(rows)
                logger.info(
                    "Assigned %d distinct keys for subject %s",
                    len(templates),
                    subject,
                )
        else:
            with ProcessPoolExecutor(
                max_workers=min(run.n_workers, len(tasks)),
                initializer=_init_worker,
                initargs=initargs,
            ) as executor:
                for subject, templates, rows in executor.map(
                    _select_subject,
                    tasks,
                    chunksize=1,
                ):
                    protected[subject] = templates
                    audit.extend(rows)
                    logger.info(
                        "Assigned %d distinct keys for subject %s",
                        len(templates),
                        subject,
                    )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    return protected, audit


def completed_outputs(
    paths: list[Path],
    override: bool,
    expected_metadata: dict[str, Any],
) -> bool:
    """Skip a complete result or require override for incomplete prior output."""
    if override:
        return False
    if all(path.is_file() for path in paths):
        try:
            existing = json.loads(paths[-1].read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise click.ClickException(
                "Cannot read existing online metadata; use --override."
            ) from exc
        if not isinstance(existing, dict) or any(
            existing.get(key) != value for key, value in expected_metadata.items()
        ):
            raise click.ClickException(
                "Existing online outputs use different settings; use --override."
            )
        logger.info("Skipping completed online result: %s", paths[0])
        return True
    if any(path.exists() for path in paths):
        raise click.ClickException(
            "Incomplete online outputs exist; use --override to regenerate them."
        )
    return False


def write_metadata(path: Path, metadata: dict[str, Any], audit: list[dict]) -> None:
    """Atomically publish provenance and accepted per-sample keys."""
    temporary = path.with_suffix(path.suffix + ".temporary")
    temporary.write_text(
        json.dumps({**metadata, "selections": audit}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)
