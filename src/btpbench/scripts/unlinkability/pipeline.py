# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import json
import logging

from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, islice, product
from pathlib import Path
from typing import Any, Protocol

import click
import numpy

from btpbench.algorithms import get_baseline_dict, get_protected_baseline_dict
from btpbench.btps import ProtectedTemplate
from btpbench.dataloader import Sample
from btpbench.scorewriter import CSVScoreWriter
from btpbench.scripts import pipeline_utils
from btpbench.scripts.workers import BTPWorker

pipeline_utils.setup_logger()
logger = logging.getLogger(__name__)


class _HasSubjectId(Protocol):
    subject_id: Any


def _chunked[T](items: Iterable[T], chunk_size: int) -> Iterator[list[T]]:
    """Yield chunks from an iterable."""
    iterator = iter(items)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk


def _group_by_subject[T: _HasSubjectId](items: Iterable[T]) -> dict[Any, list[T]]:
    """Group samples or templates by subject while preserving input order."""
    grouped: dict[Any, list[T]] = {}
    for item in items:
        grouped.setdefault(item.subject_id, []).append(item)
    return grouped


def _select_samples_per_subject(
    samples: list[Sample], samples_per_subject: int
) -> dict[Any, list[Sample]]:
    """Keep the requested number of unlinkability samples per subject."""
    grouped = _group_by_subject(samples)
    too_small = {
        subject_id: len(subject_samples)
        for subject_id, subject_samples in grouped.items()
        if len(subject_samples) < samples_per_subject
    }
    if too_small:
        subject_id, available = next(iter(too_small.items()))
        raise ValueError(
            f"Requested {samples_per_subject} samples per subject, but subject "
            f"{subject_id} only has {available} unlinkability sample(s)."
        )

    return {
        subject_id: subject_samples[:samples_per_subject]
        for subject_id, subject_samples in grouped.items()
    }


def _flatten_grouped_templates(
    grouped_templates: dict[Any, list[ProtectedTemplate]],
) -> tuple[list[ProtectedTemplate], dict[Any, list[int]]]:
    """Flatten grouped templates and keep the per-subject index mapping."""
    templates: list[ProtectedTemplate] = []
    subject_indices: dict[Any, list[int]] = {}
    for subject_id, subject_templates in grouped_templates.items():
        start = len(templates)
        templates.extend(subject_templates)
        subject_indices[subject_id] = list(range(start, len(templates)))
    return templates, subject_indices


def _mated_pairs(subject_indices: dict[Any, list[int]]) -> Iterator[tuple[int, int]]:
    """Yield all same-subject protected template pairs."""
    for indices in subject_indices.values():
        yield from combinations(indices, 2)


def _non_mated_pairs(
    subject_indices: dict[Any, list[int]],
) -> Iterator[tuple[int, int]]:
    """Yield all cross-subject protected template pairs once."""
    subjects = list(subject_indices)
    for left_subject, right_subject in combinations(subjects, 2):
        yield from product(
            subject_indices[left_subject], subject_indices[right_subject]
        )


def _sample_non_mated_templates(
    subject_protected: dict[Any, list[ProtectedTemplate]],
    non_mated_samples_per_subject: int,
    rng: numpy.random.Generator,
) -> dict[Any, list[ProtectedTemplate]]:
    """Randomly sample the per-subject pool used for non-mated scores."""
    selected: dict[Any, list[ProtectedTemplate]] = {}
    for subject_id, templates in subject_protected.items():
        if len(templates) < non_mated_samples_per_subject:
            raise ValueError(
                f"Requested {non_mated_samples_per_subject} non-mated samples per "
                f"subject, but subject {subject_id} only has {len(templates)} "
                "protected template(s)."
            )

        if len(templates) == non_mated_samples_per_subject:
            selected[subject_id] = list(templates)
            continue

        selected_indices = sorted(
            rng.choice(
                len(templates),
                size=non_mated_samples_per_subject,
                replace=False,
            ).tolist()
        )
        selected[subject_id] = [templates[idx] for idx in selected_indices]
    return selected


def _load_bucket_keys(keys_file: Path, keys_bucket: str) -> list[Any]:
    """Load a key bucket from the system key-selection JSON format."""
    keys_dict = json.loads(keys_file.read_text())
    try:
        bucket = keys_dict["keys"][keys_bucket]
    except KeyError as exc:
        raise click.ClickException(
            f"Bucket {keys_bucket!r} was not found in {keys_file}."
        ) from exc

    if isinstance(bucket, dict):
        return list(bucket.values())
    if isinstance(bucket, list):
        return bucket

    raise click.ClickException(
        f"Bucket {keys_bucket!r} in {keys_file} must be a dictionary or a list."
    )


def _build_key_systems(
    bucket_keys: list[Any],
    n_systems: int,
    n_keys_per_system: int,
    rng: numpy.random.Generator,
) -> list[Any]:
    """Build key systems, using list keys for combined BTPs."""
    if n_keys_per_system == 1:
        if len(bucket_keys) < n_systems:
            raise ValueError(
                f"Bucket contains {len(bucket_keys)} key(s), but {n_systems} "
                "key system(s) were requested."
            )
        return bucket_keys[:n_systems]

    if len(bucket_keys) < n_keys_per_system:
        raise ValueError(
            f"Combined BTP needs {n_keys_per_system} key(s) per system, but "
            f"the selected bucket only contains {len(bucket_keys)} key(s)."
        )

    return [
        [
            bucket_keys[int(key_idx)]
            for key_idx in rng.choice(
                len(bucket_keys),
                size=n_keys_per_system,
                replace=False,
            )
        ]
        for _ in range(n_systems)
    ]


def _write_protected_scores(
    protected_templates: list[ProtectedTemplate],
    pairs: Iterable[tuple[int, int]],
    btp_alg_cls: type,
    btp_alg_args: tuple,
    compliant: bool,
    n_worker: int,
    batch_size: int,
    score_writer: CSVScoreWriter,
) -> None:
    """Write protected scores for explicit template index pairs."""
    if not protected_templates:
        return

    pair_batch_size = max(1, batch_size * n_worker)
    with ProcessPoolExecutor(
        max_workers=n_worker,
        initializer=BTPWorker.init,
        initargs=(
            btp_alg_cls,
            btp_alg_args,
            compliant,
            None,
            protected_templates,
            None,
        ),
    ) as executor:
        for pair_batch in _chunked(pairs, pair_batch_size):
            chunksize = pipeline_utils.get_optimal_chunksize(len(pair_batch), n_worker)
            for score, ref_template, probe_template in executor.map(
                BTPWorker.compare_verification,
                pair_batch,
                chunksize=chunksize,
            ):
                score_writer.write_score(score, ref_template, probe_template)


@click.command()
@click.option(
    "-s",
    "--system-conf",
    "system_config_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    required=True,
    help="Specify the location of the system configuration file.",
)
@click.option(
    "-e",
    "--exp-conf",
    "exp_config_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    required=True,
    help="Specify the location of the experiment configuration file.",
)
@click.option(
    "-k",
    "--keys-file",
    "keys_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    required=True,
    help="Specify the JSON file containing key buckets.",
)
@click.option(
    "-b",
    "--keys-bucket",
    "keys_bucket",
    type=str,
    required=True,
    help="Specify the bucket to use from the keys file.",
)
@click.option(
    "--samples-per-subject",
    "samples_per_subject",
    type=click.IntRange(min=1),
    default=60,
    show_default=True,
    help="Number of unlinkability samples to keep per subject.",
)
@click.option(
    "--non-mated-samples-per-subject",
    "non_mated_samples_per_subject",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Number of protected templates sampled per subject for non-mated scores.",
)
@click.option(
    "--seed",
    "seed",
    type=int,
    default=None,
    help="Optional random seed for non-mated template sampling.",
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    help="Specify the location of the output directory.",
)
@click.option("--override", is_flag=True, help="Override existing output.")
def pipeline(
    system_config_file: Path,
    exp_config_file: Path,
    keys_file: Path,
    keys_bucket: str,
    samples_per_subject: int,
    non_mated_samples_per_subject: int,
    seed: int | None,
    output_dir: Path | None,
    override: bool,
) -> None:
    """Evaluate unlinkability with mated and non-mated protected scores."""

    # --------------------
    # Config parsing
    # --------------------

    system_config, exp_config = pipeline_utils.load_config(
        system_config_file, exp_config_file
    )
    output_dir = pipeline_utils.resolve_output_dir(exp_config, output_dir)

    baseline, has_protected_part, save, compliant, detector, n_worker = (
        pipeline_utils.load_common_parameters(system_config, exp_config)
    )
    batch_size = n_worker * 100

    pipeline_utils.validate_detector(detector)
    database_name, dataset = pipeline_utils.create_dataset(system_config, exp_config)

    if non_mated_samples_per_subject > samples_per_subject:
        raise click.ClickException(
            "--non-mated-samples-per-subject must be lower than or equal to "
            "--samples-per-subject."
        )

    bucket_keys = _load_bucket_keys(keys_file, keys_bucket)

    logger.info("---Parameters---")
    logger.info("Output dir: %s", output_dir)
    logger.info("Verification baseline: %s", baseline)
    logger.info("Has protected baseline: %s", has_protected_part)
    logger.info("Save: %s", save)
    logger.info("Compliant: %s", compliant)
    logger.info("Database: %s", database_name)
    logger.info("Samples per subject: %d", samples_per_subject)
    logger.info("Non-mated samples per subject: %d", non_mated_samples_per_subject)
    logger.info("Keys bucket: %s", keys_bucket)
    logger.info("Keys in bucket: %d", len(bucket_keys))
    logger.info("Seed: %s", seed)
    logger.info("----------------")

    if not has_protected_part:
        logger.info("No BTP configured. Exiting.")
        return

    baseline_dict = get_baseline_dict()
    protected_baseline_dict = get_protected_baseline_dict()

    pipeline_utils.check_dir(output_dir, exp_config)

    if baseline not in baseline_dict:
        logger.error("No baseline: `%s`", baseline)
        return

    metadata_names = dataset.metadata_names(verification=False)

    # --------------------
    # Load unlinkability samples
    # --------------------

    logger.info("Loading unlinkability samples")
    samples = dataset.samples(verification=False)
    try:
        samples_by_subject = _select_samples_per_subject(
            samples,
            samples_per_subject,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    selected_samples = [
        sample
        for subject_samples in samples_by_subject.values()
        for sample in subject_samples
    ]
    logger.info(
        "Selected %d samples from %d subjects",
        len(selected_samples),
        len(samples_by_subject),
    )

    # --------------------
    # Feature extraction
    # --------------------

    baseline_label = baseline
    bio_alg_cls = baseline_dict[baseline]
    bio_alg_args = (output_dir / baseline_label, save, detector)

    logger.info("Feature extraction on unlinkability samples")
    unprotected_templates = pipeline_utils.threaded_feature_extraction(
        selected_samples,
        bio_alg_cls,
        bio_alg_args,
        compliant,
        n_worker,
        batch_size,
    )
    unprotected_by_subject = _group_by_subject(unprotected_templates)

    # --------------------
    # Protected pipeline
    # --------------------

    protected_baseline_configs = exp_config["btps"]["algs"]
    keys_tag = pipeline_utils.key_bucket_tag(keys_bucket)
    rng = numpy.random.default_rng(seed)

    for protected_baseline_config in protected_baseline_configs:
        logger.info("Running unlinkability for config %s", protected_baseline_config)

        btp_alg, btp_alg_cls, btp_alg_args, score_dir = pipeline_utils.build_btp_alg(
            protected_baseline_config,
            protected_baseline_dict,
            output_dir,
            baseline_label,
            save,
        )

        if btp_alg is None:
            logger.error(
                "Error building BTP algorithm for config %s. Skipping.",
                protected_baseline_config,
            )
            continue

        n_keys_per_system = pipeline_utils.get_n_keys_per_system(
            protected_baseline_config
        )
        try:
            selected_keys = _build_key_systems(
                bucket_keys,
                samples_per_subject,
                n_keys_per_system,
                rng,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        logger.info(
            "Selected %d key system(s), %d key value(s) per system",
            len(selected_keys),
            n_keys_per_system,
        )

        alg_name = btp_alg.get_alg_name()
        base_output_name = (
            f"unlinkability-{database_name}-{alg_name}-{baseline_label}"
            f"-keys{keys_tag}-{samples_per_subject}"
        )
        mated_scores_file = score_dir / f"{base_output_name}-mated.csv"
        non_mated_scores_file = (
            score_dir
            / f"{base_output_name}-non-mated-{non_mated_samples_per_subject}.csv"
        )

        existing_outputs = [
            str(path)
            for path in (mated_scores_file, non_mated_scores_file)
            if path.exists()
        ]
        if existing_outputs and not override:
            logger.warning(
                "Score file(s) already exist for %s: %s. Skipping config.",
                alg_name,
                ", ".join(existing_outputs),
            )
            continue

        # --------------------
        # Protect each same-subject sample with a different key
        # --------------------

        subject_protected: dict[Any, list[ProtectedTemplate]] = {
            subject_id: [] for subject_id in unprotected_by_subject
        }
        for key_idx, key in enumerate(selected_keys):
            batch_templates = [
                subject_templates[key_idx]
                for subject_templates in unprotected_by_subject.values()
            ]
            key_dictionary = {template.subject_id: key for template in batch_templates}

            logger.info(
                "Protecting sample %d/%d for %d subjects with key %s",
                key_idx + 1,
                len(selected_keys),
                len(batch_templates),
                key,
            )
            protected_templates = pipeline_utils.processed_protection(
                batch_templates,
                btp_alg_cls,
                btp_alg_args,
                compliant,
                n_worker,
                batch_size,
                extra_args={"key_dictionary": key_dictionary},
            )

            for protected_template in protected_templates:
                subject_protected[protected_template.subject_id].append(
                    protected_template
                )

        # --------------------
        # Mated scores
        # --------------------

        logger.info("Writing mated unlinkability scores to %s", mated_scores_file)
        mated_templates, mated_subject_indices = _flatten_grouped_templates(
            subject_protected
        )
        mated_csv = CSVScoreWriter(mated_scores_file, metadata_names)
        _write_protected_scores(
            mated_templates,
            _mated_pairs(mated_subject_indices),
            btp_alg_cls,
            btp_alg_args,
            compliant,
            n_worker,
            batch_size,
            mated_csv,
        )
        mated_csv.close()

        # --------------------
        # Non-mated scores
        # --------------------

        try:
            non_mated_subject_protected = _sample_non_mated_templates(
                subject_protected,
                non_mated_samples_per_subject,
                rng,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        logger.info(
            "Writing non-mated unlinkability scores to %s",
            non_mated_scores_file,
        )
        non_mated_templates, non_mated_subject_indices = _flatten_grouped_templates(
            non_mated_subject_protected
        )
        non_mated_csv = CSVScoreWriter(non_mated_scores_file, metadata_names)
        _write_protected_scores(
            non_mated_templates,
            _non_mated_pairs(non_mated_subject_indices),
            btp_alg_cls,
            btp_alg_args,
            compliant,
            n_worker,
            batch_size,
            non_mated_csv,
        )
        non_mated_csv.close()

    logger.info("Unlinkability experiment finished!")


if __name__ == "__main__":
    pipeline()
