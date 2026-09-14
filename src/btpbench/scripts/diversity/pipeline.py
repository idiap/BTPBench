# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import json
import logging
import math

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bob.measure
import click
import networkx
import numpy
import pandas

import btpbench.metrics

from btpbench.algorithms import get_baseline_dict, get_protected_baseline_dict
from btpbench.baselines import Template
from btpbench.scripts import pipeline_utils
from btpbench.scripts.unlinkability.metrics import (
    compute_unlinkability_metric,
    d_local_at_or_below_threshold_ranges,
    load_scores,
)
from btpbench.utils import templates_to_single_per_subject

pipeline_utils.setup_logger()
logger = logging.getLogger(__name__)


_DEFAULT_FMRS = (0.0001, 0.001, 0.01)
_DIVERSITY_BTP_ALG: Any | None = None


@dataclass(frozen=True)
class _DiversityThreshold:
    """Score criterion used to qualify protected-template comparisons."""

    tag: str
    label: str
    max_score: float | None = None
    score_ranges: tuple[tuple[float, float], ...] = ()

    def qualifies(self, scores: numpy.ndarray) -> numpy.ndarray:
        """Return comparisons accepted by this diversity criterion."""
        if not self.score_ranges:
            return scores < self.max_score

        range_matches = numpy.zeros(scores.shape, dtype=bool)
        for min_score, max_score in self.score_ranges:
            range_matches |= (scores >= min_score) & (scores <= max_score)
        return range_matches


def _group_valid_templates_by_subject(
    templates: list[Template],
) -> dict[str, list[Template]]:
    """Group extracted templates with feature data by subject."""
    grouped: dict[str, list[Template]] = {}
    for template in templates:
        if template.get_template() is not None:
            grouped.setdefault(template.subject_id, []).append(template)
    return grouped


def _select_first_subjects(items: list[Any], n_subjects: int) -> list[Any]:
    """Keep all items belonging to the first requested subject IDs."""
    if n_subjects == -1:
        return items
    if n_subjects < 1:
        raise ValueError("The number of subjects must be -1 or greater than zero.")

    selected_subject_ids = []
    seen_subject_ids = set()
    for item in items:
        if item.subject_id in seen_subject_ids:
            continue
        selected_subject_ids.append(item.subject_id)
        seen_subject_ids.add(item.subject_id)
        if len(selected_subject_ids) == n_subjects:
            break

    return [item for item in items if item.subject_id in seen_subject_ids]


def _unlinkability_protection_batches(
    templates: list[Template],
    keys: list[Any],
) -> tuple[
    dict[str, list[Template]],
    list[Any],
    list[tuple[Any, list[Template]]],
]:
    """Pair each same-subject unlinkability template with a different key."""
    templates_by_subject = _group_valid_templates_by_subject(templates)
    if not templates_by_subject:
        return templates_by_subject, [], []

    max_templates_per_subject = max(
        len(subject_templates) for subject_templates in templates_by_subject.values()
    )
    if not keys:
        raise ValueError(
            "Unlinkability diversity needs at least one key, but the selected "
            "bucket is empty."
        )

    n_selected = min(len(keys), max_templates_per_subject)
    if n_selected < max_templates_per_subject:
        logger.warning(
            "Unlinkability diversity found %d usable template(s) for at least "
            "one subject, but the selected bucket only contains %d key(s). "
            "Using the first %d template(s) per subject.",
            max_templates_per_subject,
            len(keys),
            n_selected,
        )

    selected_templates_by_subject = {
        subject_id: subject_templates[:n_selected]
        for subject_id, subject_templates in templates_by_subject.items()
    }
    selected_keys = keys[:n_selected]
    batches = [
        (
            key,
            [
                subject_templates[key_idx]
                for subject_templates in selected_templates_by_subject.values()
                if key_idx < len(subject_templates)
            ],
        )
        for key_idx, key in enumerate(selected_keys)
    ]
    return selected_templates_by_subject, selected_keys, batches


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
                f"Requested {n_systems} key system(s), but the selected bucket "
                f"only contains {len(bucket_keys)} key(s)."
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


def _fmr_thresholds(
    protected_score_file: Path | None,
    fmrs: tuple[float, ...],
) -> list[_DiversityThreshold]:
    """Compute diversity criteria from requested FMR values."""
    if not fmrs:
        return []
    if protected_score_file is None:
        raise click.ClickException(
            "-p/--protected-score-file is required when using FMR thresholds."
        )

    logger.info("Computing FMR thresholds from %s", protected_score_file)
    _, score_df = btpbench.metrics.remove_nans(pandas.read_csv(protected_score_file))
    neg, pos = btpbench.metrics.neg_pos_scores(score_df)

    thresholds = []
    for fmr in fmrs:
        max_score = bob.measure.far_threshold(neg, pos, fmr)
        thresholds.append(
            _DiversityThreshold(
                tag=f"fmr_{fmr}",
                label=f"FMR={fmr * 100:.4f}%",
                max_score=max_score,
            )
        )
        logger.info("  FMR=%.4f%% -> threshold=%.6f", fmr * 100, max_score)
    return thresholds


def _score_range_thresholds(
    score_ranges: tuple[tuple[float, float], ...],
) -> list[_DiversityThreshold]:
    """Build one inclusive criterion over all requested score ranges."""
    if not score_ranges:
        return []

    for min_score, max_score in score_ranges:
        if min_score > max_score:
            raise click.ClickException(
                "--score-range requires MIN to be lower than or equal to MAX."
            )

    formatted_ranges = [
        f"[{min_score:g}, {max_score:g}]" for min_score, max_score in score_ranges
    ]
    tag_ranges = [
        f"{min_score:g}_{max_score:g}" for min_score, max_score in score_ranges
    ]
    tag_prefix = "score_range" if len(score_ranges) == 1 else "score_ranges"
    logger.info("  Score ranges: %s", " or ".join(formatted_ranges))
    return [
        _DiversityThreshold(
            tag=f"{tag_prefix}_{'__'.join(tag_ranges)}",
            label=f"score {'range' if len(score_ranges) == 1 else 'ranges'} "
            f"{' or '.join(formatted_ranges)}",
            score_ranges=score_ranges,
        )
    ]


def _number_to_filename_tag(value: float) -> str:
    """Convert a numeric setting to a compact file-name tag."""
    return f"{value:g}".replace(".", "d").replace("-", "m")


def _unlinkability_metric_filename_tag(
    score_ranges_requested: bool,
    d_local_threshold: float,
) -> str:
    """Build the file-name suffix for D(s)-derived diversity ranges."""
    if not score_ranges_requested:
        return ""
    return f"-threshold{_number_to_filename_tag(d_local_threshold)}"


def _unlinkability_score_range_thresholds(
    mated_score_file: Path | None,
    non_mated_score_file: Path | None,
    metric_bins: int,
    omega: float,
    d_local_threshold: float,
    x_min: float | None,
    x_max: float | None,
) -> list[_DiversityThreshold]:
    """Build score-range criteria from unlinkability D(s) values."""
    if mated_score_file is None and non_mated_score_file is None:
        return []
    if mated_score_file is None or non_mated_score_file is None:
        raise click.ClickException(
            "--unlinkability-mated-score-file and "
            "--unlinkability-non-mated-score-file must be passed together."
        )

    logger.info("Loading mated unlinkability scores from %s", mated_score_file)
    mated_scores = load_scores(mated_score_file)
    logger.info("Loading non-mated unlinkability scores from %s", non_mated_score_file)
    non_mated_scores = load_scores(non_mated_score_file)
    metric = compute_unlinkability_metric(
        mated_scores,
        non_mated_scores,
        metric_bins=metric_bins,
        omega=omega,
        x_min=x_min,
        x_max=x_max,
    )
    logger.info("Dsys: %.6f", metric.d_system)

    score_ranges = tuple(
        d_local_at_or_below_threshold_ranges(
            metric.edges,
            metric.d_local,
            d_local_threshold,
        )
    )
    if not score_ranges:
        raise click.ClickException(
            f"No D(s) <= {d_local_threshold:g} score range was found."
        )

    for index, (min_score, max_score) in enumerate(score_ranges, start=1):
        logger.info(
            "D(s) <= C range %d (C=%.6f): [%.6f, %.6f]",
            index,
            d_local_threshold,
            min_score,
            max_score,
        )
    return _score_range_thresholds(score_ranges)


def _json_safe_value(value: Any) -> Any:
    """Convert numpy values to JSON-serializable Python values."""
    if isinstance(value, numpy.generic):
        return _json_safe_value(value.item())
    if isinstance(value, numpy.ndarray):
        return _json_safe_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


def _json_cell(value: Any) -> str:
    """Serialize a value as one compact CSV-safe JSON cell."""
    return json.dumps(
        _json_safe_value(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _maximum_clique(
    qualifying_pairs: numpy.ndarray,
    n_nodes: int,
) -> tuple[int, ...]:
    """Return the node indices of an exact maximum clique.

    ``qualifying_pairs`` follows upper-triangle order. A qualifying comparison
    creates an edge between the corresponding protected-template nodes.
    """
    graph = networkx.Graph()
    graph.add_nodes_from(range(n_nodes))

    pair_idx = 0
    for left in range(n_nodes):
        for right in range(left + 1, n_nodes):
            if qualifying_pairs[pair_idx]:
                graph.add_edge(left, right)
            pair_idx += 1

    if not graph:
        return ()

    clique, _ = networkx.algorithms.clique.max_weight_clique(
        graph,
        weight=None,
    )
    return tuple(sorted(clique))


def _clique_pair_scores(
    pair_scores: numpy.ndarray,
    n_nodes: int,
    clique: tuple[int, ...],
) -> list[Any]:
    """Return existing upper-triangle scores whose endpoints are in a clique."""
    clique_nodes = set(clique)
    selected_scores = []
    pair_idx = 0
    for left in range(n_nodes):
        for right in range(left + 1, n_nodes):
            if left in clique_nodes and right in clique_nodes:
                selected_scores.append(pair_scores[pair_idx])
            pair_idx += 1
    return selected_scores


def _subject_clique_result(
    subject_id: str,
    protected_templates: list[Any],
    thresholds: list[_DiversityThreshold],
    btp_alg: Any,
) -> dict[str, Any]:
    """Compute every requested maximum clique for one subject."""
    pair_scores = numpy.asarray(
        [
            btp_alg.compare(protected_templates[left], protected_templates[right])
            for left in range(len(protected_templates))
            for right in range(left + 1, len(protected_templates))
        ]
    )
    protected_keys = [template.get_keys() for template in protected_templates]
    row: dict[str, Any] = {"subject_id": subject_id}

    for threshold in thresholds:
        clique = _maximum_clique(
            threshold.qualifies(pair_scores),
            len(protected_templates),
        )
        row[f"{threshold.tag}_clique_size"] = len(clique)
        row[f"{threshold.tag}_clique_keys"] = _json_cell(
            [protected_keys[node] for node in clique]
        )
        row[f"{threshold.tag}_clique_scores"] = _json_cell(
            _clique_pair_scores(
                pair_scores,
                len(protected_templates),
                clique,
            )
        )

    return row


def _init_diversity_worker(btp_alg_cls: type, btp_alg_args: tuple[Any, ...]) -> None:
    """Create one BTP algorithm instance in each subject worker process."""
    global _DIVERSITY_BTP_ALG
    _DIVERSITY_BTP_ALG = btp_alg_cls(*btp_alg_args)


def _subject_clique_worker(
    task: tuple[str, list[Any], list[_DiversityThreshold]],
) -> dict[str, Any]:
    """Process-pool entry point for one subject graph."""
    if _DIVERSITY_BTP_ALG is None:
        raise RuntimeError("Diversity worker was not initialized.")
    subject_id, protected_templates, thresholds = task
    return _subject_clique_result(
        subject_id,
        protected_templates,
        thresholds,
        _DIVERSITY_BTP_ALG,
    )


def _subject_clique_results(
    subject_protected: dict[str, list[Any]],
    thresholds: list[_DiversityThreshold],
    btp_alg: Any,
    btp_alg_cls: type,
    btp_alg_args: tuple[Any, ...],
    n_workers: int,
) -> list[dict[str, Any]]:
    """Compute subject graphs in parallel while preserving subject order."""
    tasks = [
        (subject_id, protected_templates, thresholds)
        for subject_id, protected_templates in subject_protected.items()
    ]
    if not tasks:
        return []
    if n_workers < 1:
        raise ValueError("Diversity analysis needs at least one worker.")
    if n_workers == 1:
        return [
            _subject_clique_result(subject_id, templates, criteria, btp_alg)
            for subject_id, templates, criteria in tasks
        ]

    worker_count = min(n_workers, len(tasks))
    chunksize = pipeline_utils.get_optimal_chunksize(len(tasks), worker_count)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_diversity_worker,
        initargs=(btp_alg_cls, btp_alg_args),
    ) as executor:
        return list(
            executor.map(
                _subject_clique_worker,
                tasks,
                chunksize=chunksize,
            )
        )


def _append_average_row(
    subject_results: pandas.DataFrame,
    thresholds: list[_DiversityThreshold],
) -> pandas.DataFrame:
    """Append mean clique sizes with blank clique detail cells."""
    output = subject_results.copy()
    average_row: dict[str, Any] = {"subject_id": "AVERAGE"}
    for threshold in thresholds:
        size_column = f"{threshold.tag}_clique_size"
        keys_column = f"{threshold.tag}_clique_keys"
        scores_column = f"{threshold.tag}_clique_scores"
        # Object dtype preserves integer formatting for per-subject clique sizes
        # while allowing a fractional mean in the same CSV column.
        output[size_column] = output[size_column].astype(object)
        average_row[size_column] = subject_results[size_column].mean()
        average_row[keys_column] = ""
        average_row[scores_column] = ""

    return pandas.concat(
        [output, pandas.DataFrame([average_row])],
        ignore_index=True,
    )


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
    help="Specify the JSON file containing the bucket of keys (system-specific key selection format).",
)
@click.option(
    "-b",
    "--keys-bucket",
    "keys_bucket",
    type=str,
    required=True,
    help="Specify the bucket to use from the keys file (e.g. '-0.9').",
)
@click.option(
    "-p",
    "--protected-score-file",
    "protected_score_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    default=None,
    help="Protected score file used to compute FMR thresholds.",
)
@click.option(
    "-f",
    "--fmr",
    "fmrs",
    type=float,
    multiple=True,
    default=(),
    help="FMR value(s) for matching thresholds. Defaults to 0.01%%, 0.1%%, "
    "and 1%% when no FMR, score range, or unlinkability score pair is passed.",
)
@click.option(
    "--score-range",
    "score_ranges",
    type=(float, float),
    multiple=True,
    help="Inclusive diversity score range as MIN MAX. Qualifies scores with "
    "MIN <= score <= MAX. Repeat for alternative accepted ranges.",
)
@click.option(
    "--unlinkability-mated-score-file",
    "unlinkability_mated_score_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    default=None,
    help="Mated unlinkability score CSV used to derive D(s) score ranges.",
)
@click.option(
    "--unlinkability-non-mated-score-file",
    "unlinkability_non_mated_score_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    default=None,
    help="Non-mated unlinkability score CSV used to derive D(s) score ranges.",
)
@click.option(
    "--metric-bins",
    "metric_bins",
    type=click.IntRange(min=2),
    default=10,
    show_default=True,
    help="Number of bins used to compute unlinkability D(s) score ranges.",
)
@click.option(
    "--omega",
    "omega",
    type=click.FloatRange(min=0.0, min_open=True),
    default=1.0,
    show_default=True,
    help="Prior ratio omega used in the unlinkability metric.",
)
@click.option(
    "--d-local-threshold",
    "--d-local-match-value",
    "d_local_threshold",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.0,
    show_default=True,
    help="Local D(s) threshold C; D(s) <= C regions become diversity ranges.",
)
@click.option(
    "--x-min",
    "x_min",
    type=float,
    default=None,
    help="Minimum score used to compute unlinkability metric bins.",
)
@click.option(
    "--x-max",
    "x_max",
    type=float,
    default=None,
    help="Maximum score used to compute unlinkability metric bins.",
)
@click.option(
    "--seed",
    "seed",
    type=int,
    default=None,
    help="Optional random seed for reproducible shuffles.",
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    help="Specify the location of the output directory.",
)
@click.option(
    "--random-vectors",
    "random_vectors",
    type=int,
    default=0,
    show_default=True,
    help="If > 0, skip dataset/feature extraction and use this many random "
    "input vectors (one per synthetic subject) instead of real features.",
)
@click.option(
    "--vector-dim",
    "vector_dim",
    type=int,
    default=512,
    show_default=True,
    help="Dimensionality of random input vectors (only used with --random-vectors).",
)
@click.option(
    "--real-sample-set",
    "real_sample_set",
    type=click.Choice(["verification", "unlinkability"]),
    default="verification",
    show_default=True,
    help="Dataset sample set to use when extracting real templates.",
)
@click.option(
    "--n-subjects",
    "n_subjects",
    type=click.IntRange(min=-1),
    default=-1,
    show_default=True,
    help="Process only the first N subjects in dataset order. Use -1 for all subjects.",
)
@click.option("--override", is_flag=True, help="Override existing output.")
def pipeline(
    system_config_file: Path,
    exp_config_file: Path,
    keys_file: Path,
    keys_bucket: str,
    protected_score_file: Path | None,
    fmrs: tuple[float, ...],
    score_ranges: tuple[tuple[float, float], ...],
    unlinkability_mated_score_file: Path | None,
    unlinkability_non_mated_score_file: Path | None,
    metric_bins: int,
    omega: float,
    d_local_threshold: float,
    x_min: float | None,
    x_max: float | None,
    seed: int | None,
    output_dir: Path | None,
    random_vectors: int,
    vector_dim: int,
    real_sample_set: str,
    n_subjects: int,
    override: bool,
):
    """Find the largest mutually non-matching protected-template set."""

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

    pipeline_utils.validate_detector(detector)

    if n_subjects == 0:
        raise click.ClickException("--n-subjects must be -1 or greater than zero.")

    use_random = random_vectors > 0
    if use_random and real_sample_set != "verification":
        raise click.ClickException(
            "--real-sample-set only applies when real templates are used."
        )

    if use_random:
        database_name = f"random{random_vectors}d{vector_dim}"
        dataset = None
        logger.info(
            "Random-vector mode: %d synthetic subjects, dim=%d",
            random_vectors,
            vector_dim,
        )
    else:
        database_name, dataset = pipeline_utils.create_dataset(
            system_config, exp_config
        )

    # --------------------
    # Load keys from bucket
    # --------------------

    logger.info("Loading keys from bucket '%s' in %s", keys_bucket, keys_file)
    keys_dict = json.loads(keys_file.read_text())
    bucket = keys_dict["keys"][keys_bucket]
    bucket_keys = list(bucket.values()) if isinstance(bucket, dict) else list(bucket)
    logger.info("Loaded %d keys from bucket '%s'", len(bucket_keys), keys_bucket)

    # Keep the existing default FMR run unless explicit score ranges replace it.
    unlinkability_score_ranges_requested = (
        unlinkability_mated_score_file is not None
        or unlinkability_non_mated_score_file is not None
    )
    if not fmrs and not score_ranges and not unlinkability_score_ranges_requested:
        fmrs = _DEFAULT_FMRS
    thresholds = _fmr_thresholds(protected_score_file, fmrs)
    thresholds.extend(_score_range_thresholds(score_ranges))
    thresholds.extend(
        _unlinkability_score_range_thresholds(
            unlinkability_mated_score_file,
            unlinkability_non_mated_score_file,
            metric_bins,
            omega,
            d_local_threshold,
            x_min,
            x_max,
        )
    )

    logger.info("---Parameters---")
    logger.info("Output dir: %s", output_dir)
    logger.info("Verification baseline: %s", baseline)
    logger.info("Database: %s", database_name)
    logger.info("Keys in bucket: %d", len(bucket_keys))
    logger.info("Real sample set: %s", real_sample_set)
    logger.info("FMRs: %s", fmrs)
    logger.info("Score ranges: %s", score_ranges)
    logger.info("Unlinkability metric bins: %d", metric_bins)
    logger.info("Unlinkability D(s) threshold: %s", d_local_threshold)
    logger.info("Unlinkability x-axis range: [%s, %s]", x_min, x_max)
    logger.info("Subject limit: %s", "all" if n_subjects == -1 else n_subjects)
    logger.info("Seed: %s", seed)
    logger.info("----------------")

    rng = numpy.random.default_rng(seed)

    baseline_dict = get_baseline_dict()
    protected_baseline_dict = get_protected_baseline_dict()

    pipeline_utils.check_dir(output_dir, exp_config)

    # --------------------
    # Build per-subject templates (real features or random vectors)
    # --------------------

    if use_random:
        if baseline not in baseline_dict:
            logger.warning(
                "Baseline `%s` not registered; using its name as label only.",
                baseline,
            )
        baseline_label = baseline

        n_random_subjects = (
            random_vectors if n_subjects == -1 else min(random_vectors, n_subjects)
        )
        logger.info(
            "Generating %d random templates of dim %d",
            n_random_subjects,
            vector_dim,
        )
        unique_templates = [
            Template(str(i), "0", rng.uniform(-1, 1, size=vector_dim))
            for i in range(n_random_subjects)
        ]
        base_templates_by_subject = {
            template.subject_id: [template] for template in unique_templates
        }
        extracted_templates = unique_templates
    else:
        samples = dataset.samples(verification=real_sample_set == "verification")
        try:
            samples = _select_first_subjects(samples, n_subjects)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        # --------------------
        # Unprotected Pipeline (feature extraction)
        # --------------------

        if baseline not in baseline_dict:
            logger.error("No baseline: `%s`", baseline)
            return

        baseline_label = baseline
        bio_alg_cls = baseline_dict[baseline]
        bio_alg_args = (output_dir / baseline_label, save, detector)

        logger.info("FE on %s samples", real_sample_set)
        samples_indexes = list(range(len(samples)))
        ref_chunksize = pipeline_utils.get_optimal_chunksize(
            len(samples_indexes), n_worker, min_chunks_per_worker=3
        )
        logger.info(
            "Processing %d reference samples with %d workers, chunksize=%d",
            len(samples_indexes),
            n_worker,
            ref_chunksize,
        )

        templates = pipeline_utils.threaded_feature_extraction(
            samples, bio_alg_cls, bio_alg_args, compliant, n_worker, ref_chunksize
        )

        extracted_templates = templates
        if real_sample_set == "verification":
            unique_templates, _ = templates_to_single_per_subject(templates)
            base_templates_by_subject = {
                template.subject_id: [template] for template in unique_templates
            }

    # ------------------
    # Protected Pipeline
    # ------------------

    if not has_protected_part:
        logger.info("No BTP configured. Exiting.")
        return

    logger.info("Starting diversity analysis")
    protected_baseline_configs = exp_config["btps"]["algs"]

    keys_tag = pipeline_utils.key_bucket_tag(keys_bucket)
    sample_set_tag = "-unlinkability" if real_sample_set == "unlinkability" else ""
    subject_limit_tag = "" if n_subjects == -1 else f"-subjects{n_subjects}"
    unlinkability_metric_tag = _unlinkability_metric_filename_tag(
        unlinkability_score_ranges_requested,
        d_local_threshold,
    )

    for protected_baseline_config in protected_baseline_configs:
        logger.info("Running for config %s", protected_baseline_config)

        btp_alg, btp_alg_cls, btp_alg_args, score_dir = pipeline_utils.build_btp_alg(
            protected_baseline_config,
            protected_baseline_dict,
            output_dir,
            baseline_label,
            save,
        )

        n_keys_per_system = pipeline_utils.get_n_keys_per_system(
            protected_baseline_config
        )
        try:
            keys = _build_key_systems(
                bucket_keys,
                len(bucket_keys),
                n_keys_per_system,
                rng,
            )
            if real_sample_set == "unlinkability":
                templates_by_subject, keys, protection_batches = (
                    _unlinkability_protection_batches(extracted_templates, keys)
                )
                n_unlinkability_templates = sum(
                    len(subject_templates)
                    for subject_templates in templates_by_subject.values()
                )
                logger.info(
                    "Selected %d unlinkability template(s) from %d subject(s) "
                    "with %d matching key system(s)",
                    n_unlinkability_templates,
                    len(templates_by_subject),
                    len(keys),
                )
            else:
                templates_by_subject = base_templates_by_subject
                protection_batches = [(key, unique_templates) for key in keys]
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        n_keys = len(keys)
        selected_subject_count = len(templates_by_subject)
        logger.info(
            "Running diversity for %d subject(s), %d key system(s), "
            "%d key value(s) per system",
            selected_subject_count,
            n_keys,
            n_keys_per_system,
        )

        output_file = (
            score_dir
            / f"diversity-{database_name}{sample_set_tag}{subject_limit_tag}-{btp_alg.get_alg_name()}-{baseline_label}-keys{keys_tag}-{n_keys}{unlinkability_metric_tag}.csv"
        )

        if output_file.exists() and not override:
            logger.warning("Output file exists: %s. Skipping...", output_file)
            continue

        # --------------------
        # Protect the selected template batch for each key.
        # --------------------

        logger.info("Protecting templates across %d selected key(s)", n_keys)
        max_batch_size = max(
            (len(batch_templates) for _, batch_templates in protection_batches),
            default=0,
        )
        protect_chunksize = pipeline_utils.get_optimal_chunksize(
            max_batch_size, n_worker, min_chunks_per_worker=4
        )

        # subject_id -> list of ProtectedTemplate (one per key)
        subject_protected: dict[str, list] = {}

        for key_idx, (key, batch_templates) in enumerate(protection_batches):
            logger.info("  Protecting with key %d/%d: %s", key_idx + 1, n_keys, key)

            prot_templates = pipeline_utils.processed_protection(
                batch_templates,
                btp_alg_cls,
                btp_alg_args,
                compliant,
                n_worker,
                protect_chunksize,
                extra_args={
                    "key_dictionary": {t.subject_id: key for t in batch_templates}
                },
            )

            for prot_tpl in prot_templates:
                if prot_tpl.get_template() is None:
                    continue
                subject_protected.setdefault(prot_tpl.subject_id, []).append(prot_tpl)

        # --------------------
        # Find a maximum non-matching clique per subject and criterion.
        # --------------------

        subject_worker_count = min(n_worker, len(subject_protected))
        logger.info(
            "Computing pairwise comparisons and maximum cliques for %d "
            "subject(s) with %d worker process(es)",
            len(subject_protected),
            subject_worker_count,
        )
        results = _subject_clique_results(
            subject_protected,
            thresholds,
            btp_alg,
            btp_alg_cls,
            btp_alg_args,
            n_worker,
        )

        for row in results:
            for threshold in thresholds:
                logger.info(
                    "Subject %s: maximum clique size=%d for %s",
                    row["subject_id"],
                    row[f"{threshold.tag}_clique_size"],
                    threshold.label,
                )

        if not results:
            logger.error(
                "No protected templates were available for diversity analysis."
            )
            continue

        # --------------------
        # Save per-subject maximum cliques.
        # --------------------

        subject_results_df = pandas.DataFrame(results)
        results_df = _append_average_row(subject_results_df, thresholds)

        results_df.to_csv(output_file, index=False)
        logger.info("Diversity results saved to %s", output_file)

        # Log summary
        logger.info("--- Diversity Summary ---")
        logger.info("Algorithm: %s", btp_alg.get_alg_name())
        logger.info("Database: %s", database_name)
        logger.info(
            "Subjects with results: %d / %d",
            len(results),
            selected_subject_count,
        )
        for threshold in thresholds:
            clique_sizes = subject_results_df[f"{threshold.tag}_clique_size"]
            logger.info(
                "  %s: maximum clique size min=%d, mean=%.2f, max=%d",
                threshold.label,
                clique_sizes.min(),
                clique_sizes.mean(),
                clique_sizes.max(),
            )
        logger.info("-------------------------")

    logger.info("Diversity experiment finished!")


if __name__ == "__main__":
    pipeline()
