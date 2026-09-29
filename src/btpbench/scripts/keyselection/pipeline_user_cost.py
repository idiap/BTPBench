# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

"""Measure the cost of threshold-based user key selection."""

import csv
import hashlib
import json
import logging
import math

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import bob.measure
import click
import numpy
import pandas

from btpbench.algorithms import get_baseline_dict, get_protected_baseline_dict
from btpbench.baselines import Template
from btpbench.metrics import neg_pos_scores
from btpbench.scripts import pipeline_utils
from btpbench.scripts.workers import BTPWorker
from btpbench.utils import templates_matrix_distribution, templates_to_matrix

pipeline_utils.setup_logger()
logger = logging.getLogger(__name__)

FIELDS = (
    "row_type",
    "subject_id",
    "template_id",
    "fmr",
    "threshold",
    "elapsed_seconds",
    "n_trials",
    "status",
    "n_subjects",
    "n_selected",
    "n_missing_templates",
    "num_processes",
    "key_sampling_seed",
)


def _validate_fmrs(
    ctx: click.Context, param: click.Parameter, values: tuple[float, ...]
) -> tuple[float, ...]:
    """Validate and de-duplicate requested false match rates."""
    del ctx, param
    if any(not math.isfinite(value) or not 0 < value < 1 for value in values):
        raise click.BadParameter(
            "FMRs must be finite fractions strictly between 0 and 1."
        )
    return tuple(dict.fromkeys(values))


def _thresholds_from_scores(
    verification_file: Path, fmrs: tuple[float, ...]
) -> dict[float, float]:
    """Load verification scores and calculate one threshold per FMR."""
    try:
        scores = pandas.read_csv(
            verification_file,
            dtype={
                "probe_subject_id": str,
                "bio_ref_subject_id": str,
            },
        )
    except (OSError, pandas.errors.EmptyDataError, pandas.errors.ParserError) as exc:
        raise click.ClickException(
            f"Could not read verification scores: {verification_file}"
        ) from exc

    required_columns = {"score", "probe_subject_id", "bio_ref_subject_id"}
    if scores.empty or not required_columns.issubset(scores.columns):
        raise click.ClickException(
            "Verification CSV must contain score, probe_subject_id, and "
            "bio_ref_subject_id columns."
        )

    for column in ("probe_subject_id", "bio_ref_subject_id"):
        if scores[column].isna().any():
            raise click.ClickException(
                f"Verification column {column} must not contain missing values."
            )
        identifiers = scores[column].astype(str).str.strip()
        if identifiers.eq("").any():
            raise click.ClickException(
                f"Verification column {column} must contain non-empty identifiers."
            )
        scores[column] = identifiers

    numeric_scores = pandas.to_numeric(scores["score"], errors="coerce")
    invalid_numeric = scores["score"].notna() & numeric_scores.isna()
    if invalid_numeric.any():
        raise click.ClickException("Verification scores must be numeric.")
    finite_scores = numeric_scores.dropna().to_numpy(dtype=float)
    if not numpy.isfinite(finite_scores).all():
        raise click.ClickException("Verification scores must be finite or NaN.")

    scores = scores.assign(score=numeric_scores).dropna(subset=["score"])
    negative_scores, positive_scores = neg_pos_scores(scores)
    if len(negative_scores) < 2:
        raise click.ClickException(
            "Verification scores must contain at least two non-mated scores after "
            "removing NaNs."
        )

    thresholds: dict[float, float] = {}
    for fmr in fmrs:
        threshold = float(
            bob.measure.far_threshold(negative_scores, positive_scores, fmr)
        )
        if not math.isfinite(threshold):
            raise click.ClickException(
                f"Could not calculate a finite threshold for FMR {fmr:g}."
            )
        thresholds[fmr] = threshold
    return thresholds


def _one_template_per_subject(templates: list[Template]) -> list[Template]:
    """Keep one row per subject, preferring the first usable template."""
    selected: dict[Any, Template] = {}
    for template in templates:
        current = selected.get(template.subject_id)
        if current is None or (
            current.get_template() is None and template.get_template() is not None
        ):
            selected[template.subject_id] = template
    return list(selected.values())


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_sha256(value: Any) -> str:
    """Return a stable SHA-256 digest for configuration data."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _resolved_config_path(config: dict[str, Any], key: str) -> str | None:
    """Resolve a configured path for provenance reporting."""
    value = config.get(key)
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


def _build_provenance(
    system_config: dict[str, Any],
    exp_config: dict[str, Any],
    btp_config: dict[str, Any],
    verification_file: Path,
    fmr: float,
    threshold: float,
    database_name: str,
    baseline: str,
    detector: str,
    n_worker: int,
    key_sampling_seed: int | None,
) -> dict[str, Any]:
    """Build the complete provenance record required to reuse a result."""
    databases = system_config.get("databases", {})
    database_config = (
        databases.get(database_name, {}) if isinstance(databases, dict) else {}
    )
    if not isinstance(database_config, dict):
        database_config = {}
    protocol_selection = {
        key: exp_config.get(key)
        for key in ("protocol", "protocols", "split", "splits")
        if key in exp_config
    }
    verification_samples_path = _resolved_config_path(
        database_config, "verification_samples"
    )
    verification_samples_sha256 = None
    if verification_samples_path is not None:
        samples_path = Path(verification_samples_path)
        if samples_path.is_file():
            verification_samples_sha256 = _file_sha256(samples_path)
    return {
        "schema_version": 1,
        "system_config_sha256": _value_sha256(system_config),
        "experiment_config_sha256": _value_sha256(exp_config),
        "btp_config_sha256": _value_sha256(btp_config),
        "verification_file": str(verification_file.expanduser().resolve()),
        "verification_file_sha256": _file_sha256(verification_file),
        "fmr": fmr,
        "threshold": threshold,
        "database": database_name,
        "database_config_sha256": _value_sha256(database_config),
        "dataset_path": _resolved_config_path(database_config, "dataset_dir"),
        "protocol_dir": _resolved_config_path(database_config, "proto_dir"),
        "verification_samples": verification_samples_path,
        "verification_samples_sha256": verification_samples_sha256,
        "protocol_selection": protocol_selection,
        "baseline": baseline,
        "detector": detector,
        "num_processes": n_worker,
        "key_sampling_seed": key_sampling_seed,
    }


def _provenance_path(result_path: Path) -> Path:
    """Return the provenance marker path for a result CSV."""
    return result_path.with_suffix(f"{result_path.suffix}.provenance.json")


def _result_is_current(result_path: Path, expected: dict[str, Any]) -> bool:
    """Return whether a result and its provenance match the current run."""
    marker_path = _provenance_path(result_path)
    if not result_path.is_file() or not marker_path.is_file():
        return False
    try:
        stored = json.loads(marker_path.read_text(encoding="utf-8"))
        if not isinstance(stored, dict):
            return False
        if set(stored) != {*expected, "output_sha256"}:
            return False
        if any(stored[key] != value for key, value in expected.items()):
            return False
        return stored["output_sha256"] == _file_sha256(result_path)
    except (OSError, json.JSONDecodeError):
        return False


def _invalidate_provenance(result_path: Path) -> None:
    """Remove completion markers before recomputing a result."""
    marker_path = _provenance_path(result_path)
    marker_path.unlink(missing_ok=True)
    marker_path.with_suffix(f"{marker_path.suffix}.partial").unlink(missing_ok=True)


def _publish_provenance(result_path: Path, provenance: dict[str, Any]) -> None:
    """Publish a provenance marker after its result CSV is complete."""
    marker_path = _provenance_path(result_path)
    partial_path = marker_path.with_suffix(f"{marker_path.suffix}.partial")
    completed = {**provenance, "output_sha256": _file_sha256(result_path)}
    with partial_path.open("w", encoding="utf-8") as stream:
        json.dump(completed, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
    partial_path.replace(marker_path)


def _validate_experiment(
    baseline: str,
    has_btp: bool,
    n_worker: int,
    configs: Any,
    baseline_dict: dict[str, type],
    protected_baseline_dict: dict[str, type],
) -> list[dict[str, Any]]:
    """Validate the configuration used by the cost experiment."""
    if not isinstance(n_worker, int) or isinstance(n_worker, bool) or n_worker < 1:
        raise click.ClickException("num_processes must be a positive integer.")
    if not isinstance(baseline, str) or baseline not in baseline_dict:
        raise click.ClickException(f"Unknown baseline: {baseline}")
    if not has_btp or not isinstance(configs, list) or not configs:
        raise click.ClickException(
            "Configure at least one user-specific BTP in btps.algs."
        )

    for config in configs:
        if not isinstance(config, dict) or not isinstance(config.get("type"), str):
            raise click.ClickException("Each BTP configuration must define a type.")
        if config["type"] not in protected_baseline_dict:
            raise click.ClickException(f"Unknown BTP algorithm: {config['type']}")
        if config.get("system_specific", False):
            raise click.ClickException(
                "Cost evaluation requires system_specific: false."
            )
        method = config.get("ks_method", "legacy")
        if method not in ("legacy", "multiple_guesses"):
            raise click.ClickException(f"Unknown key selection method: {method}")
        n_elements = config.get("ks_n_elements")
        if method == "multiple_guesses" and (
            not isinstance(n_elements, int)
            or isinstance(n_elements, bool)
            or n_elements < 1
        ):
            raise click.ClickException(
                "multiple_guesses requires a positive ks_n_elements."
            )
    return configs


def _init_cost_worker(*args: Any) -> None:
    """Initialize a worker and suppress per-candidate informational logs."""
    BTPWorker.init(*args)
    logging.getLogger("btpbench.btps").setLevel(logging.WARNING)


def _write_cost_file(
    path: Path,
    alg_cls: type,
    alg_args: tuple,
    compliant: bool,
    templates: list[Template],
    ref_distribution: Any,
    threshold: float,
    fmr: float,
    compare_f: Any,
    key_sampling_seed: int | None,
    n_worker: int,
) -> None:
    """Write subject results to a partial file and publish it atomically."""
    partial_path = path.with_suffix(f"{path.suffix}.partial")
    common = {
        "fmr": fmr,
        "threshold": threshold,
        "num_processes": n_worker,
        "key_sampling_seed": key_sampling_seed,
    }
    total_seconds = 0.0
    total_trials = 0
    n_selected = 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(
            max_workers=n_worker,
            initializer=_init_cost_worker,
            initargs=(
                alg_cls,
                alg_args,
                compliant,
                templates,
                None,
                None,
                {
                    "ref_distribution": ref_distribution,
                    "thresh": threshold,
                    "compare_f": compare_f,
                    "key_sampling_seed": key_sampling_seed,
                },
            ),
        ) as executor:
            for row in executor.map(
                BTPWorker.key_selection_usr_cost,
                range(len(templates)),
                chunksize=1,
            ):
                writer.writerow({**common, "row_type": "subject", **row})
                stream.flush()
                if row["status"] == "selected":
                    total_seconds += float(row["elapsed_seconds"])
                    total_trials += int(row["n_trials"])
                    n_selected += 1

        summary = {
            **common,
            "row_type": "mean",
            "status": "complete" if n_selected else "no_selection",
            "n_subjects": len(templates),
            "n_selected": n_selected,
            "n_missing_templates": len(templates) - n_selected,
        }
        if n_selected:
            summary["elapsed_seconds"] = total_seconds / n_selected
            summary["n_trials"] = total_trials / n_selected
        writer.writerow(summary)
    partial_path.replace(path)


@click.command()
@click.option(
    "-s",
    "--system-conf",
    "system_config_file",
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    required=True,
    help="System configuration file, as for pipeline_user.",
)
@click.option(
    "-e",
    "--exp-conf",
    "exp_config_file",
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    required=True,
    help="Experiment configuration with user-specific BTP algorithms.",
)
@click.option(
    "-v",
    "--verification-file",
    "-d",
    "--dedup-file",
    "verification_file",
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    required=True,
    help="Unprotected verification scores used to derive FMR thresholds.",
)
@click.option(
    "-f",
    "--fmr",
    "fmrs",
    type=float,
    multiple=True,
    default=(0.05, 0.10, 0.20),
    show_default=True,
    callback=_validate_fmrs,
    help="Target FMR fraction; repeat for multiple values (5% = 0.05).",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory; defaults to output_dir in the experiment config.",
)
@click.option("--override", is_flag=True, help="Overwrite completed cost CSVs.")
def pipeline(
    system_config_file: Path,
    exp_config_file: Path,
    verification_file: Path,
    fmrs: tuple[float, ...],
    output_dir: Path | None,
    override: bool,
) -> None:
    """Measure key-search time and trial counts for each subject and FMR."""
    system_config, exp_config = pipeline_utils.load_config(
        system_config_file, exp_config_file
    )
    baseline, has_btp, _, compliant, detector, n_worker = (
        pipeline_utils.load_common_parameters(system_config, exp_config)
    )
    baseline_dict = get_baseline_dict()
    protected_baseline_dict = get_protected_baseline_dict()
    btps_config = exp_config.get("btps")
    raw_configs = btps_config.get("algs") if isinstance(btps_config, dict) else None
    configs = _validate_experiment(
        baseline,
        has_btp,
        n_worker,
        raw_configs,
        baseline_dict,
        protected_baseline_dict,
    )
    try:
        pipeline_utils.validate_detector(detector)
        key_sampling_seed = pipeline_utils.key_sampling_seed(exp_config)
    except (TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    thresholds = _thresholds_from_scores(verification_file, fmrs)

    output_dir = pipeline_utils.resolve_output_dir(exp_config, output_dir)
    database_name, dataset = pipeline_utils.create_dataset(system_config, exp_config)
    pending: list[tuple[Any, type, tuple, float, float, Path, dict[str, Any]]] = []
    for config in configs:
        alg, alg_cls, alg_args, score_dir = pipeline_utils.build_btp_alg(
            config, protected_baseline_dict, output_dir, baseline, False
        )
        if alg is None:
            raise click.ClickException(
                f"Could not build BTP algorithm: {config['type']}"
            )
        if alg.is_system_specific():
            raise click.ClickException(
                "Cost evaluation requires user-specific BTP algorithms."
            )
        for fmr, threshold in thresholds.items():
            path = score_dir / (
                f"key_selection_cost_fmr{fmr}-{alg.get_key_selection_tag()}"
                f"-{alg.get_inversion_config_tag()}-{database_name}"
                f"-{alg.get_alg_name()}-{baseline}.csv"
            )
            provenance = _build_provenance(
                system_config,
                exp_config,
                config,
                verification_file,
                fmr,
                threshold,
                database_name,
                baseline,
                detector,
                n_worker,
                key_sampling_seed,
            )
            if not override and _result_is_current(path, provenance):
                logger.info("Skipping existing cost CSV: %s", path)
                continue
            if path.exists() and not override:
                logger.info("Recomputing stale cost CSV: %s", path)
            _invalidate_provenance(path)
            pending.append((alg, alg_cls, alg_args, fmr, threshold, path, provenance))
    if not pending:
        return

    samples = dataset.samples()
    bio_alg_cls = baseline_dict[baseline]
    bio_alg_args = (output_dir / baseline, False, detector)
    bio_alg = bio_alg_cls(*bio_alg_args)
    templates = pipeline_utils.threaded_feature_extraction(
        samples,
        bio_alg_cls,
        bio_alg_args,
        compliant,
        n_worker,
        pipeline_utils.get_optimal_chunksize(
            len(samples), n_worker, min_chunks_per_worker=3
        ),
    )
    selection_templates = _one_template_per_subject(templates)
    usable_templates = [
        template for template in templates if template.get_template() is not None
    ]
    if not selection_templates or not usable_templates:
        raise click.ClickException("No usable templates available for key selection.")

    distributions: dict[Any, Any] = {}
    for alg, alg_cls, alg_args, fmr, threshold, path, provenance in pending:
        precision = alg.get_inversion_config()["precision"]
        if precision not in distributions:
            matrix, _ = templates_to_matrix(usable_templates, precision)
            distributions[precision] = templates_matrix_distribution(matrix)
        logger.info(
            "Measuring %s at FMR %.1f%% (threshold %g)",
            alg.get_alg_name(),
            fmr * 100,
            threshold,
        )
        _write_cost_file(
            path,
            alg_cls,
            alg_args,
            compliant,
            selection_templates,
            distributions[precision],
            threshold,
            fmr,
            bio_alg.compare,
            key_sampling_seed,
            n_worker,
        )
        _publish_provenance(path, provenance)
        logger.info("Wrote cost CSV: %s", path)


if __name__ == "__main__":
    pipeline()
