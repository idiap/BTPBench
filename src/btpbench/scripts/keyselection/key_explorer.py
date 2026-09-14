# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

"""Key explorer: find suitable keys for random vectors across multiple thresholds.

For each BTP configuration, generates random vectors and searches for keys k_i
such that Compare(BTP^-1(v_i, k_i), v_i) <= T for various thresholds T, with
all found keys being unique. Processing continues in batches until the time
limit is reached.
"""

import json
import logging
import time

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import numpy
import yaml

from btpbench.algorithms import get_protected_baseline_dict
from btpbench.baselines import Template, negative_cosine_distance
from btpbench.scripts import pipeline_utils
from btpbench.scripts.workers import BTPWorker
from btpbench.utils import templates_matrix_distribution, templates_to_matrix

pipeline_utils.setup_logger()
logger = logging.getLogger(__name__)


def generate_random_templates(
    rng: numpy.random.Generator,
    count: int,
    vector_dim: int,
    start_id: int = 0,
) -> list[Template]:
    """Generate a list of Template objects with uniform random vectors."""
    templates = []
    for i in range(count):
        vec = rng.uniform(-1, 1, size=vector_dim)
        templates.append(Template(str(start_id + i), "0", vec))
    return templates


@click.command()
@click.option(
    "-e",
    "--exp-conf",
    "exp_config_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    required=True,
    help="Experiment configuration file.",
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    help="Output directory for result JSON files. Defaults to exp config output_dir.",
)
@click.option(
    "-t",
    "--threshold",
    "thresholds",
    type=float,
    multiple=True,
    required=True,
    help="Threshold value(s). Can be specified multiple times (e.g. -t -0.75 -t -1.0).",
)
@click.option(
    "--time-limit",
    type=float,
    default=48 * 3600,
    help="Time limit in seconds (default: 48h = 172800s).",
)
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Number of random vectors per batch (default: 100).",
)
@click.option(
    "--vector-dim",
    type=int,
    default=512,
    help="Dimensionality of random vectors (default: 512).",
)
@click.option(
    "--n-dist-samples",
    type=int,
    default=1000,
    help="Number of random samples for computing the reference distribution (default: 1000).",
)
@click.option("--override", is_flag=True, help="Override existing output files.")
@click.option("--bad", is_flag=True, help="Bad keys only.")
def key_explorer(
    exp_config_file: Path,
    output_dir: Path | None,
    thresholds: tuple[float, ...],
    time_limit: float,
    batch_size: int,
    vector_dim: int,
    n_dist_samples: int,
    override: bool,
    bad: bool,
):
    """Find suitable keys for random vectors across multiple thresholds.

    For each BTP config, generates random vectors v_i and finds keys k_i such
    that Compare(BTP^-1(v_i, k_i), v_i) <= T.  Key selection uses the most
    lenient (largest) threshold; the returned score places each key into all
    qualifying threshold buckets.  All keys are unique.

    Processing runs in batches until the time limit is reached.  Results are
    saved periodically so that partial progress is preserved.
    """

    # ------------------------------------------------------------------
    # Config parsing
    # ------------------------------------------------------------------
    with exp_config_file.open() as f:
        exp_config = yaml.safe_load(f)
    output_dir = pipeline_utils.resolve_output_dir(exp_config, output_dir)

    has_protected_part = "btps" in exp_config
    compliant = exp_config["compliant"]
    n_worker = exp_config["num_processes"]

    if not has_protected_part:
        logger.error("No BTP configuration found in experiment config.")
        return

    protected_baseline_dict = get_protected_baseline_dict()
    protected_baseline_configs = exp_config["btps"]["algs"]

    # Sort thresholds: most lenient (largest) first
    sorted_thresholds = sorted(thresholds, reverse=True)
    max_threshold = sorted_thresholds[0]
    if bad:
        max_threshold = 0

    rng = numpy.random.default_rng()

    logger.info("---Key Explorer Parameters---")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Thresholds: {sorted_thresholds}")
    logger.info(f"Most lenient threshold (used for key selection): {max_threshold}")
    logger.info(f"Time limit: {time_limit}s")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Vector dim: {vector_dim}")
    logger.info(f"Distribution samples: {n_dist_samples}")
    logger.info(f"Workers: {n_worker}")
    logger.info("-----------------------------")

    # ------------------------------------------------------------------
    # Process each BTP configuration
    # ------------------------------------------------------------------
    for protected_baseline_config in protected_baseline_configs:
        logger.info(f"Processing BTP config: {protected_baseline_config}")

        btp_alg, btp_alg_cls, btp_alg_args, score_dir = pipeline_utils.build_btp_alg(
            protected_baseline_config,
            protected_baseline_dict,
            output_dir,
            "random",
            False,  # save=False: no intermediate template I/O
        )

        if btp_alg is None:
            continue

        # ------------------------------------------------------------------
        # Reference distribution (from random vectors)
        # ------------------------------------------------------------------
        logger.info(
            f"Generating {n_dist_samples} random vectors for reference distribution"
        )
        dist_templates = generate_random_templates(rng, n_dist_samples, vector_dim)
        precision = btp_alg.get_inversion_config()["precision"]
        ref_matrix, _ = templates_to_matrix(dist_templates, precision)
        ref_distribution = templates_matrix_distribution(ref_matrix)

        # ------------------------------------------------------------------
        # Output file path
        # ------------------------------------------------------------------
        thresholds_tag = "_".join(str(t) for t in sorted_thresholds)
        tag = "bad_" if bad else ""
        output_file = (
            score_dir / f"{tag}key_explorer_{thresholds_tag}"
            f"-{btp_alg.get_key_selection_tag()}"
            f"-{btp_alg.get_inversion_config_tag()}"
            f"-{btp_alg.get_alg_name()}.json"
        )

        if output_file.exists() and not override:
            logger.warning(
                f"Output file exists: {output_file}. Skipping. Use --override to overwrite."
            )
            continue

        # ------------------------------------------------------------------
        # Key exploration loop
        # ------------------------------------------------------------------
        buckets: dict[str, dict[str, int]] = {str(t): {} for t in sorted_thresholds}
        used_keys: set[int] = set()
        vector_counter = 0
        total_found = 0

        start_time = time.time()

        while time.time() - start_time < time_limit:
            # Generate a batch of random vectors
            batch_templates = generate_random_templates(
                rng, batch_size, vector_dim, start_id=vector_counter
            )

            batch_indexes = list(range(len(batch_templates)))
            chunksize = pipeline_utils.get_optimal_chunksize(
                len(batch_indexes), n_worker
            )

            # Run key selection in parallel with the most lenient threshold
            with ProcessPoolExecutor(
                max_workers=n_worker,
                initializer=BTPWorker.init,
                initargs=(
                    btp_alg_cls,
                    btp_alg_args,
                    compliant,
                    batch_templates,
                    None,
                    None,
                    {
                        "ref_distribution": ref_distribution,
                        "thresh": max_threshold,
                        "compare_f": negative_cosine_distance,
                    },
                ),
            ) as exe:
                results = list(
                    exe.map(
                        BTPWorker.key_selection_usr,
                        batch_indexes,
                        chunksize=chunksize,
                    )
                )

            # Place results into threshold buckets
            for prot_template, _inv_template, score in results:
                if prot_template.get_template() is None:
                    continue

                key = int(prot_template.get_keys())
                vector_id = prot_template.subject_id

                # Enforce global key uniqueness
                if key in used_keys:
                    logger.debug(
                        f"Duplicate key {key} for vector {vector_id}, skipping"
                    )
                    continue

                used_keys.add(key)
                total_found += 1

                # Assign to every threshold bucket the score qualifies for
                for t in sorted_thresholds:
                    if bad:
                        if score >= t:
                            buckets[str(t)][vector_id] = key
                    else:
                        if score <= t:
                            buckets[str(t)][vector_id] = key

            vector_counter += batch_size
            elapsed = time.time() - start_time

            logger.info(
                f"Batch done | Vectors processed: {vector_counter} | "
                f"Keys found: {total_found} | "
                f"Elapsed: {elapsed:.0f}s / {time_limit:.0f}s"
            )

            # Periodic save (crash resilience)
            _save_results(
                output_file, buckets, sorted_thresholds, vector_counter, elapsed
            )

        # ------------------------------------------------------------------
        # Final summary
        # ------------------------------------------------------------------
        elapsed = time.time() - start_time
        _save_results(output_file, buckets, sorted_thresholds, vector_counter, elapsed)

        for t in sorted_thresholds:
            logger.info(f"Threshold {t}: {len(buckets[str(t)])} keys found")

        logger.info(f"Results saved to {output_file}")

    logger.info("Key exploration complete!")


def _save_results(
    output_file: Path,
    buckets: dict[str, dict[str, int]],
    sorted_thresholds: list[float],
    vectors_processed: int,
    elapsed: float,
) -> None:
    """Write the current state to the output JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "thresholds": sorted_thresholds,
            "vectors_processed": vectors_processed,
            "elapsed_seconds": round(elapsed, 2),
            "keys_per_threshold": {
                str(t): len(buckets[str(t)]) for t in sorted_thresholds
            },
        },
        "keys": buckets,
    }
    output_file.write_text(json.dumps(payload, indent=4))


if __name__ == "__main__":
    key_explorer()
