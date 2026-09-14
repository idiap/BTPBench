# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import json
import logging
import time

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import bob.measure
import click
import pandas

import btpbench
import btpbench.metrics

from btpbench.algorithms import get_baseline_dict, get_protected_baseline_dict
from btpbench.scorewriter import CSVScoreWriter
from btpbench.scripts import pipeline_utils
from btpbench.scripts.workers import BTPWorker
from btpbench.utils import (
    templates_matrix_distribution,
    templates_to_matrix,
    templates_to_single_per_subject,
)

pipeline_utils.setup_logger()
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-s",
    "--system-conf",
    "system_config_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    required=True,
    help=("Specify the location of the system configuration file."),
)
@click.option(
    "-e",
    "--exp-conf",
    "exp_config_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    required=True,
    help=("Specify the location of the experiment configuration file."),
)
@click.option(
    "-v",
    "--verification-file",
    "verification_file",
    type=click.Path(dir_okay=False, exists=True, file_okay=True, path_type=Path),
    help="Verification score file used to calculate the decision threshold.",
)
@click.option(
    "-f",
    "--fmr",
    "fmr",
    type=float,
    default=0.01,
    help="False match rate used to calculate the decision threshold.",
)
@click.option(
    "-t",
    "--thresh",
    "thresh",
    type=float,
    default=None,
    help="Use this decision threshold instead of calculating one from scores.",
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    help=("Specify the location of the output directory."),
)
@click.option(
    "--random", is_flag=True, help="Use random key selection (for baseline comparison)."
)
@click.option("--override", is_flag=True, help="Override existing output.")
def pipeline(
    system_config_file: Path,
    exp_config_file: Path,
    verification_file: Path | None,
    fmr: float,
    thresh: float | None,
    output_dir: Path | None,
    random: bool,
    override: bool,
):
    """Select one inversion-resistant key per subject."""

    # --------------------
    # Config parsing
    # --------------------

    system_config, exp_config = pipeline_utils.load_config(
        system_config_file, exp_config_file
    )
    output_dir = pipeline_utils.resolve_output_dir(exp_config, output_dir)

    key_sampling_seed = pipeline_utils.key_sampling_seed(exp_config)

    baseline, has_protected_part, save, compliant, detector, n_worker = (
        pipeline_utils.load_common_parameters(system_config, exp_config)
    )

    pipeline_utils.validate_detector(detector)
    database_name, dataset = pipeline_utils.create_dataset(system_config, exp_config)

    logger.info("---Parameters---")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Verification baseline: {baseline}")
    logger.info(f"Has protection baseline: {has_protected_part}")
    logger.info(f"Save: {save}")
    logger.info(f"Compliant: {compliant}")
    logger.info(f"Database: {database_name}")
    logger.info(f"Key sampling seed: {key_sampling_seed}")
    logger.info("----------------")

    baseline_dict = get_baseline_dict()
    protected_baseline_dict = get_protected_baseline_dict()

    pipeline_utils.check_dir(output_dir, exp_config)

    samples = dataset.samples()
    metadata_names = dataset.metadata_names()

    logger.info(f"Running key selection evaluation on {database_name}")

    # --------------------
    # Unprotected Pipeline
    # --------------------

    if baseline not in baseline_dict:
        logger.error(f"No baseline: `{baseline}`")
        return

    baseline_label = baseline

    bio_alg_cls = baseline_dict[baseline]
    bio_alg_args = (output_dir / baseline_label, save, detector)
    bio_alg = bio_alg_cls(*bio_alg_args)

    # Feature extraction on reference samples
    logger.info("FE on reference samples")
    samples_indexes = list(range(len(samples)))

    # Calculate optimal chunksize for feature extraction (can handle larger chunks)
    ref_chunksize = pipeline_utils.get_optimal_chunksize(
        len(samples_indexes), n_worker, min_chunks_per_worker=3
    )
    logger.info(
        f"Processing {len(samples_indexes)} reference samples with {n_worker} workers, chunksize={ref_chunksize}"
    )

    templates = pipeline_utils.threaded_feature_extraction(
        samples, bio_alg_cls, bio_alg_args, compliant, n_worker, ref_chunksize
    )

    # ------------------
    # Protected Pipeline
    # ------------------

    # Retrieve the decision threshold from verification scores for protection.
    threshold: float = 0.0
    thresh_based = False

    if not random and thresh is not None:
        threshold = thresh
        thresh_based = True
    if not random and thresh is None:
        if verification_file is None:
            raise click.ClickException(
                "--verification-file is required unless --thresh or --random is used."
            )
        _, verification_df = btpbench.metrics.remove_nans(
            pandas.read_csv(verification_file)
        )
        negative_scores, positive_scores = btpbench.metrics.neg_pos_scores(
            verification_df
        )
        threshold = bob.measure.far_threshold(negative_scores, positive_scores, fmr)

    selection_templates, _ = templates_to_single_per_subject(templates)
    if not selection_templates:
        raise click.ClickException("No usable feature templates for key selection.")

    # Skip protected pipeline if not specified
    if not has_protected_part:
        logger.info("Experiment finished!")
        return

    logger.info("Starting BTP analysis")
    protected_baseline_configs = exp_config["btps"]["algs"]

    # Go through all the desired configurations
    for protected_baseline_config in protected_baseline_configs:
        logger.info(f"->Running for config {protected_baseline_config}")

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
        if btp_alg.is_system_specific():
            logger.warning(
                "BTP algorithm %s is system-specific. Skipping.",
                btp_alg.get_alg_name(),
            )
            continue

        # Compute Reference's distribution
        ref_templates_matrix, _ = templates_to_matrix(
            templates, btp_alg.get_inversion_config()["precision"]
        )
        ref_distribution = templates_matrix_distribution(ref_templates_matrix)

        fmr_tag = "random"
        if thresh_based:
            fmr_tag = f"thresh{int(threshold * 10000)}"
        elif not random:
            fmr_tag = f"fmr{int(fmr * 10000)}"

        selection_scores_file_path = (
            score_dir
            / f"ref_keys_select_{fmr_tag}-{btp_alg.get_key_selection_tag()}-{btp_alg.get_inversion_config_tag()}-{database_name}-{btp_alg.get_alg_name()}-{baseline_label}.csv"
        )

        keys_file_path = (
            score_dir
            / f"keys_select_{fmr_tag}-{btp_alg.get_key_selection_tag()}-{btp_alg.get_inversion_config_tag()}-{database_name}-{btp_alg.get_alg_name()}-{baseline_label}.json"
        )

        existing_outputs = [keys_file_path, selection_scores_file_path]
        if not override and any(path.exists() for path in existing_outputs):
            logger.warning(
                "Key-selection output already exists for %s. Skipping. Use "
                "--override to replace it.",
                btp_alg.get_alg_name(),
            )
            continue

        selection_csv = CSVScoreWriter(selection_scores_file_path, metadata_names)

        # Select a key using one template per subject.
        logger.info("Computing keys on selected templates")

        selection_template_indexes = list(range(len(selection_templates)))

        # Calculate optimal chunksize for protection (moderate chunks for memory balance)
        selection_chunksize = pipeline_utils.get_optimal_chunksize(
            len(selection_template_indexes), n_worker, min_chunks_per_worker=4
        )

        logger.info(
            f"Computing key on {len(selection_template_indexes)} templates with {n_worker} workers, chunksize={selection_chunksize}"
        )

        starting_time = time.time()

        with ProcessPoolExecutor(
            max_workers=n_worker,
            initializer=BTPWorker.init,
            initargs=(
                btp_alg_cls,
                btp_alg_args,
                compliant,
                selection_templates,
                None,
                None,
                {
                    "ref_distribution": ref_distribution,
                    "thresh": threshold,
                    "compare_f": bio_alg.compare,
                    "key_sampling_seed": key_sampling_seed,
                },
            ),
        ) as exe:
            selected_results = list(
                exe.map(
                    BTPWorker.key_selection_usr,
                    selection_template_indexes,
                    chunksize=selection_chunksize,
                )
            )

        logger.info(
            f"Elapsed time per subject: {(time.time() - starting_time) * n_worker / len(selection_template_indexes):.2f} seconds"
        )
        keys_dict = {
            t[0].subject_id: int(t[0].get_keys())
            for t in selected_results
            if t[0].get_template() is not None
        }

        keys_file_path.write_text(json.dumps(keys_dict, indent=4))

        for prot_template, inverted_template, score in selected_results:
            if prot_template.get_template() is None:
                continue

            selection_csv.write_score(score, prot_template, inverted_template)

        selection_csv.close()

    logger.info("Experiment finished!")


if __name__ == "__main__":
    pipeline()
