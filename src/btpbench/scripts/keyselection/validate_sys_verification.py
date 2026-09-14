# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import logging
import random

from pathlib import Path

import click

from btpbench.algorithms import get_baseline_dict, get_protected_baseline_dict
from btpbench.scorewriter import CSVScoreWriter
from btpbench.scripts import pipeline_utils

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
    "-o",
    "--output-dir",
    "output_dir",
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    help=("Specify the location of the output directory."),
)
@click.option("--override", is_flag=True, help="Override existing output.")
def pipeline(
    system_config_file: Path,
    exp_config_file: Path,
    output_dir: Path | None,
    override: bool,
):
    """Entry point to run a pipeline."""

    # --------------------
    # Config parsing
    # --------------------

    system_config, exp_config = pipeline_utils.load_config(
        system_config_file, exp_config_file
    )
    output_dir = pipeline_utils.resolve_output_dir(exp_config, output_dir)

    sampling_mode = exp_config.get("sampling_mode", "keys")
    n_keys = exp_config["n_keys"]
    key_sampling_seed = pipeline_utils.key_sampling_seed(exp_config)
    key_pool, _, key_pool_tag = pipeline_utils.load_key_pool(exp_config)
    base_keys_tag = f"-{key_pool_tag}"

    baseline, has_protected_part, save, compliant, detector, n_worker = (
        pipeline_utils.load_common_parameters(system_config, exp_config)
    )
    batch_size = n_worker * 100

    pipeline_utils.validate_detector(detector)
    database_name, dataset = pipeline_utils.create_dataset(system_config, exp_config)

    logger.info("---Parameters---")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Verification baseline: {baseline}")
    logger.info(f"Has protection baseline: {has_protected_part}")
    logger.info(f"Save: {save}")
    logger.info(f"Compliant: {compliant}")
    logger.info(f"Database: {database_name}")
    logger.info(f"Nb Keys: {n_keys}")
    logger.info(f"Key sampling seed: {key_sampling_seed}")
    logger.info(f"Sampling mode: {sampling_mode}")
    logger.info("----------------")

    baseline_dict = get_baseline_dict()
    protected_baseline_dict = get_protected_baseline_dict()

    pipeline_utils.check_dir(output_dir, exp_config)

    metadata_names = dataset.metadata_names()
    metadata_names.append("key")

    if baseline not in baseline_dict:
        logger.error(f"No baseline: `{baseline}`")
        return

    # Skip protected pipeline if not specified
    if not has_protected_part:
        logger.info("Experiment finished!")
        return

    baseline_label = baseline

    bio_alg_cls = baseline_dict[baseline]
    bio_alg_args = (output_dir / baseline_label, save, detector)

    logger.info("FE on samples")
    samples = dataset.samples()
    unprotected_templates = pipeline_utils.threaded_feature_extraction(
        samples, bio_alg_cls, bio_alg_args, compliant, n_worker, batch_size
    )

    logger.info("Starting BTP verification analysis")
    protected_baseline_configs = exp_config["btps"]["algs"]

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
                f"Error building BTP algorithm for config {protected_baseline_config}... Skipping..."
            )
            continue

        keys_tag = base_keys_tag
        n_keys_per_system = pipeline_utils.get_n_keys_per_system(
            protected_baseline_config
        )
        logger.info(
            "Sampling %d key system(s), %d key value(s) per system",
            n_keys,
            n_keys_per_system,
        )
        keys = pipeline_utils.sample_key_systems(
            key_pool,
            n_keys,
            n_keys_per_system,
            random.Random(key_sampling_seed),
        )

        if btp_alg.has_key_distribution():
            keys_tag += btp_alg.get_key_distribution_name()

        if not btp_alg.is_system_specific():
            logger.warning(
                f"BTP algorithm {btp_alg.get_alg_name()} is not system-specific. Skipping..."
            )
            continue

        protected_scores_file_path = (
            score_dir
            / f"sys_keys_verification-{database_name}-{btp_alg.get_alg_name()}-{baseline_label}-{n_keys}{keys_tag}.csv"
        )

        logger.info(f"Protected scores will be saved to: {protected_scores_file_path}")

        if protected_scores_file_path.exists() and not override:
            logger.warning(
                f"Score file exists: {str(protected_scores_file_path)}... Exiting..."
            )
            continue

        protected_csv = CSVScoreWriter(protected_scores_file_path, metadata_names)

        for key in keys:
            logger.info(f"Running for key: {key}")

            btp_alg.set_key(key)
            btp_alg_args[2]["key"] = key

            # At this stage no matter what was set save is set to False.
            btp_alg_args = (
                btp_alg_args[0],
                False,
                *btp_alg_args[2:],
            )

            logger.info("Protecting templates")
            protected_templates = pipeline_utils.processed_protection(
                unprotected_templates,
                btp_alg_cls,
                btp_alg_args,
                compliant,
                n_worker,
                batch_size,
            )

            for template in protected_templates:
                template.metadata["key"] = key

            logger.info("Performing protected verification matching")
            pipeline_utils.processed_protected_verification_matching(
                protected_templates,
                btp_alg_cls,
                btp_alg_args,
                compliant,
                n_worker,
                batch_size,
                protected_csv,
            )

        protected_csv.close()

    logger.info("Experiment finished!")


if __name__ == "__main__":
    pipeline()
