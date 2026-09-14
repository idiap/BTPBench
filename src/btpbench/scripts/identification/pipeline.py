# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import logging

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

    protocols_name = exp_config["protocols"]
    splits_name = exp_config["splits"]

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
    logger.info("----------------")

    baseline_dict = get_baseline_dict()
    protected_baseline_dict = get_protected_baseline_dict()

    pipeline_utils.check_dir(output_dir, exp_config)

    for protocol, split in pipeline_utils.iter_protocol_splits(
        dataset,
        protocols_name,
        splits_name,
        database_name,
    ):
        dataloader = dataset.load_protocol(protocol, split)
        metadata_names = dataloader.metadata_names()

        logger.info(f"Running evaluation on {database_name}-{protocol}-{split}")

        # --------------------
        # Unprotected Pipeline
        # --------------------

        if baseline not in baseline_dict:
            logger.error(f"No baseline: `{baseline}`")
            return

        baseline_label = baseline

        bio_alg_cls = baseline_dict[baseline]
        bio_alg_args = (output_dir / baseline_label, save, detector)

        # Feature extraction on reference samples
        logger.info("FE on reference samples")
        ref_templates = pipeline_utils.threaded_feature_extraction(
            list(dataloader.references()),
            bio_alg_cls,
            bio_alg_args,
            compliant,
            n_worker,
            batch_size,
        )

        # Feature extraction on probe samples
        logger.info("FE on probe samples")
        probes_templates = pipeline_utils.threaded_feature_extraction(
            list(dataloader.probes()),
            bio_alg_cls,
            bio_alg_args,
            compliant,
            n_worker,
            batch_size,
        )

        # Unprotected comparison
        logger.info("Matching")
        unprotected_scores_file_path = (
            output_dir / f"scores-{protocol}-{split}-{baseline_label}.csv"
        )
        pipeline_utils.threaded_unprotected_id_matching(
            ref_templates,
            probes_templates,
            metadata_names,
            bio_alg_cls,
            bio_alg_args,
            compliant,
            n_worker,
            batch_size,
            unprotected_scores_file_path,
            override,
        )

        # ------------------
        # Protected Pipeline
        # ------------------

        # Skip protected pipeline if not specified
        if not has_protected_part:
            logger.info("Experiment finished!")
            return

        logger.info("Starting BTP analysis")
        protected_baseline_configs = exp_config["btps"]["algs"]

        # Go through all the desired configurations
        for protected_baseline_config in protected_baseline_configs:
            logger.info(f"->Running for config {protected_baseline_config}")

            btp_alg, btp_alg_cls, btp_alg_args, score_dir = (
                pipeline_utils.build_btp_alg(
                    protected_baseline_config,
                    protected_baseline_dict,
                    output_dir,
                    baseline_label,
                    save,
                )
            )

            if btp_alg is None:
                logger.error(
                    f"Error building BTP algorithm for config {protected_baseline_config}... Skipping..."
                )
                continue

            protected_scores_file_path = (
                score_dir
                / f"scores-{protocol}-{split}-{btp_alg.get_alg_name()}-{baseline_label}.csv"
            )

            if protected_scores_file_path.exists() and not override:
                logger.warning(
                    f"Score file exists: {str(protected_scores_file_path)}... Exiting..."
                )
                continue

            # protected score csv file
            protected_csv = CSVScoreWriter(protected_scores_file_path, metadata_names)

            # Protect reference templates
            logger.info("Protecting reference templates")
            prot_ref_templates = pipeline_utils.processed_protection(
                ref_templates,
                btp_alg_cls,
                btp_alg_args,
                compliant,
                n_worker,
                batch_size,
            )

            # Protect probe templates
            logger.info("Protecting probe templates")
            prot_probe_templates = pipeline_utils.processed_protection(
                probes_templates,
                btp_alg_cls,
                btp_alg_args,
                compliant,
                n_worker,
                batch_size,
            )

            # Compare protected template
            logger.info("Performing protected comparison")
            pipeline_utils.processed_protected_id_matching(
                prot_ref_templates,
                prot_probe_templates,
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
