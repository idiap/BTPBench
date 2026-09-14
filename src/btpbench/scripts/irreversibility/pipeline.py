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
from btpbench.utils import templates_matrix_distribution, templates_to_matrix

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
    """Run an irreversibility evaluation."""

    # --------------------
    # Config parsing
    # --------------------

    system_config, exp_config = pipeline_utils.load_config(
        system_config_file, exp_config_file
    )
    output_dir = pipeline_utils.resolve_output_dir(exp_config, output_dir)
    n_attack_trials, attack_seed = pipeline_utils.inversion_evaluation_parameters(
        exp_config
    )
    key_sampling_seed = pipeline_utils.key_sampling_seed(exp_config)
    sampling_mode = exp_config.get("sampling_mode", "configured")
    n_key_systems = int(exp_config.get("n_keys", 1))
    key_pool = None
    sampled_key_source = "configured"
    key_pool_tag = "configured"
    if sampling_mode == "configured":
        n_key_systems = 1
    else:
        key_pool, sampled_key_source, key_pool_tag = pipeline_utils.load_key_pool(
            exp_config
        )

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
    logger.info(f"Attack trials: {n_attack_trials}")
    logger.info(f"Attack seed: {attack_seed}")
    logger.info(f"Key source: {sampling_mode}")
    logger.info(f"Key systems: {n_key_systems}")
    logger.info(f"Key sampling seed: {key_sampling_seed}")
    logger.info("----------------")

    baseline_dict = get_baseline_dict()
    protected_baseline_dict = get_protected_baseline_dict()

    pipeline_utils.check_dir(output_dir, exp_config)

    dataloader = dataset.load_irreversibility()
    metadata_names = pipeline_utils.inversion_metadata_names(
        dataloader.metadata_names()
    )

    logger.info(f"Running irreversibility evaluation on {database_name}")

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
    ref_samples = list(dataloader.references())
    ref_indexes = list(range(len(ref_samples)))

    # Calculate optimal chunksize for feature extraction (can handle larger chunks)
    ref_chunksize = pipeline_utils.get_optimal_chunksize(
        len(ref_indexes), n_worker, min_chunks_per_worker=3
    )
    logger.info(
        f"Processing {len(ref_indexes)} reference samples with {n_worker} workers, chunksize={ref_chunksize}"
    )

    ref_templates = pipeline_utils.threaded_feature_extraction(
        ref_samples, bio_alg_cls, bio_alg_args, compliant, n_worker, ref_chunksize
    )

    # Feature extraction on probe samples
    logger.info("FE on probe samples")
    probe_samples = list(dataloader.probes())
    probe_indexes = list(range(len(probe_samples)))

    # Calculate optimal chunksize for probe feature extraction
    probe_chunksize = pipeline_utils.get_optimal_chunksize(
        len(probe_indexes), n_worker, min_chunks_per_worker=3
    )
    logger.info(
        f"Processing {len(probe_indexes)} probe samples with {n_worker} workers, chunksize={probe_chunksize}"
    )

    probes_templates = pipeline_utils.threaded_feature_extraction(
        probe_samples, bio_alg_cls, bio_alg_args, compliant, n_worker, probe_chunksize
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

        # Compute Reference's distribution
        ref_templates_matrix, _ = templates_to_matrix(
            ref_templates, btp_alg.get_inversion_config()["precision"]
        )
        ref_distribution = templates_matrix_distribution(ref_templates_matrix)

        key_scope = "system" if btp_alg.is_system_specific() else "user"
        key_scenarios: list[tuple[object | None, str]]
        if sampling_mode == "configured":
            if btp_alg.has_key_dictionary():
                configured_source = f"dictionary:{btp_alg.get_key_dictionary_name()}"
                key_pool_tag = "dictionary"
            elif btp_alg.is_system_specific():
                configured_source = "configured"
                key_pool_tag = "configured"
            else:
                configured_source = "subject-id"
                key_pool_tag = "subject-id"
            key_scenarios = [(None, configured_source)]
        else:
            assert key_pool is not None
            key_rng = random.Random(key_sampling_seed)
            n_keys_per_system = pipeline_utils.get_n_keys_per_system(
                protected_baseline_config
            )
            if btp_alg.is_system_specific():
                sampled_keys = pipeline_utils.sample_key_systems(
                    key_pool,
                    n_key_systems,
                    n_keys_per_system,
                    key_rng,
                )
                key_scenarios = [(key, sampled_key_source) for key in sampled_keys]
            else:
                subject_ids = list(
                    dict.fromkeys(template.subject_id for template in probes_templates)
                )
                user_assignments = pipeline_utils.sample_user_key_dictionaries(
                    subject_ids,
                    key_pool,
                    n_key_systems,
                    n_keys_per_system,
                    key_rng,
                )
                key_scenarios = [
                    (assignment, sampled_key_source) for assignment in user_assignments
                ]

        protected_scores_file_path = score_dir / (
            f"irreversibility-{key_pool_tag}-systems{len(key_scenarios)}-"
            f"trials{n_attack_trials}-"
            f"{btp_alg.get_inversion_config_tag()}-{database_name}-"
            f"{btp_alg.get_alg_name()}-{baseline_label}.csv"
        )

        if protected_scores_file_path.exists() and not override:
            logger.warning(
                f"Score file exists: {str(protected_scores_file_path)}... Exiting..."
            )
            continue

        # protected score csv file
        protected_csv = CSVScoreWriter(protected_scores_file_path, metadata_names)

        # Calculate optimal chunksize for protection and inversion.
        protect_chunksize = pipeline_utils.get_optimal_chunksize(
            len(probe_indexes), n_worker, min_chunks_per_worker=4
        )
        invert_chunksize = pipeline_utils.get_optimal_chunksize(
            len(probe_indexes), n_worker, min_chunks_per_worker=6
        )
        invert_chunksize = min(invert_chunksize, 10)

        for key_system, (key_assignment, key_source) in enumerate(key_scenarios):
            logger.info(
                "Evaluating key system %d/%d from %s",
                key_system + 1,
                len(key_scenarios),
                key_source,
            )
            protection_args = None
            if btp_alg.is_system_specific() and key_assignment is not None:
                btp_alg.set_key(key_assignment)
                btp_alg_args[2]["key"] = key_assignment
            elif key_assignment is not None:
                protection_args = {"key_dictionary": key_assignment}

            prot_probe_templates = pipeline_utils.processed_protection(
                probes_templates,
                btp_alg_cls,
                btp_alg_args,
                compliant,
                n_worker,
                protect_chunksize,
                protection_args,
            )
            pipeline_utils.evaluate_inversion_trials(
                probes_templates,
                prot_probe_templates,
                btp_alg_cls,
                btp_alg_args,
                compliant,
                n_worker,
                ref_distribution,
                invert_chunksize,
                n_attack_trials,
                attack_seed,
                bio_alg.compare,
                protected_csv,
                key_source,
                key_scope,
                key_system,
            )

        protected_csv.close()

    logger.info("Experiment finished!")


if __name__ == "__main__":
    pipeline()
