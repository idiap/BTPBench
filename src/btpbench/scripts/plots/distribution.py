# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import logging

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import click
import matplotlib
import numpy

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from btpbench.algorithms import get_baseline_dict, get_protected_baseline_dict
from btpbench.scripts import pipeline_utils
from btpbench.scripts.workers import BTPWorker, FRWorker
from btpbench.utils import templates_to_matrix

pipeline_utils.setup_logger()
logger = logging.getLogger(__name__)


def norm_plot(data: numpy.ndarray, title: str, output_file: Path):
    """Create a norm plot of the data."""
    plt.figure(figsize=(12, 6))
    norms = numpy.linalg.norm(data, axis=1)
    plt.plot(norms, color="red")

    plt.title(title, fontsize=18)
    plt.xlabel("Template Index", fontsize=16)
    plt.ylabel("Template Norm", fontsize=16)
    plt.grid(True)
    plt.xlim(0, data.shape[0] - 1)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def template_plot(data: numpy.ndarray, title: str, output_file: Path):
    """Create a simple plot of the data."""
    plt.figure(figsize=(12, 6))

    for row in data:
        plt.scatter(range(len(row)), row, alpha=0.4, s=2, color="blue")

    plt.title(title, fontsize=18)
    plt.xlabel("Element Index", fontsize=16)
    plt.ylabel("Element Value", fontsize=16)
    plt.grid(True)
    plt.xlim(0, data.shape[1] - 1)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def tsne_plot(data: numpy.ndarray, labels: list[str], title: str, output_file: Path):
    """Create a t-SNE plot of the data."""
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(data)

    plt.figure(figsize=(8, 8))
    plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="tab20", alpha=0.6
    )
    plt.title(title, fontsize=18)
    plt.xlabel("t-SNE Component 1", fontsize=16)
    plt.ylabel("t-SNE Component 2", fontsize=16)
    plt.grid(True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


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
def pipeline(
    system_config_file: Path,
    exp_config_file: Path,
    output_dir: Path | None,
):
    """Entry point to run a pipeline."""
    dataset_names = dict()
    dataset_names["icarb"] = "iCarB-Face"
    dataset_names["soteria"] = "SOTERIA"
    dataset_names["multipie"] = "Multi-PIE"

    baseline_names = dict()
    baseline_names["iresnet100"] = "iResNet100"
    baseline_names["iresnet50"] = "iResNet50"
    baseline_names["edgeface"] = "EdgeFace"
    baseline_names["edgefacexs"] = "EdgeFace-XS"
    baseline_names["facenet"] = "FaceNet"

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

    logger.info(f"Running distribution plots on {database_name}")

    # --------------------
    # Unprotected Pipeline
    # --------------------

    if baseline not in baseline_dict:
        logger.error(f"No baseline: `{baseline}`")
        return

    baseline_label = baseline

    bio_alg_cls = baseline_dict[baseline]
    bio_alg_args = (output_dir / baseline_label, save, detector)

    # Feature extraction on dist samples
    logger.info("FE on dist samples")
    dist_samples = list(dataset.load_irreversibility().references())
    dist_indexes = list(range(len(dist_samples)))
    with ThreadPoolExecutor(
        max_workers=n_worker,
        initializer=FRWorker.init,
        initargs=(
            bio_alg_cls,
            bio_alg_args,
            compliant,
            dist_samples,
            None,
            None,
        ),
    ) as exe:
        dist_templates = list(
            exe.map(FRWorker.feature_extraction, dist_indexes, chunksize=batch_size)
        )

        # Generate T-SNE plot for unprotected templates
        title = f"Element range: Unprotected templates \n (Model: {baseline_names[baseline]}, Dataset: {dataset_names[database_name]})"
        label = f"dist_{database_name}_{baseline}"
        plot_file = output_dir / f"{label}.png"
        matrix, _ = templates_to_matrix(dist_templates, precision=-1)
        template_plot(matrix, title, plot_file)

        norm_label = f"norm_{database_name}_{baseline}"
        norm_plot_file = output_dir / f"{norm_label}.png"
        norm_plot(matrix, norm_label, norm_plot_file)

    # Feature extraction on probe samples
    logger.info("FE on probe samples")
    samples = list(dataset.samples(n_subjects=50))
    indexes = list(range(len(samples)))
    with ThreadPoolExecutor(
        max_workers=n_worker,
        initializer=FRWorker.init,
        initargs=(
            bio_alg_cls,
            bio_alg_args,
            compliant,
            samples,
            None,
            None,
        ),
    ) as exe:
        templates = list(
            exe.map(FRWorker.feature_extraction, indexes, chunksize=batch_size)
        )

        # Generate T-SNE plot for unprotected templates
        title = f"t-SNE plot: Unprotected template clustering \n (Model: {baseline_names[baseline]}, Dataset: {dataset_names[database_name]})"
        label = f"tsne_{database_name}_{baseline}"
        plot_file = output_dir / f"{label}.png"
        matrix, subjects = templates_to_matrix(templates, precision=-1)
        tsne_plot(matrix, subjects, title, plot_file)

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

        btp_alg_extra_args: list[Any] = list()

        protected_baseline = protected_baseline_config["type"]
        if protected_baseline not in protected_baseline_dict:
            logger.error(f"No protected baseline: {protected_baseline}")
            return
        btp_alg_cls = protected_baseline_dict[protected_baseline]

        # If the alg is combined, need to add an extra argument
        if protected_baseline == "combined":
            btp_alg_extra_args.append(protected_baseline_dict)

        btp_alg_args = (
            output_dir / baseline_label / protected_baseline,
            save,
            protected_baseline_config,
        ) + tuple(btp_alg_extra_args)

        btp_alg = btp_alg_cls(*btp_alg_args)

        # Protect templates
        logger.info("Protecting probe templates")
        with ProcessPoolExecutor(
            max_workers=n_worker,
            initializer=BTPWorker.init,
            initargs=(
                btp_alg_cls,
                btp_alg_args,
                compliant,
                templates,
                None,
                None,
            ),
        ) as exe:
            prot_templates = list(
                exe.map(BTPWorker.protect, indexes, chunksize=batch_size)
            )

            # Generate T-SNE plot for protected templates
            title = f"t-SNE plot: PolyProtected template clustering \n (Model: {baseline_names[baseline]}, Dataset: {dataset_names[database_name]}, Overlap: 3)"
            label = f"tsne_{database_name}_{baseline}_{btp_alg.get_alg_name()}"
            plot_file = output_dir / f"{label}.png"
            matrix, subjects = templates_to_matrix(prot_templates, precision=-1)  # type: ignore
            tsne_plot(matrix, subjects, title, plot_file)

            # Generate Template plot for unprotected templates
            title = f"Element range: PolyProtected templates \n (Model: {baseline_names[baseline]}, Dataset: {dataset_names[database_name]}, Overlap: 3)"
            label = f"dist_{database_name}_{baseline}_{btp_alg.get_alg_name()}"
            plot_file = output_dir / f"{label}.png"
            template_plot(matrix, title, plot_file)

    logger.info("Experiment finished!")


if __name__ == "__main__":
    pipeline()
