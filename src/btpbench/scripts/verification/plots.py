# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import logging

from pathlib import Path

import bob.measure
import click
import matplotlib
import matplotlib.pyplot as plt
import pandas

import btpbench.metrics

from btpbench.scripts import pipeline_utils

matplotlib.use("Agg")

pipeline_utils.setup_logger()

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-f",
    "--score-file",
    "score_files",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    multiple=True,
    required=True,
    help=("Specify the score files to use."),
)
@click.option(
    "-l",
    "--label",
    "labels",
    type=str,
    multiple=True,
    required=False,
    help=("Specify the label of a given score file"),
)
@click.option(
    "-t",
    "--title",
    "title",
    type=str,
    default="DET curves",
    required=False,
    help=("Specify the plot's title"),
)
@click.option(
    "-o",
    "--output-file",
    "output_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=False, path_type=Path),
    required=True,
    help=("Specify the output file to store plot."),
)
def main(
    score_files: list[Path], labels: list[str], title: str, output_file: Path
) -> None:
    """Generate custom DET plots for verification results."""

    if len(labels) != len(score_files):
        logger.warning("Inconsitencies between labels and score files!")
        labels = [None] * len(score_files)

    plt.figure(figsize=(4, 3))
    plt.title(title)

    cols_dtypes = {
        "probe_subject_id": "str",
        "probe_template_id": "category",
        "bio_ref_subject_id": "str",
        "score": "float32",
    }
    cols_of_interest = list(cols_dtypes.keys())

    for score_file, label in zip(score_files, labels):
        logger.info(f"Processing {score_file}...")

        # If label is not specified, create a custom one.
        if label is None:
            baseline = "-".join(score_file.stem.split("-")[1:])
            dataset = score_file.parent.stem

            dataset = score_file.parent.stem
            label = f"{dataset}_{baseline}"

        neg: list[float] = []
        pos: list[float] = []

        for chunk in pandas.read_csv(
            score_file, chunksize=10**6, usecols=cols_of_interest, dtype=cols_dtypes
        ):
            _, clean_chunk = btpbench.metrics.remove_nans(chunk)

            neg_tmp, pos_tmp = btpbench.metrics.neg_pos_scores(clean_chunk)
            neg += neg_tmp.tolist()
            pos += pos_tmp.tolist()

        det_curve = bob.measure.det(neg, pos, n_points=2000)

        plt.plot(det_curve[0], det_curve[1], label=label)

    btpbench.metrics.setup_det_plot()
    plt.legend()
    plt.savefig(output_file, bbox_inches="tight")


if __name__ == "__main__":
    main()
