# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import csv
import logging

from pathlib import Path

import bob.measure
import click
import numpy
import pandas

import btpbench
import btpbench.metrics

from btpbench.scripts import pipeline_utils

pipeline_utils.setup_logger()

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-s",
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
    required=True,
    help="Specify one label for each score file.",
)
@click.option(
    "-o",
    "--output-file",
    "output_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=False, path_type=Path),
    required=True,
    help=("Specify the output file to store plot."),
)
@click.option(
    "-f",
    "--fmr",
    "fmrs",
    type=float,
    multiple=True,
    default=[],
    help=("Specify the performance fmrs."),
)
def metrics(
    score_files: list[Path], labels: list[str], output_file: Path, fmrs: list[float]
):
    """Entry point to analyze results."""

    if len(score_files) != len(labels):
        raise click.ClickException(
            "The number of score files must match the number of labels."
        )

    csv_headers = [
        "label",
        "fta",
    ]

    for f in fmrs:
        csv_headers += [
            f"fmr_{f * 10000}",
            f"fnmr_{f * 10000}",
        ]

    csv_file = Path.open(output_file, "w", newline="")
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(csv_headers)

    for score_file, label in zip(score_files, labels):
        logger.info(f"processing `{score_file}`...")

        neg: list[float] = []
        pos: list[float] = []

        ftas: list[float] = []
        ftas_weights: list[float] = []

        logger.info("-> reading score file")

        for chunk in pandas.read_csv(score_file, chunksize=10**6):
            fta, clean_chunk = btpbench.metrics.remove_nans(chunk)

            ftas.append(fta)
            ftas_weights.append(chunk.shape[0])

            neg_tmp, pos_tmp = btpbench.metrics.neg_pos_scores(clean_chunk)
            neg += neg_tmp.tolist()
            pos += pos_tmp.tolist()

        logger.info("-> computing metrics")

        fta = numpy.average(ftas, weights=ftas_weights)
        row = [
            label,
            fta,
        ]

        for f in fmrs:
            t = bob.measure.far_threshold(neg, pos, f)
            fmr, fnmr = bob.measure.farfrr(neg, pos, t)
            row += [fmr, fnmr]

        csv_file_writer.writerow(row)

    csv_file.close()


if __name__ == "__main__":
    metrics()
