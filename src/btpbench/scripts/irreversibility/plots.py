# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import logging

from pathlib import Path

import bob.measure
import click
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas

import btpbench.metrics

from btpbench.scripts import pipeline_utils
from btpbench.scripts.irreversibility.analysis import (
    calculate_inversion_rates,
)

matplotlib.use("Agg")

pipeline_utils.setup_logger()

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-i",
    "--irr-file",
    "irr_files",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    multiple=True,
    required=True,
    help=("Specify the irreversibility files to use."),
)
@click.option(
    "-l",
    "--label",
    "labels",
    type=str,
    multiple=True,
    required=True,
    help=("Specify the label for a given irreversibility file to use."),
)
@click.option(
    "-s",
    "--id-file",
    "id_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    multiple=False,
    required=True,
    help=("Specify the biometric score file used to derive FMR thresholds."),
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
    "-t",
    "--title",
    "title",
    type=str,
    default="Inversion success rate vs FMR",
    required=False,
    help=("Specify the plot's title"),
)
@click.option(
    "-x",
    "--x-lim",
    "x_lim",
    type=float,
    default=0.1,
    required=False,
    help=("Specify FMR x-axis limit"),
)
def main(
    irr_files: list[Path],
    labels: list[str],
    id_file: Path,
    output_file: Path,
    title: str,
    x_lim: float,
) -> None:
    """Generate custom plots for irreversibility results."""

    if len(labels) != len(irr_files):
        raise click.UsageError(
            "Number of labels must match number of irreversibility files."
        )

    fmrs = numpy.linspace(0.0001, 0.5, num=450)
    _, id_df = btpbench.metrics.remove_nans(pandas.read_csv(id_file))

    id_neg, id_pos = btpbench.metrics.neg_pos_scores(id_df)

    thresholds: list[float] = [
        bob.measure.far_threshold(id_neg, id_pos, f) for f in fmrs
    ]

    plt.figure(figsize=(12, 4))
    plt.title(title)
    plt.xlabel("FMR (%)")
    plt.ylabel("Inversion Success Rate (%)")
    plt.xscale("log")
    plt.xlim(0.0001, x_lim)
    plt.xticks(
        [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        ["0.01", "0.1", "0.5", "1", "5", "10", "50"],
    )
    plt.ylim(0.01, 100.0)
    plt.grid(True)

    for i, score_file in enumerate(irr_files):
        logger.info(f"Processing {score_file}...")

        label = labels[i]

        rates = calculate_inversion_rates(score_file, thresholds)
        plt.plot(fmrs, rates.success_rates * 100, label=label)

    plt.legend()
    plt.savefig(output_file, bbox_inches="tight")


if __name__ == "__main__":
    main()
