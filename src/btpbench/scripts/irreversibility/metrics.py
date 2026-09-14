# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import csv
import logging

from pathlib import Path

import bob.measure
import click
import pandas

import btpbench.metrics

from btpbench.scripts import pipeline_utils
from btpbench.scripts.irreversibility.analysis import (
    calculate_inversion_rates,
)

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
    help=("Specify the label of a given score file"),
)
@click.option(
    "-s",
    "--id-file",
    "id_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    multiple=False,
    required=True,
    help=("Specify the biometric score file used to derive FAR thresholds."),
)
@click.option(
    "-o",
    "--output-file",
    "output_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=False, path_type=Path),
    required=True,
    help=("Specify the output CSV file."),
)
@click.option(
    "-f",
    "--far",
    "fars",
    type=float,
    multiple=True,
    help=("Specify the evaluation far(s)."),
)
def main(
    irr_files: list[Path],
    labels: list[str],
    id_file: Path,
    output_file: Path,
    fars: list[float],
) -> None:
    """Compute irreversibility rates at score-derived thresholds."""

    _, id_df = btpbench.metrics.remove_nans(pandas.read_csv(id_file))

    if len(labels) != len(irr_files):
        raise click.UsageError(
            "Number of labels must match number of irreversibility files."
        )

    id_neg, id_pos = btpbench.metrics.neg_pos_scores(id_df)
    thresholds: list[float] = [
        bob.measure.far_threshold(id_neg, id_pos, f) for f in fars
    ]

    # Compute TMRs for the given FARs
    # TMR is 1-FNMR
    tmrs: list[float] = [
        1 - bob.measure.farfrr(id_neg, id_pos, t)[1] for t in thresholds
    ]

    csv_file = output_file.open("w", newline="")
    csv_writer = csv.writer(csv_file)

    csv_headers = [
        "label",
        "solution_rate",
    ]

    for f in fars:
        csv_headers += [
            f"match_rate_far{f:.4f}",
            f"success_rate_far{f:.4f}",
            f"tmr_far{f:.4f}",
        ]

    csv_writer.writerow(csv_headers)

    for score_file, label in zip(irr_files, labels):
        logger.info(f"Processing {score_file}...")

        rates = calculate_inversion_rates(score_file, thresholds)
        row = [label, rates.solution_rate]

        for i in range(len(thresholds)):
            row.append(rates.match_rates[i])
            row.append(rates.success_rates[i])
            row.append(tmrs[i])

        csv_writer.writerow(row)

    csv_file.close()
    logger.info(f"Results written to {output_file}.")


if __name__ == "__main__":
    main()
