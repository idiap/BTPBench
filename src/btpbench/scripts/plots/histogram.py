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

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-u",
    "--unprotected-file",
    "unprotected_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    required=True,
    help=("Specify the unprotected score file."),
)
@click.option(
    "-i",
    "--inversion-file",
    "inversion_files",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    required=False,
    multiple=True,
    help=("Specify inversion score file(s) (can be used multiple times)."),
)
@click.option(
    "-l",
    "--inversion-label",
    "inversion_labels",
    type=str,
    required=False,
    multiple=True,
    help=("Specify label(s) for inversion scores (can be used multiple times)."),
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
    default="Score Distributions: Unprotected vs Inverted",
    required=False,
    help=("Specify the plot's title"),
)
@click.option(
    "-f",
    "--fmr",
    "fmrs",
    type=float,
    multiple=True,
    required=False,
    help=(
        "Specify FMR value(s) to display threshold lines (can be used multiple times)."
    ),
)
@click.option(
    "-c",
    "--inversion-color",
    "inversion_colors",
    type=str,
    required=False,
    multiple=True,
    help=("Specify color(s) for inversion histograms (can be used multiple times)."),
)
@click.option(
    "-H",
    "--inversion-hatch",
    "inversion_hatches",
    type=str,
    required=False,
    multiple=True,
    help=(
        "Specify hatch pattern(s) for inversion histograms (can be used multiple times)."
    ),
)
@click.option(
    "-v",
    "--verbose",
    "verbosity",
    type=click.IntRange(0, 4),
    default=3,
    help=(
        "Set verbosity level: 0=CRITICAL (silent), 1=ERROR, 2=WARNING, 3=INFO (default), 4=DEBUG."
    ),
)
def main(
    unprotected_file: Path,
    inversion_files: tuple[Path, ...],
    inversion_labels: tuple[str, ...],
    output_file: Path,
    title: str,
    fmrs: tuple[float, ...],
    inversion_colors: tuple[str, ...],
    inversion_hatches: tuple[str, ...],
    verbosity: int,
) -> None:
    """Plot unprotected genuine/impostor vs inverted score distributions."""

    pipeline_utils.setup_logger_from_verbosity(verbosity)

    # Validate that if labels are provided, they match the number of files
    if inversion_labels and len(inversion_labels) != len(inversion_files):
        raise ValueError(
            f"Number of inversion labels ({len(inversion_labels)}) must match "
            f"number of inversion files ({len(inversion_files)})"
        )

    if inversion_colors and len(inversion_colors) != len(inversion_files):
        raise ValueError(
            f"Number of inversion colors ({len(inversion_colors)}) must match "
            f"number of inversion files ({len(inversion_files)})"
        )

    if inversion_hatches and len(inversion_hatches) != len(inversion_files):
        raise ValueError(
            f"Number of inversion hatches ({len(inversion_hatches)}) must match "
            f"number of inversion files ({len(inversion_files)})"
        )

    # If no labels provided, generate default ones
    if not inversion_labels and inversion_files:
        inversion_labels = tuple(
            f"Inverted Scores {i + 1}" for i in range(len(inversion_files))
        )

    logger.info(f"Loading unprotected scores from {unprotected_file}")
    unprotected_df = pandas.read_csv(unprotected_file)

    # Remove NaN scores and get genuine/impostor distributions
    fta_unprotected, unprotected_df = btpbench.metrics.remove_nans(unprotected_df)

    neg, pos = btpbench.metrics.neg_pos_scores(unprotected_df)

    logger.info(f"Unprotected FTA: {fta_unprotected:.3f}")
    logger.info(f"Genuine scores: {len(pos)}")
    logger.info(f"Impostor scores: {len(neg)}")

    # Load inversion scores if provided
    inversion_data = []
    if inversion_files:
        for idx, inversion_file in enumerate(inversion_files):
            logger.info(f"Loading inversion scores from {inversion_file}")
            inversion_df = pandas.read_csv(inversion_file)
            fta_inversion, inversion_df = btpbench.metrics.remove_nans(inversion_df)
            inversion_scores = inversion_df["score"].to_numpy()
            logger.info(f"Inversion FTA: {fta_inversion:.3f}")
            logger.info(f"Inversion scores: {len(inversion_scores)}")

            inversion_data.append(
                {"label": inversion_labels[idx], "scores": inversion_scores}
            )

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Calculate common bin range
    all_scores = [neg, pos]
    for inv_data in inversion_data:
        all_scores.append(inv_data["scores"])
    all_scores = numpy.concatenate(all_scores)
    score_min, score_max = numpy.min(all_scores), numpy.max(all_scores)
    bins = numpy.linspace(score_min, score_max, 50)

    # Plot histograms
    plt.hist(
        neg,
        bins=bins,
        alpha=0.5,
        label="Impostor (Unprotected)",
        color="red",
        density=True,
        edgecolor="black",
        linewidth=0.5,
    )

    plt.hist(
        pos,
        bins=bins,
        alpha=0.5,
        label="Genuine (Unprotected)",
        color="green",
        density=True,
        edgecolor="black",
        linewidth=0.5,
    )

    # Plot each inversion file with a different color
    if inversion_data:
        default_colors = [
            "blue",
            "purple",
            "orange",
            "gray",
            "cyan",
            "magenta",
            "brown",
            "pink",
        ]
        default_hatches = ["//", "\\\\", "..", "||", "++", "xx", "oo", "**"]

        for idx, inv_data in enumerate(inversion_data):
            color = (
                inversion_colors[idx]
                if inversion_colors
                else default_colors[idx % len(default_colors)]
            )
            hatch = (
                inversion_hatches[idx]
                if inversion_hatches
                else default_hatches[idx % len(default_hatches)]
            )
            plt.hist(
                inv_data["scores"],
                bins=bins,
                alpha=0.5,
                label=inv_data["label"],
                color=color,
                density=True,
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch,
            )

    plt.xlim(-1.3, 0)
    plt.ylim(0, 15)
    plt.xlabel("Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add FMR threshold lines if specified
    if fmrs:
        logger.info(f"Adding threshold lines for FMR values: {fmrs}")
        linestyles = [
            "-",
            "--",
            "-.",
            ":",
            (0, (3, 1, 1, 1)),
        ]  # solid, dashed, dashdot, dotted, densely dashdotdotted
        for i, fmr in enumerate(fmrs):
            threshold = bob.measure.far_threshold(neg, pos, fmr)
            linestyle = linestyles[i % len(linestyles)]
            plt.axvline(
                x=threshold,
                color="black",
                linestyle=linestyle,
                linewidth=1.5,
                label=f"FMR={fmr * 100:.2f}% (t={threshold:.3f})",
            )
            logger.info(f"FMR {fmr * 100:.2f}% -> threshold {threshold:.3f}")

        # Update legend to include threshold lines
        plt.legend()

    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    logger.info(f"Plot saved to {output_file}")


if __name__ == "__main__":
    main()
