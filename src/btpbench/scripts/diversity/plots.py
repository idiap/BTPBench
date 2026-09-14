# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

"""Violin plots for per-subject diversity results."""

from pathlib import Path
from textwrap import fill

import click
import matplotlib
import numpy
import pandas

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from btpbench.scripts.diversity.summary import _subject_rows

_DIVERSITY_SUFFIXES = ("_clique_size", "_practical_N", "_mean", "_N")


def _format_number(value: float) -> str:
    """Format a numeric criterion value compactly."""
    return f"{value:g}"


def _criterion_label(column: str) -> str:
    """Build a readable x-axis label from a diversity column name."""
    criterion = column
    for suffix in _DIVERSITY_SUFFIXES:
        if criterion.endswith(suffix):
            criterion = criterion.removesuffix(suffix)
            break
    if criterion.startswith("fmr_"):
        try:
            fmr = float(criterion.removeprefix("fmr_"))
        except ValueError:
            return criterion.replace("_", " ")
        return f"FMR {_format_number(fmr * 100)}%"

    for prefix in ("score_ranges_", "score_range_"):
        if not criterion.startswith(prefix):
            continue
        ranges = []
        for score_range in criterion.removeprefix(prefix).split("__"):
            try:
                score_min, score_max = score_range.split("_", maxsplit=1)
                ranges.append(
                    f"[{_format_number(float(score_min))}, "
                    f"{_format_number(float(score_max))}]"
                )
            except ValueError:
                return criterion.replace("_", " ")
        label = "Score range" if len(ranges) == 1 else "Score ranges"
        return f"{label} {' or '.join(ranges)}"

    return criterion.replace("_", " ")


def load_diversity_distributions(
    result_file: Path,
) -> tuple[list[str], list[numpy.ndarray]]:
    """Load one diversity distribution per criterion from a result CSV."""
    results = pandas.read_csv(result_file)
    subjects = _subject_rows(results)
    size_columns = []
    for suffix in _DIVERSITY_SUFFIXES:
        size_columns = [
            column for column in subjects.columns if column.endswith(suffix)
        ]
        if size_columns:
            break
    if not size_columns:
        raise click.ClickException(
            f"No supported diversity columns found in {result_file}. Expected "
            "'*_clique_size', '*_practical_N', '*_mean', or '*_N'."
        )

    labels = []
    distributions = []
    for column in size_columns:
        values = pandas.to_numeric(subjects[column], errors="coerce").dropna()
        if values.empty:
            continue
        labels.append(_criterion_label(column))
        distributions.append(values.to_numpy(dtype=float))

    if not distributions:
        raise click.ClickException(
            f"No per-subject diversity values found in {result_file}."
        )
    return labels, distributions


def _violin_values(values: numpy.ndarray) -> numpy.ndarray:
    """Make constant or singleton data suitable for kernel density estimation."""
    if len(values) > 1 and numpy.ptp(values) > 0:
        return values
    center = float(values[0])
    epsilon = 0.05
    return numpy.asarray([center - epsilon, center + epsilon])


def plot_diversity_distributions(
    result_file: Path,
    output_file: Path,
    title: str,
) -> None:
    """Plot per-subject diversity distributions from one experiment result."""
    labels, distributions = load_diversity_distributions(result_file)
    positions = numpy.arange(1, len(distributions) + 1)
    plot_values = [_violin_values(values) for values in distributions]

    width = max(7.0, 2.2 * len(distributions))
    fig, ax = plt.subplots(figsize=(width, 6), constrained_layout=True)
    fig.suptitle(fill(title, width=75), fontsize=14)
    violin_parts = ax.violinplot(
        plot_values,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin_parts["bodies"]:
        body.set_facecolor("#4C78A8")
        body.set_edgecolor("#2F4B63")
        body.set_alpha(0.75)

    minima = numpy.asarray([values.min() for values in distributions])
    maxima = numpy.asarray([values.max() for values in distributions])
    means = numpy.asarray([values.mean() for values in distributions])
    medians = numpy.asarray([numpy.median(values) for values in distributions])
    ax.vlines(positions, minima, maxima, color="#2F4B63", linewidth=1.2)
    ax.hlines(
        medians,
        positions - 0.16,
        positions + 0.16,
        color="white",
        linewidth=2.0,
        label="Median",
    )
    ax.scatter(
        positions,
        means,
        color="#E45756",
        edgecolor="white",
        linewidth=0.7,
        s=42,
        zorder=3,
        label="Mean",
    )

    ax.set_xlabel("Diversity criterion")
    ax.set_ylabel("Diversity")
    ax.set_xticks(positions, labels)
    all_values = numpy.concatenate(distributions)
    if numpy.allclose(all_values, numpy.round(all_values)):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(max(0, numpy.floor(minima.min()) - 1), numpy.ceil(maxima.max()) + 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


@click.command(name="plots")
@click.option(
    "-f",
    "--result-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Diversity result CSV for one experiment configuration.",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output plot filename, for example diversity-violin.png.",
)
@click.option(
    "-t",
    "--title",
    default="Diversity distribution across subjects",
    show_default=True,
    help="Plot title.",
)
def plots(result_file: Path, output_file: Path, title: str) -> None:
    """Plot subject-level diversity distributions as violins."""
    plot_diversity_distributions(result_file, output_file, title)
    click.echo(f"Wrote {output_file}")


if __name__ == "__main__":
    plots()
