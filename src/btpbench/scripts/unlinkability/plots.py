# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

"""Density plots for unlinkability mated and non-mated score files."""

import json
import logging

from pathlib import Path
from typing import Any

import click
import matplotlib
import numpy
import pandas

from scipy.ndimage import gaussian_filter1d

from btpbench.scripts import pipeline_utils
from btpbench.scripts.unlinkability import metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt

pipeline_utils.setup_logger()
logger = logging.getLogger(__name__)


def _flatten_json_numbers(value: Any) -> list[float]:
    """Return finite numeric values from a JSON scalar/list/dict."""
    if value is None:
        return []
    if isinstance(value, dict):
        values: list[float] = []
        for item in value.values():
            values.extend(_flatten_json_numbers(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(_flatten_json_numbers(item))
        return values
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return []
    if not numpy.isfinite(numeric):
        return []
    return [numeric]


def _load_optimized_mated_scores(score_file: Path) -> tuple[numpy.ndarray, int]:
    """Load maximum-clique scores and count unique clique keys."""
    scores: list[float] = []
    unique_keys: set[Any] = set()
    df = pandas.read_csv(score_file)

    if {"selected_keys", "selected_key_scores"}.issubset(df.columns):
        keys_column = "selected_keys"
        scores_column = "selected_key_scores"
    else:
        score_columns = [
            column for column in df.columns if column.endswith("_clique_scores")
        ]
        if len(score_columns) != 1:
            raise click.ClickException(
                "Optimized mated scores require a diversity CSV containing "
                "exactly one '*_clique_scores' column."
            )
        scores_column = score_columns[0]
        keys_column = f"{scores_column.removesuffix('_clique_scores')}_clique_keys"
        if keys_column not in df.columns:
            raise click.ClickException(
                f"Could not read matching clique-key column '{keys_column}' "
                f"from {score_file}."
            )

    for _, row in df.iterrows():
        if pandas.isna(row[keys_column]) or pandas.isna(row[scores_column]):
            continue
        selected_keys = json.loads(row[keys_column])
        selected_scores = json.loads(row[scores_column])
        for key in _flatten_json_numbers(selected_keys):
            unique_keys.add(key)
        scores.extend(_flatten_json_numbers(selected_scores))

    if not scores:
        raise click.ClickException(f"No selected key scores found in {score_file}.")

    return numpy.asarray(scores, dtype=float), len(unique_keys)


def _density(
    scores: numpy.ndarray,
    edges: numpy.ndarray,
    smooth_sigma: float,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Estimate a smooth probability density on fixed histogram bins."""
    counts, _ = numpy.histogram(scores, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])
    total = counts.sum()
    if total == 0:
        raise click.ClickException("No scores fall inside the requested x-axis range.")
    density = counts.astype(float) / (total * bin_width)

    if smooth_sigma > 0:
        density = gaussian_filter1d(density, smooth_sigma, mode="nearest")
        density = numpy.maximum(density, 0.0)
        area = metrics.trapezoid(density, centers)
        if area > 0:
            density /= area

    return centers, density


def _density_region_annotations(
    x_values: numpy.ndarray,
    mated_density: numpy.ndarray,
    non_mated_density: numpy.ndarray,
    d_local_above_threshold_regions: list[tuple[float, float, int]],
    optimized_mated_density: numpy.ndarray | None = None,
) -> tuple[float, list[tuple[float, float, int]]]:
    """Place sample-count annotations above density peaks."""
    densities = [mated_density, non_mated_density]
    if optimized_mated_density is not None:
        densities.append(optimized_mated_density)
    density_peak = max(float(numpy.max(density)) for density in densities)
    annotation_offset = 0.06 * density_peak
    annotations: list[tuple[float, float, int]] = []

    for score_min, score_max, mated_sample_count in d_local_above_threshold_regions:
        annotation_x = 0.5 * (score_min + score_max)
        in_region = (x_values >= score_min) & (x_values <= score_max)
        if numpy.any(in_region):
            region_density_peak = max(
                float(numpy.max(density[in_region])) for density in densities
            )
        else:
            region_density_peak = max(
                float(numpy.interp(annotation_x, x_values, density))
                for density in densities
            )
        annotations.append(
            (
                float(annotation_x),
                float(region_density_peak + annotation_offset),
                mated_sample_count,
            )
        )

    return density_peak, annotations


def _plotly_modules() -> tuple[Any, Any]:
    """Import Plotly only when interactive HTML output is requested."""
    try:
        import plotly.graph_objects as go

        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise click.ClickException(
            "Interactive HTML output requires Plotly. Install the project "
            "dependencies, then rerun the command with an .html output file."
        ) from exc
    return go, make_subplots


def _save_interactive_plot(
    output_file: Path,
    title: str,
    x_values: numpy.ndarray,
    mated_density: numpy.ndarray,
    non_mated_density: numpy.ndarray,
    optimized_mated_density: numpy.ndarray | None,
    metric_centers: numpy.ndarray,
    d_local: numpy.ndarray,
    d_system: float,
    d_local_threshold: float,
    metric_bins: int,
    d_local_transition_scores: numpy.ndarray,
    density_peak: float,
    density_annotations: list[tuple[float, float, int]],
    left: float,
    right: float,
    mated_label: str,
    non_mated_label: str,
    optimized_mated_label: str | None,
) -> None:
    """Save an interactive Plotly HTML figure."""
    go, make_subplots = _plotly_modules()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=mated_density,
            mode="lines",
            name=mated_label,
            line={"color": "#2fb34a", "width": 2},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=non_mated_density,
            mode="lines",
            name=non_mated_label,
            line={"color": "#e65252", "width": 2, "dash": "dash"},
        ),
        secondary_y=False,
    )
    if optimized_mated_density is not None and optimized_mated_label is not None:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=optimized_mated_density,
                mode="lines",
                name=optimized_mated_label,
                line={"color": "#7a52cc", "width": 2, "dash": "dash"},
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=metric_centers,
            y=d_local,
            mode="lines+markers",
            name="D<sub>&harr;</sub>(s)",
            line={"color": "#3f5f99", "width": 3},
            marker={"size": 6},
        ),
        secondary_y=True,
    )

    shapes = [
        {
            "type": "line",
            "xref": "x",
            "yref": "paper",
            "x0": float(transition_score),
            "x1": float(transition_score),
            "y0": 0,
            "y1": 1,
            "line": {"color": "#3f5f99", "width": 1, "dash": "dot"},
            "opacity": 0.65,
        }
        for transition_score in d_local_transition_scores
    ]

    for annotation_x, annotation_y, mated_sample_count in density_annotations:
        fig.add_annotation(
            x=annotation_x,
            y=annotation_y,
            text=str(mated_sample_count),
            xref="x",
            yref="y",
            showarrow=False,
            font={"color": "#1d1d1d", "size": 12},
            yanchor="bottom",
        )

    fig.update_layout(
        title={
            "text": (
                f"{title}<br>"
                "D<sub>&harr;</sub><sup>sys</sup>"
                f" = {d_system:.2f}, C = {d_local_threshold:.2f}, "
                f"metric bins = {metric_bins}"
            )
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
        hovermode="x unified",
        shapes=shapes,
        template="plotly_white",
        margin={"l": 70, "r": 70, "t": 95, "b": 60},
    )
    fig.update_xaxes(title_text="Score", range=[left, right])
    fig.update_yaxes(
        title_text="Probability Density",
        range=[
            0,
            max(
                1.25 * density_peak,
                max((y for _, y, _ in density_annotations), default=0.0)
                + 0.2 * density_peak,
            ),
        ],
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="D<sub>&harr;</sub>(s)",
        range=[0, 1.1],
        secondary_y=True,
    )
    fig.write_html(output_file, include_plotlyjs=True, full_html=True)


def _save_static_plot(
    output_file: Path,
    title: str,
    x_values: numpy.ndarray,
    mated_density: numpy.ndarray,
    non_mated_density: numpy.ndarray,
    optimized_mated_density: numpy.ndarray | None,
    metric_centers: numpy.ndarray,
    d_local: numpy.ndarray,
    d_system: float,
    d_local_threshold: float,
    metric_bins: int,
    d_local_transition_scores: numpy.ndarray,
    density_peak: float,
    density_annotations: list[tuple[float, float, int]],
    left: float,
    right: float,
    mated_label: str,
    non_mated_label: str,
    optimized_mated_label: str | None,
) -> None:
    """Save a static Matplotlib figure."""
    fig, ax = plt.subplots(figsize=(4.7, 3.6), constrained_layout=True)
    mated_line = ax.plot(
        x_values,
        mated_density,
        color="#2fb34a",
        linewidth=1.8,
        label=mated_label,
    )[0]
    non_mated_line = ax.plot(
        x_values,
        non_mated_density,
        color="#e65252",
        linestyle="--",
        linewidth=1.8,
        label=non_mated_label,
    )[0]
    legend_lines = [mated_line, non_mated_line]
    legend_labels = [mated_label, non_mated_label]
    if optimized_mated_density is not None and optimized_mated_label is not None:
        optimized_mated_line = ax.plot(
            x_values,
            optimized_mated_density,
            color="#7a52cc",
            linestyle="-.",
            linewidth=1.8,
            label=optimized_mated_label,
        )[0]
        legend_lines.append(optimized_mated_line)
        legend_labels.append(optimized_mated_label)

    ax2 = ax.twinx()
    d_line = ax2.plot(
        metric_centers,
        d_local,
        color="#3f5f99",
        linewidth=2.5,
        label=r"$D_{\leftrightarrow}(s)$",
    )[0]
    for transition_score in d_local_transition_scores:
        ax2.axvline(
            transition_score,
            color="#3f5f99",
            linestyle=":",
            linewidth=1.0,
            alpha=0.65,
        )

    for annotation_x, annotation_y, mated_sample_count in density_annotations:
        ax.text(
            annotation_x,
            annotation_y,
            str(mated_sample_count),
            color="#1d1d1d",
            fontsize="small",
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    ax.set_title(
        f"{title}\n"
        r"$D_{\leftrightarrow}^{sys}$"
        f" = {d_system:.2f}, C = {d_local_threshold:.2f}, "
        f"metric bins = {metric_bins}"
    )
    ax.set_xlabel("Score")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(left, right)
    ax.set_ylim(
        0,
        max(
            1.25 * density_peak,
            max((y for _, y, _ in density_annotations), default=0.0)
            + 0.2 * density_peak,
        ),
    )
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel(r"$D_{\leftrightarrow}(s)$")
    ax.legend(
        [*legend_lines, d_line],
        [*legend_labels, r"$D_{\leftrightarrow}(s)$"],
        loc="upper center",
        ncol=2 if optimized_mated_density is not None else 3,
        fontsize="small",
        frameon=True,
    )

    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.option(
    "-m",
    "--mated-score-file",
    "mated_score_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    required=True,
    help="Mated unlinkability score CSV.",
)
@click.option(
    "-n",
    "--non-mated-score-file",
    "non_mated_score_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    required=True,
    help="Non-mated unlinkability score CSV.",
)
@click.option(
    "--optimized-mated-score-file",
    "optimized_mated_score_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    default=None,
    help=(
        "Diversity CSV containing one criterion's clique keys and scores. "
        "Legacy selected_keys/selected_key_scores CSVs are also accepted."
    ),
)
@click.option(
    "-t",
    "--title",
    "title",
    type=str,
    required=True,
    help="Plot title.",
)
@click.option(
    "-o",
    "--output-file",
    "output_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=False, path_type=Path),
    required=True,
    help="Output file for the plot. Use .html or .htm for interactive output.",
)
@click.option(
    "--mated-label",
    "mated_label",
    type=str,
    default="Mated",
    show_default=True,
    help="Legend label for the mated scores.",
)
@click.option(
    "--non-mated-label",
    "non_mated_label",
    type=str,
    default="Non-Mated",
    show_default=True,
    help="Legend label for the non-mated scores.",
)
@click.option(
    "--bins",
    "bins",
    type=click.IntRange(min=10),
    default=512,
    show_default=True,
    help="Number of histogram bins used for density estimation.",
)
@click.option(
    "--smooth-sigma",
    "smooth_sigma",
    type=click.FloatRange(min=0.0),
    default=2.0,
    show_default=True,
    help="Gaussian smoothing sigma in histogram-bin units.",
)
@click.option(
    "--metric-bins",
    "metric_bins",
    type=click.IntRange(min=2),
    default=10,
    show_default=True,
    help="Number of bins used to compute D(s) and Dsys.",
)
@click.option(
    "--omega",
    "omega",
    type=click.FloatRange(min=0.0, min_open=True),
    default=1.0,
    show_default=True,
    help="Prior ratio omega used in the unlinkability metric.",
)
@click.option(
    "--d-local-threshold",
    "--d-local-match-value",
    "d_local_threshold",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.0,
    show_default=True,
    help="Local D(s) threshold C; D(s) <= C regions are marked.",
)
@click.option(
    "--x-min",
    "x_min",
    type=float,
    default=None,
    help="Minimum score shown on the x-axis.",
)
@click.option(
    "--x-max",
    "x_max",
    type=float,
    default=None,
    help="Maximum score shown on the x-axis.",
)
def main(
    mated_score_file: Path,
    non_mated_score_file: Path,
    optimized_mated_score_file: Path | None,
    title: str,
    output_file: Path,
    mated_label: str,
    non_mated_label: str,
    bins: int,
    smooth_sigma: float,
    metric_bins: int,
    omega: float,
    d_local_threshold: float,
    x_min: float | None,
    x_max: float | None,
) -> None:
    """Plot mated and non-mated unlinkability score distributions."""
    logger.info("Loading mated scores from %s", mated_score_file)
    mated_scores = metrics.load_scores(mated_score_file)
    logger.info("Loading non-mated scores from %s", non_mated_score_file)
    non_mated_scores = metrics.load_scores(non_mated_score_file)

    optimized_mated_scores = None
    optimized_mated_label = None
    if optimized_mated_score_file is not None:
        logger.info(
            "Loading optimized mated scores from %s",
            optimized_mated_score_file,
        )
        optimized_mated_scores, n_unique_keys = _load_optimized_mated_scores(
            optimized_mated_score_file,
        )
        optimized_mated_label = f"Mated (Selected, {n_unique_keys} keys)"

    metric = metrics.compute_unlinkability_metric(
        mated_scores,
        non_mated_scores,
        metric_bins=metric_bins,
        omega=omega,
        optimized_mated_scores=optimized_mated_scores,
        x_min=x_min,
        x_max=x_max,
    )
    left, right = metric.left, metric.right
    edges = numpy.linspace(left, right, bins + 1)

    x_values, mated_density = _density(mated_scores, edges, smooth_sigma)
    _, non_mated_density = _density(non_mated_scores, edges, smooth_sigma)
    optimized_mated_density = None
    if optimized_mated_scores is not None:
        _, optimized_mated_density = _density(
            optimized_mated_scores,
            edges,
            smooth_sigma,
        )

    metric_edges = metric.edges
    metric_centers = metric.centers
    d_local = metric.d_local
    d_system = metric.d_system
    logger.info("Dsys: %.6f", d_system)

    d_local_transition_scores = metrics.d_local_transition_scores(
        metric_centers,
        d_local,
        d_local_threshold,
    )
    d_local_above_threshold_regions = metrics.d_local_above_threshold_regions(
        metric_edges,
        d_local,
        d_local_threshold,
        mated_scores,
    )
    d_local_threshold_ranges = metrics.d_local_at_or_below_threshold_ranges(
        metric_edges,
        d_local,
        d_local_threshold,
    )
    if d_local_threshold_ranges:
        for index, (score_min, score_max) in enumerate(
            d_local_threshold_ranges,
            start=1,
        ):
            logger.info(
                "D(s) <= C range %d (C=%.6f): [%.6f, %.6f]",
                index,
                d_local_threshold,
                score_min,
                score_max,
            )
    else:
        logger.info("No D(s) <= C range found (C=%.6f).", d_local_threshold)
    density_peak, density_annotations = _density_region_annotations(
        x_values,
        mated_density,
        non_mated_density,
        d_local_above_threshold_regions,
        optimized_mated_density,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.suffix.lower() in {".html", ".htm"}:
        _save_interactive_plot(
            output_file,
            title,
            x_values,
            mated_density,
            non_mated_density,
            optimized_mated_density,
            metric_centers,
            d_local,
            d_system,
            d_local_threshold,
            metric_bins,
            d_local_transition_scores,
            density_peak,
            density_annotations,
            left,
            right,
            mated_label,
            non_mated_label,
            optimized_mated_label,
        )
    else:
        _save_static_plot(
            output_file,
            title,
            x_values,
            mated_density,
            non_mated_density,
            optimized_mated_density,
            metric_centers,
            d_local,
            d_system,
            d_local_threshold,
            metric_bins,
            d_local_transition_scores,
            density_peak,
            density_annotations,
            left,
            right,
            mated_label,
            non_mated_label,
            optimized_mated_label,
        )
    logger.info("Saved plot to %s", output_file)


if __name__ == "__main__":
    main()
