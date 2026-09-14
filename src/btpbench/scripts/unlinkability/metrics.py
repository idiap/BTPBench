# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

"""Shared unlinkability metric calculations."""

from dataclasses import dataclass
from pathlib import Path

import click
import numpy
import pandas


@dataclass(frozen=True)
class UnlinkabilityMetric:
    """Binned local and global unlinkability values over one score range."""

    left: float
    right: float
    edges: numpy.ndarray
    centers: numpy.ndarray
    d_local: numpy.ndarray
    d_system: float


def trapezoid(y_values: numpy.ndarray, x_values: numpy.ndarray) -> float:
    """Integrate with NumPy 2's trapezoid API while keeping NumPy 1.x support."""
    trapezoid_fn = getattr(numpy, "trapezoid", None)
    if trapezoid_fn is None:
        trapezoid_fn = getattr(numpy, "trapz")
    return float(trapezoid_fn(y=y_values, x=x_values))


def load_scores(score_file: Path) -> numpy.ndarray:
    """Load finite scores from a score CSV file."""
    scores: list[numpy.ndarray] = []
    try:
        for chunk in pandas.read_csv(
            score_file,
            usecols=["score"],
            dtype={"score": "float64"},
            chunksize=10**6,
        ):
            values = chunk["score"].to_numpy(dtype=float)
            scores.append(values[numpy.isfinite(values)])
    except ValueError as exc:
        raise click.ClickException(
            f"Could not read a 'score' column from {score_file}."
        ) from exc

    if not scores:
        raise click.ClickException(f"No scores found in {score_file}.")

    merged = numpy.concatenate(scores)
    if len(merged) == 0:
        raise click.ClickException(f"No finite scores found in {score_file}.")
    return merged


def score_range(
    mated_scores: numpy.ndarray,
    non_mated_scores: numpy.ndarray,
    optimized_mated_scores: numpy.ndarray | None,
    x_min: float | None,
    x_max: float | None,
) -> tuple[float, float]:
    """Compute the score range used for unlinkability metric bins."""
    score_sets = [mated_scores, non_mated_scores]
    if optimized_mated_scores is not None:
        score_sets.append(optimized_mated_scores)

    data_min = min(float(numpy.min(scores)) for scores in score_sets)
    data_max = max(float(numpy.max(scores)) for scores in score_sets)

    left = data_min if x_min is None else x_min
    right = data_max if x_max is None else x_max
    if left >= right:
        raise click.ClickException("--x-min must be lower than --x-max.")

    if x_min is None and x_max is None:
        padding = 0.02 * (right - left)
        if padding == 0:
            padding = 0.5
        left -= padding
        right += padding

    return left, right


def unlinkability_metric(
    mated_scores: numpy.ndarray,
    non_mated_scores: numpy.ndarray,
    edges: numpy.ndarray,
    omega: float,
) -> tuple[numpy.ndarray, numpy.ndarray, float]:
    """Compute local and global unlinkability from binned score densities."""
    mated_density, _ = numpy.histogram(mated_scores, bins=edges, density=True)
    non_mated_density, _ = numpy.histogram(
        non_mated_scores,
        bins=edges,
        density=True,
    )
    centers = 0.5 * (edges[1:] + edges[:-1])

    likelihood_ratio = numpy.divide(
        mated_density,
        non_mated_density,
        out=numpy.ones_like(mated_density),
        where=non_mated_density != 0,
    )
    weighted_lr = omega * likelihood_ratio
    d_local = 2 * (weighted_lr / (1 + weighted_lr)) - 1
    d_local[weighted_lr <= 1] = 0
    d_local[non_mated_density == 0] = 1

    d_system = trapezoid(d_local * mated_density, centers)
    return centers, d_local, d_system


def compute_unlinkability_metric(
    mated_scores: numpy.ndarray,
    non_mated_scores: numpy.ndarray,
    *,
    metric_bins: int,
    omega: float,
    optimized_mated_scores: numpy.ndarray | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
) -> UnlinkabilityMetric:
    """Compute the binned unlinkability metric exactly as the plot command does."""
    if metric_bins < 2:
        raise ValueError("At least two metric bins are required.")

    left, right = score_range(
        mated_scores,
        non_mated_scores,
        optimized_mated_scores,
        x_min,
        x_max,
    )
    edges = numpy.linspace(left, right, metric_bins + 1)
    centers, d_local, d_system = unlinkability_metric(
        mated_scores,
        non_mated_scores,
        edges,
        omega,
    )
    return UnlinkabilityMetric(left, right, edges, centers, d_local, d_system)


def region_indexes(mask: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Find the first and last indexes of contiguous true mask regions."""
    transitions = numpy.diff(mask.astype(numpy.int8), prepend=0, append=0)
    return (
        numpy.flatnonzero(transitions == 1),
        numpy.flatnonzero(transitions == -1) - 1,
    )


def d_local_transition_scores(
    scores: numpy.ndarray,
    d_local: numpy.ndarray,
    threshold: float,
) -> numpy.ndarray:
    """Return D(s) score centers at threshold transitions."""
    if scores.shape != d_local.shape:
        raise ValueError("D(s) scores and values must have the same shape.")
    if scores.ndim != 1:
        raise ValueError("D(s) scores and values must be one-dimensional.")

    above_threshold = d_local > threshold
    neighboring_above_threshold = numpy.zeros_like(above_threshold)
    neighboring_above_threshold[1:] |= above_threshold[:-1]
    neighboring_above_threshold[:-1] |= above_threshold[1:]
    return scores[(d_local <= threshold) & neighboring_above_threshold]


def d_local_at_or_below_threshold_ranges(
    score_edges: numpy.ndarray,
    d_local: numpy.ndarray,
    threshold: float,
) -> list[tuple[float, float]]:
    """Return score ranges where D(s) is at or below the local threshold."""
    if score_edges.ndim != 1 or d_local.ndim != 1:
        raise ValueError("D(s) score edges and values must be one-dimensional.")
    if len(score_edges) != len(d_local) + 1:
        raise ValueError("D(s) score edges must bound each D(s) value.")
    if len(d_local) == 0:
        return []

    at_or_below_threshold = d_local <= threshold
    start_indexes, end_indexes = region_indexes(at_or_below_threshold)

    return [
        (float(score_edges[start]), float(score_edges[end + 1]))
        for start, end in zip(start_indexes, end_indexes, strict=True)
    ]


def d_local_above_threshold_regions(
    score_edges: numpy.ndarray,
    d_local: numpy.ndarray,
    threshold: float,
    mated_scores: numpy.ndarray,
) -> list[tuple[float, float, int]]:
    """Return above-threshold D(s) score ranges and mated sample counts."""
    if score_edges.ndim != 1 or d_local.ndim != 1:
        raise ValueError("D(s) score edges and values must be one-dimensional.")
    if len(score_edges) != len(d_local) + 1:
        raise ValueError("D(s) score edges must bound each D(s) value.")
    if len(d_local) == 0:
        return []

    mated_counts, _ = numpy.histogram(mated_scores, bins=score_edges)
    above_threshold = d_local > threshold
    start_indexes, end_indexes = region_indexes(above_threshold)

    return [
        (
            float(score_edges[start]),
            float(score_edges[end + 1]),
            int(mated_counts[start : end + 1].sum()),
        )
        for start, end in zip(start_indexes, end_indexes, strict=True)
    ]
