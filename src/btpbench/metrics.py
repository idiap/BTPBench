# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import hashlib
import logging
import math
import os
import random

from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import bob.measure
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy
import pandas

from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

logger = logging.getLogger(__name__)


def remove_nans(score_df: pandas.DataFrame) -> tuple[float, pandas.DataFrame]:
    """Remove nans and return FtA."""
    fta = score_df["score"].isna().sum() / score_df.shape[0]
    score_df = score_df[score_df["score"].notna()]
    return fta, score_df


def neg_pos_scores(
    score_df: pandas.DataFrame,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return the negative and positive scores sorted in descending order."""

    # Get positive and negative scores
    pos_df = score_df[score_df["probe_subject_id"] == score_df["bio_ref_subject_id"]]
    neg_df = score_df.drop(pos_df.index)

    # Return scores sorted descending
    neg = neg_df["score"].to_numpy()
    pos = pos_df["score"].to_numpy()
    neg[::-1].sort()
    pos[::-1].sort()

    return neg, pos


def subject_neg_pos_scores(
    score_df: pandas.DataFrame,
) -> list[tuple[list[float], list[float]]]:
    """Return the negative and positive scores for each comparison sorted in descending order."""

    scores = []
    for _, comparison_score_df in score_df.groupby("probe_template_id", observed=True):
        neg, pos = neg_pos_scores(comparison_score_df)

        if neg is None:
            neg = []

        if pos is None:
            pos = []

        scores.append((neg, pos))

    return scores


def __decimate_ids(args):
    score_df, ref_subject_ids, n_decimated_sample = args
    # Arbitrary remove subject IDs from reference in order to create
    # open set identification
    banish_ids = random.sample(ref_subject_ids, n_decimated_sample)
    return subject_neg_pos_scores(
        score_df[~score_df["bio_ref_subject_id"].isin(banish_ids)]
    )


def decimated_subject_neg_pos_scores(
    score_df: pandas.DataFrame,
    n_resamples: int = 50,
    decimation_percentage: float = 0.1,
) -> list[list[tuple[list[float], list[float]]]]:
    """Return the negative and positive scores for each comparison sorted in descending order."""

    probe_subject_ids = score_df["probe_subject_id"].unique().tolist()
    ref_subject_ids = score_df["bio_ref_subject_id"].unique().tolist()

    actual_decimation = 1 - len(ref_subject_ids) / len(probe_subject_ids)

    remaining_decimation = 0.0
    if actual_decimation < decimation_percentage:
        remaining_decimation = decimation_percentage - actual_decimation

    n_decimated_sample = int(len(ref_subject_ids) * remaining_decimation)

    if n_decimated_sample == 0:
        raise RuntimeError("No sample to decimate.")

    # Fix the seed
    random.seed(42)

    args = [(score_df, ref_subject_ids, n_decimated_sample) for _ in range(n_resamples)]
    with Pool(processes=8) as pool:
        return list(pool.imap_unordered(__decimate_ids, args))


@dataclass(frozen=True)
class Rank1ScoreSamples:
    """Array-backed rank-1 CMC sufficient statistics.

    The negative array has shape ``(n_resamples, n_probes)``, the positive
    array has shape ``(n_probes,)``, and the gallery mask has shape
    ``(n_resamples, n_probes)``. Repeated system-key blocks are reduced into
    these maxima, matching the grouping performed by
    :func:`subject_neg_pos_scores` without retaining every score.
    """

    negative_maxima: numpy.ndarray
    positive_maxima: numpy.ndarray
    in_gallery: numpy.ndarray

    @property
    def n_resamples(self) -> int:
        """Return the number of gallery resamples."""
        return int(self.negative_maxima.shape[0])


def _iter_reference_subject_blocks(
    score_file: Path, chunk_size: int
) -> Iterator[tuple[str, pandas.DataFrame]]:
    """Yield consecutive reference-subject blocks from a reference-major CSV."""
    dtypes = {
        "probe_subject_id": "str",
        "bio_ref_subject_id": "str",
        "score": "float32",
    }
    pending_parts: list[pandas.DataFrame] = []
    pending_subject: str | None = None

    for chunk in pandas.read_csv(
        score_file,
        usecols=list(dtypes),
        dtype=dtypes,
        chunksize=chunk_size,
    ):
        reference_subjects = chunk["bio_ref_subject_id"].to_numpy(copy=False)
        boundaries = numpy.flatnonzero(
            reference_subjects[1:] != reference_subjects[:-1]
        )
        starts = numpy.concatenate(([0], boundaries + 1))
        stops = numpy.concatenate((boundaries + 1, [len(chunk)]))

        for start, stop in zip(starts, stops, strict=True):
            subject = str(reference_subjects[start])
            part = chunk.iloc[int(start) : int(stop)]
            if pending_subject is not None and subject != pending_subject:
                yield pending_subject, pandas.concat(pending_parts, ignore_index=True)
                pending_parts = []
            pending_subject = subject
            pending_parts.append(part)

    if pending_subject is not None:
        yield pending_subject, pandas.concat(pending_parts, ignore_index=True)


def streamed_decimated_subject_neg_pos_scores(
    score_file: Path,
    n_resamples: int = 50,
    decimation_percentage: float = 0.1,
    precision: int = -1,
    chunk_size: int = 500_000,
    cache_dir: Path | None = None,
) -> tuple[float, int, Rank1ScoreSamples]:
    """Build fast, memory-efficient rank-1 statistics from a score CSV.

    This is equivalent to :func:`decimated_subject_neg_pos_scores` for rank-1
    metrics, but never holds the complete score table or complete per-probe
    score vectors in memory. It exploits the reference-major ordering emitted
    by the identification matcher and makes one streaming pass over the file.
    NumPy updates replace per-resample pandas group-bys and Python dictionaries.

    A maximum is sufficient because rank-1 TPIR depends only on whether the
    best genuine score exceeds both the threshold and every impostor score;
    open-set FPIR depends only on the best impostor score.

    Returns
    -------
    tuple
        Failure-to-acquire rate, total input row count, and compressed CMC
        scores for every resample.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1.")
    if not 0 < decimation_percentage <= 1:
        raise ValueError("decimation_percentage must be in the interval (0, 1].")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    score_file = Path(score_file)
    cache_path = None
    if cache_dir is not None:
        source_stat = score_file.stat()
        cache_key = "\0".join(
            (
                "rank1-v1",
                str(score_file.resolve()),
                str(source_stat.st_size),
                str(source_stat.st_mtime_ns),
                str(n_resamples),
                repr(decimation_percentage),
                str(precision),
            )
        )
        digest = hashlib.sha256(cache_key.encode()).hexdigest()[:24]
        cache_path = Path(cache_dir) / f"rank1-{digest}.npz"
        if cache_path.is_file():
            try:
                with numpy.load(cache_path, allow_pickle=False) as cached:
                    cached_scores = Rank1ScoreSamples(
                        cached["negative_maxima"],
                        cached["positive_maxima"],
                        cached["in_gallery"],
                    )
                    logger.info("Loaded rank-1 score cache: %s", cache_path)
                    return (
                        float(cached["fta"]),
                        int(cached["total_rows"]),
                        cached_scores,
                    )
            except (KeyError, OSError, ValueError):
                logger.warning("Ignoring unreadable rank-1 score cache: %s", cache_path)

    total_rows = 0
    nan_rows = 0
    probe_subject_ids: list[str] | None = None
    banished_subjects: list[set[str]] | None = None
    gallery_mask: numpy.ndarray | None = None
    first_reference_subject: str | None = None
    seen_reference_subjects: set[str] = set()
    all_reference_subjects: set[str] = set()

    key_probe_subjects: numpy.ndarray | None = None
    key_positive_maxima: numpy.ndarray | None = None
    key_negative_maxima: numpy.ndarray | None = None
    negative_maxima: numpy.ndarray | None = None
    positive_maxima: numpy.ndarray | None = None
    key_block_count = 0

    def finish_key() -> None:
        nonlocal key_block_count, negative_maxima, positive_maxima
        if key_positive_maxima is None or key_negative_maxima is None:
            return
        if positive_maxima is None:
            positive_maxima = key_positive_maxima
            negative_maxima = key_negative_maxima
        else:
            assert negative_maxima is not None
            numpy.maximum(positive_maxima, key_positive_maxima, out=positive_maxima)
            numpy.maximum(negative_maxima, key_negative_maxima, out=negative_maxima)
        key_block_count += 1
        logger.info(
            "Processed system-key block %d (%d score rows).",
            key_block_count,
            total_rows,
        )

    for reference_subject, block in _iter_reference_subject_blocks(
        score_file, chunk_size
    ):
        if first_reference_subject is None:
            first_reference_subject = reference_subject
        elif (
            reference_subject == first_reference_subject
            and reference_subject in seen_reference_subjects
        ):
            finish_key()
            key_probe_subjects = None
            key_positive_maxima = None
            key_negative_maxima = None
            seen_reference_subjects.clear()

        seen_reference_subjects.add(reference_subject)
        all_reference_subjects.add(reference_subject)
        scores = block["score"].to_numpy(dtype=numpy.float32, copy=True)
        total_rows += len(scores)
        nan_mask = numpy.isnan(scores)
        nan_rows += int(nan_mask.sum())
        scores[nan_mask] = -numpy.inf
        if precision > 0:
            numpy.round(scores, precision, out=scores)

        block_probe_subjects = block["probe_subject_id"].to_numpy(copy=False)
        if key_probe_subjects is None:
            key_probe_subjects = block_probe_subjects.copy()
            if probe_subject_ids is None:
                probe_subject_ids = list(dict.fromkeys(key_probe_subjects.tolist()))
                n_decimated_sample = int(len(probe_subject_ids) * decimation_percentage)
                if n_decimated_sample == 0:
                    raise RuntimeError("No sample to decimate.")
                if n_decimated_sample >= len(probe_subject_ids):
                    raise RuntimeError(
                        "Decimation must leave at least one reference subject."
                    )
                rng = random.Random(42)
                banished_subjects = [
                    set(rng.sample(probe_subject_ids, n_decimated_sample))
                    for _ in range(n_resamples)
                ]
                gallery_mask = numpy.vstack(
                    [
                        ~numpy.isin(key_probe_subjects, list(banished))
                        for banished in banished_subjects
                    ]
                )
            elif len(key_probe_subjects) != gallery_mask.shape[1]:  # type: ignore[union-attr]
                raise RuntimeError(
                    "Probe count changed between key blocks; the large-file "
                    "optimizer requires reference-major identification scores."
                )

            key_positive_maxima = numpy.full(
                len(key_probe_subjects), -numpy.inf, dtype=numpy.float32
            )
            key_negative_maxima = numpy.full(
                (n_resamples, len(key_probe_subjects)),
                -numpy.inf,
                dtype=numpy.float32,
            )
        elif len(block_probe_subjects) != len(key_probe_subjects):
            raise RuntimeError(
                "Reference blocks have different probe counts; the large-file "
                "optimizer requires reference-major identification scores."
            )

        positive_mask = key_probe_subjects == reference_subject
        key_positive_maxima[positive_mask] = numpy.maximum(  # type: ignore[index]
            key_positive_maxima[positive_mask],  # type: ignore[index]
            scores[positive_mask],
        )
        scores[positive_mask] = -numpy.inf

        for sample_index, banished in enumerate(banished_subjects):  # type: ignore[arg-type]
            if reference_subject not in banished:
                numpy.maximum(
                    key_negative_maxima[sample_index],  # type: ignore[index]
                    scores,
                    out=key_negative_maxima[sample_index],  # type: ignore[index]
                )

    if total_rows == 0:
        raise RuntimeError("The score file is empty.")
    finish_key()

    if set(probe_subject_ids) != all_reference_subjects:  # type: ignore[arg-type]
        raise RuntimeError(
            "The optimized one-pass decimation requires identical probe and "
            "reference subject populations."
        )

    assert positive_maxima is not None
    assert negative_maxima is not None
    assert gallery_mask is not None

    # A missing/NaN genuine score behaves like an out-of-gallery probe after
    # remove_nans(), so exclude it from the in-gallery masks.
    gallery_mask &= numpy.isfinite(positive_maxima)[None, :]

    result = (
        nan_rows / total_rows,
        total_rows,
        Rank1ScoreSamples(negative_maxima, positive_maxima, gallery_mask),
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache = cache_path.with_name(
            f"{cache_path.name}.{os.getpid()}.temporary"
        )
        with temporary_cache.open("wb") as cache_file:
            numpy.savez(
                cache_file,
                fta=result[0],
                total_rows=result[1],
                negative_maxima=result[2].negative_maxima,
                positive_maxima=result[2].positive_maxima,
                in_gallery=result[2].in_gallery,
            )
        temporary_cache.replace(cache_path)
        logger.info("Saved rank-1 score cache: %s", cache_path)
    return result


def _rank1_open_set_maxima(
    scores: Rank1ScoreSamples, sample_index: int
) -> numpy.ndarray:
    """Return finite open-set impostor maxima for one resampling."""
    maxima = scores.negative_maxima[sample_index, ~scores.in_gallery[sample_index]]
    return maxima[numpy.isfinite(maxima)]


def _rank1_detection_identification_rate(
    scores: Rank1ScoreSamples, sample_index: int, threshold: float
) -> float:
    """Compute rank-1 TPIR from sufficient statistics for one resampling."""
    closed_mask = scores.in_gallery[sample_index]
    positives = scores.positive_maxima[closed_mask]
    negatives = scores.negative_maxima[sample_index, closed_mask]
    if not positives.size:
        return numpy.nan
    return float(numpy.mean((positives >= threshold) & (positives > negatives)))


def fpir_r1_thresh(
    scores: list[list[tuple[list[float], list[float]]]] | Rank1ScoreSamples,
    thresh: float = -0.5,
) -> tuple[float, float]:
    """Compute distribution of FPIR at rank 1 for a given threshold."""

    fpirs: list[float] = []

    if isinstance(scores, Rank1ScoreSamples):
        for sample_index in range(scores.n_resamples):
            negatives = _rank1_open_set_maxima(scores, sample_index)
            if not negatives.size:
                raise ValueError(
                    "There need to be at least one pair with only negative scores"
                )
            fpirs.append(float(numpy.mean(negatives >= thresh)))
        return numpy.mean(fpirs), numpy.std(fpirs)

    for cmc_score in scores:
        fpirs.append(bob.measure.false_alarm_rate(cmc_score, thresh))
    return numpy.mean(fpirs), numpy.std(fpirs)


def tpir_r1_thresh(
    scores: list[list[tuple[list[float], list[float]]]] | Rank1ScoreSamples,
    thresh: float = -0.5,
) -> tuple[float, float]:
    """Compute distribution of TPIR at rank 1 for a given threshold."""

    tpirs: list[float] = []

    if isinstance(scores, Rank1ScoreSamples):
        tpirs = [
            _rank1_detection_identification_rate(scores, i, thresh)
            for i in range(scores.n_resamples)
        ]
        return numpy.mean(tpirs), numpy.std(tpirs)

    for cmc_score in scores:
        tpirs.append(bob.measure.detection_identification_rate(cmc_score, thresh, 1))
    return numpy.mean(tpirs), numpy.std(tpirs)


def __tpirs_threshs(args):
    cmc_scores, fpirs = args

    # for each probe, for which no positives exists, get the highest negative
    # score; and sort them to compute the FAR thresholds
    negatives = sorted(
        max(neg)
        for neg, pos in cmc_scores
        if (pos is None or not numpy.array(pos).size) and neg is not None
    )
    if not negatives:
        raise ValueError("There need to be at least one pair with only negative scores")

    # compute thresholds based on FAR values
    thresholds = [bob.measure.far_threshold(negatives, [], v, True) for v in fpirs]

    # compute detection and identification rate based on the thresholds for
    # the given rank
    tpirs = [
        bob.measure.detection_identification_rate(cmc_scores, t, 1)
        if not math.isnan(t)
        else numpy.nan
        for t in thresholds
    ]

    return tpirs, thresholds


def fpirs_tpirss_threshss(
    scores: list[list[tuple[list[float], list[float]]]] | Rank1ScoreSamples,
):
    """For a given input of thresholds return a list of fpir and tpir values."""

    fpirs = numpy.array([math.pow(10.0, i * 1.0 / 4) for i in range(-4 * 4, 0)] + [1.0])

    if isinstance(scores, Rank1ScoreSamples):
        tpirss = numpy.zeros((len(fpirs), scores.n_resamples))
        thresholdss = numpy.zeros((len(fpirs), scores.n_resamples))
        for sample_index in range(scores.n_resamples):
            negatives = _rank1_open_set_maxima(scores, sample_index)
            if not negatives.size:
                raise ValueError(
                    "There need to be at least one pair with only negative scores"
                )
            negatives.sort()
            thresholds = numpy.asarray(
                [
                    bob.measure.far_threshold(negatives, [], value, True)
                    for value in fpirs
                ]
            )
            thresholdss[:, sample_index] = thresholds
            tpirss[:, sample_index] = [
                _rank1_detection_identification_rate(scores, sample_index, threshold)
                if not math.isnan(threshold)
                else numpy.nan
                for threshold in thresholds
            ]
        return fpirs, tpirss, thresholdss

    tpirss = numpy.zeros((len(fpirs), len(scores)))
    thresholdss = numpy.zeros((len(fpirs), len(scores)))

    args = [(cmc_scores, fpirs) for cmc_scores in scores]
    with Pool(processes=8) as pool:
        tpirss_thresholdss = list(pool.imap_unordered(__tpirs_threshs, args))

    for i, tpirs_thresholds in enumerate(tpirss_thresholdss):
        tpirss[:, i] = tpirs_thresholds[0]
        thresholdss[:, i] = tpirs_thresholds[1]

    return fpirs, tpirss, thresholdss


def plot_dir_decimated(
    name: str, values: dict[float, tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]
):
    """Generate a DIR plot.

    Args:
        name: Evaluation protocol's name.
        values: Dictionary with keys representing the decimation percentage and
            values representing the FPIR, TPIR, and threshold values.
    """
    plt.figure()
    plt.grid()

    styles = ["dashed", "solid", "dotted"]
    i = 0

    for dec, val in values.items():
        tpirs = numpy.mean(val[1], axis=1)

        plt.plot(
            val[0],
            tpirs,
            label=f"Decimation {dec * 100:.2f}%",
            color="blue",
            linestyle=styles[i % 3],
        )
        i += 1

    plt.title(f"DIR Curve for {name}")
    plt.xlabel("False Positive Identification Rate (%)")
    plt.ylabel("True Positive Identification Rate (%)")
    plt.vlines([0.01], 0, 1, colors="black", linestyles="dashed", label="FPIR=1%")
    plt.hlines([0.95], 0.0001, 1, colors="red", linestyles="dashed", label="TPIR=95%")
    plt.ylim([0, 1.01])
    plt.xlim([0.0001, 1])
    plt.legend()
    plt.xscale("log")


def __plot_confidence_ellipse(
    x: numpy.ndarray, y: numpy.ndarray, ax: Axes, n_std: float = 1.0, **kwargs
):
    """Plot a confidence ellipse for a 2D distribution.

    This function is internal and shouldn't be called outside.
    """

    cov = numpy.cov(x, y)
    pearson = cov[0, 1] / numpy.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = numpy.sqrt(1 + pearson)
    ell_radius_y = numpy.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor="none",
        **kwargs,
    )

    scale_x = numpy.sqrt(cov[0, 0]) * n_std
    mean_x = numpy.mean(x)

    scale_y = numpy.sqrt(cov[1, 1]) * n_std
    mean_y = numpy.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_dist_decimated(
    name: str, values: dict[float, tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]
):
    """Generate a 2D distribution plot.

    Args:
        name: Evaluation protocol's name.
        values: Dictionary with keys representing the decimation percentage and
            values representing the FPIR, TPIR, and threshold values.
    """
    plt.figure()
    ax = plt.gca()
    plt.grid()
    plt.title(f"TPIR@FPIR=1% distribution for {name}")
    plt.axis("equal")
    plt.xlabel("Threshold")
    plt.ylabel("True Positive Identification Rate (%)")
    colors = ["red", "green", "blue"]
    i = 0
    for dec, val in values.items():
        fpirs, tpirss, threshss = val
        fpir_1per_idx = numpy.abs(numpy.array(fpirs) - 0.01).argmin()
        tpirs = tpirss[fpir_1per_idx, :]
        threshs = threshss[fpir_1per_idx, :]

        for thresh, tpir in zip(threshs, tpirs):
            plt.scatter(thresh, tpir, color=colors[i % 3], alpha=0.2, marker="x")

        plt.scatter(
            numpy.mean(threshs),
            numpy.mean(tpirs),
            color=colors[i % 3],
            marker="x",
            label=f"Decimation {int(dec * 100)}%",
        )
        __plot_confidence_ellipse(threshs, tpirs, ax, edgecolor=colors[i % 3])

        i += 1
    plt.legend()


def setup_det_plot():
    """Set the axis limit of a det curve to be used with bob."""

    desired_ticks = [
        "0.000001",
        "0.000002",
        "0.000005",
        "0.00001",
        "0.00002",
        "0.00005",
        "0.0001",
        "0.0002",
        "0.0005",
        "0.001",
        "0.002",
        "0.005",
        "0.01",
        "0.02",
        "0.05",
        "0.1",
        "0.2",
        "0.4",
        "0.6",
        "0.8",
        "0.9",
        "0.95",
        "0.98",
        "0.99",
        "0.995",
        "0.998",
        "0.999",
        "0.9995",
        "0.9998",
        "0.9999",
        "0.99995",
        "0.99998",
        "0.99999",
    ]

    desired_labels = [
        "0.0001",
        "0.0002",
        "0.0005",
        "0.001",
        "0.002",
        "0.005",
        "0.01",
        "0.02",
        "0.05",
        "0.1",
        "0.2",
        "0.5",
        "1",
        "2",
        "5",
        "10",
        "20",
        "40",
        "60",
        "80",
        "90",
        "95",
        "98",
        "99",
        "99.5",
        "99.8",
        "99.9",
        "99.95",
        "99.98",
        "99.99",
        "99.995",
        "99.998",
        "99.999",
    ]

    # now the trick: we must plot the tick marks by hand using the PPNDF method
    pticks = bob.measure.ppndf(numpy.array(desired_ticks, dtype=float))
    ax = plt.gca()  # and finally we set our own tick marks
    ax.set_xticks(pticks)
    ax.set_xticklabels(desired_labels)
    ax.set_yticks(pticks)
    ax.set_yticklabels(desired_labels)

    plt.xlabel("FMR (%)")
    plt.ylabel("FNMR (%)")
    plt.grid(True)
    bob.measure.plot.det_axis([0.01, 95, 0.01, 95])
    plt.tick_params(axis="x", rotation=50)
