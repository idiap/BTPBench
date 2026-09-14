# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import logging

from pathlib import Path

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas

import btpbench.metrics

from btpbench.scripts import pipeline_utils

matplotlib.use("Agg")

pipeline_utils.setup_logger()

logger = logging.getLogger(__name__)

DEFAULT_LARGE_FILE_THRESHOLD_MB = 1024
DEFAULT_CHUNK_SIZE = 500_000


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
    default="DIR curves",
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
@click.option(
    "-n",
    "--nb-resampling",
    "nb_resampling",
    type=int,
    default=20,
    help=("Specify the number of resampling."),
)
@click.option(
    "-d",
    "--decimation",
    "decimation",
    type=float,
    default=0.1,
    help=("Specify the decimation percentage."),
)
@click.option(
    "--large-file-threshold-mb",
    type=click.IntRange(min=0),
    default=DEFAULT_LARGE_FILE_THRESHOLD_MB,
    show_default=True,
    help=("Stream files at least this large instead of loading them into memory."),
)
@click.option(
    "--chunk-size",
    type=click.IntRange(min=1),
    default=DEFAULT_CHUNK_SIZE,
    show_default=True,
    help=("Number of CSV rows per chunk in large-file streaming mode."),
)
@click.option(
    "--rank1-cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=("Optional directory for reusable large-file rank-1 summaries."),
)
def main(
    score_files: list[Path],
    labels: list[str],
    title: str,
    output_file: Path,
    nb_resampling: int,
    decimation: float,
    large_file_threshold_mb: int,
    chunk_size: int,
    rank1_cache_dir: Path | None,
) -> None:
    """Generate custom DIR plots for identification results."""

    logger.info("Starting DIR plot generation with %d score file(s).", len(score_files))

    if len(labels) != len(score_files):
        logger.warning(
            "Inconsistencies between labels and score files, generating labels automatically."
        )
        labels = [None] * len(score_files)

    cols_dtypes = {
        "probe_subject_id": "str",
        "probe_template_id": "category",
        "bio_ref_subject_id": "str",
        "score": "float32",
    }
    cols_of_interest = list(cols_dtypes.keys())

    results: list[tuple[str, dict]] = list()

    for score_file, label in zip(score_files, labels):
        # If label is not specified, create a custom one.
        if label is None:
            protocol = "-".join(score_file.stem.split("-")[1:3])
            baseline = "-".join(score_file.stem.split("-")[3:])

            dataset = score_file.parent.stem
            label = f"{dataset}_{protocol}_{baseline}"

        logger.info("  Computing metrics for decimation=%.1f%%", decimation * 100)
        try:
            dec_cmc_scoress: (
                list[list[tuple[list[float], list[float]]]]
                | btpbench.metrics.Rank1ScoreSamples
            )
            threshold_bytes = large_file_threshold_mb * 1024 * 1024
            use_streaming = score_file.stat().st_size >= threshold_bytes
            if use_streaming:
                logger.info(
                    "Streaming large score file in chunks of %d rows: %s (label=%s)",
                    chunk_size,
                    score_file,
                    label,
                )
                fta, n_before, dec_cmc_scoress = (
                    btpbench.metrics.streamed_decimated_subject_neg_pos_scores(
                        score_file,
                        nb_resampling,
                        decimation,
                        chunk_size=chunk_size,
                        cache_dir=rank1_cache_dir,
                    )
                )
                n_removed = round(fta * n_before)
            else:
                logger.info("Loading score file: %s (label=%s)", score_file, label)
                score_df = pandas.read_csv(
                    score_file, usecols=cols_of_interest, dtype=cols_dtypes
                )
                n_before = len(score_df)
                _, score_df = btpbench.metrics.remove_nans(score_df)
                n_removed = n_before - len(score_df)
                dec_cmc_scoress = btpbench.metrics.decimated_subject_neg_pos_scores(
                    score_df, nb_resampling, decimation
                )

            if n_removed > 0:
                logger.warning(
                    "Removed %d NaN rows from %s.", n_removed, score_file.name
                )

            fpirs_tpirss_threshss = btpbench.metrics.fpirs_tpirss_threshss(
                dec_cmc_scoress
            )

            results.append((label, fpirs_tpirss_threshss))
        except RuntimeError as e:
            logger.error("Failed for decimation=%.1f%%: %s", decimation * 100, e)

    logger.info("Generating plot for decimation=%.1f%%.", decimation * 100)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.suptitle(title, fontsize=16)

    # ax.set_title(f"Decimation: {decimation * 100:.1f}%", fontsize=11, fontweight="bold")
    ax.set_ylabel("TPIR (%)")
    ax.set_xlabel("FPIR (%)")

    ax.grid(True, which="both", alpha=0.3)
    ax.axvline(1, color="black", linestyle="--", linewidth=0.8, label="FPIR=1%")
    ax.axhline(95, color="red", linestyle="--", linewidth=0.8, label="TPIR=95%")
    ax.set_ylim([0.0, 101])
    ax.set_xlim([0.1, 15])
    ax.set_xscale("log")

    for label, fpirs_tpirss_threshss in results:
        tpirs = numpy.mean(fpirs_tpirss_threshss[1], axis=1)
        ax.plot(fpirs_tpirss_threshss[0] * 100, tpirs * 100, label=label, linewidth=1.5)

    ax.legend(fontsize=9)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info("Plot saved to %s", output_file)


if __name__ == "__main__":
    main()
