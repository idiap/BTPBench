# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import csv
import logging

from pathlib import Path

import click
import numpy
import pandas

import btpbench
import btpbench.metrics

from btpbench.scripts import pipeline_utils

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
    required=True,
    help=("Specify the labels to use."),
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
    "decimations",
    type=float,
    multiple=True,
    default=[0.1, 0.3, 0.5],
    help=("Specify the decimation percentage(s)."),
)
@click.option(
    "-p",
    "--precision",
    "precision",
    type=int,
    default=-1,
    help=("Specify the precision."),
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
def metrics(
    score_files: list[Path],
    labels: list[str],
    output_file: Path,
    nb_resampling: int,
    decimations: list[float],
    precision: int,
    large_file_threshold_mb: int,
    chunk_size: int,
    rank1_cache_dir: Path | None,
):
    """Entry point to analyze results."""

    assert len(score_files) == len(labels), (
        "The number of score files must match the number of labels."
    )

    cols_dtypes = {
        "probe_subject_id": "str",
        "probe_template_id": "category",
        "bio_ref_subject_id": "str",
        "score": "float32",
    }
    cols_of_interest = list(cols_dtypes.keys())
    csv_headers = [
        "label",
        "fta",
    ]

    for d in decimations:
        dec_headers = [
            f"thresh_mean_{int(d * 100)}",
            f"thresh_std_{int(d * 100)}",
            f"tpir_mean_{int(d * 100)}",
            f"tpir_std_{int(d * 100)}",
            f"fpir_mean_{int(d * 100)}",
            f"fpir_std_{int(d * 100)}",
        ]
        csv_headers += dec_headers

    csv_file = Path.open(output_file, "w", newline="")
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(csv_headers)

    for score_file, label in zip(score_files, labels):
        logger.info(f"processing `{score_file}`...")
        threshold_bytes = large_file_threshold_mb * 1024 * 1024
        use_streaming = score_file.stat().st_size >= threshold_bytes
        score_df = None
        fta = numpy.nan
        if use_streaming:
            logger.info("Streaming large score file in chunks of %d rows.", chunk_size)
        else:
            score_df = pandas.read_csv(
                score_file, usecols=cols_of_interest, dtype=cols_dtypes
            )

            if precision > 0:  # If precision is specified, round the scores
                score_df["score"] = score_df["score"].round(precision)

            fta, score_df = btpbench.metrics.remove_nans(score_df)

        decimated_fpirs_tpirss_threshss = dict()
        decimated_cmc_scoress = dict()

        for d in decimations:
            try:
                dec_cmc_scoress: (
                    list[list[tuple[list[float], list[float]]]]
                    | btpbench.metrics.Rank1ScoreSamples
                )
                if use_streaming:
                    streamed_fta, _, dec_cmc_scoress = (
                        btpbench.metrics.streamed_decimated_subject_neg_pos_scores(
                            score_file,
                            nb_resampling,
                            d,
                            precision=precision,
                            chunk_size=chunk_size,
                            cache_dir=rank1_cache_dir,
                        )
                    )
                    fta = streamed_fta
                else:
                    dec_cmc_scoress = btpbench.metrics.decimated_subject_neg_pos_scores(
                        score_df, nb_resampling, d
                    )

                fpirs_tpirss_threshss = btpbench.metrics.fpirs_tpirss_threshss(
                    dec_cmc_scoress
                )

                decimated_fpirs_tpirss_threshss[d] = fpirs_tpirss_threshss
                decimated_cmc_scoress[d] = dec_cmc_scoress
            except RuntimeError as e:
                logger.error(e)

        csv_row = [
            label,
            fta,
        ]

        for d in decimations:
            if d not in decimated_fpirs_tpirss_threshss:
                csv_row += [
                    numpy.nan,
                    numpy.nan,
                    numpy.nan,
                    numpy.nan,
                    numpy.nan,
                    numpy.nan,
                ]
                continue

            fpirs, _, threshss = decimated_fpirs_tpirss_threshss[d]
            idx = numpy.abs(numpy.array(fpirs) - 0.01).argmin()
            thresh = numpy.mean(threshss[idx, :])
            thresh_std = numpy.std(threshss[idx, :])

            mean_tpir, std_tpir = btpbench.metrics.tpir_r1_thresh(
                decimated_cmc_scoress[d], thresh
            )

            mean_fpir, std_fpir = btpbench.metrics.fpir_r1_thresh(
                decimated_cmc_scoress[d], thresh
            )

            csv_row += [
                thresh,
                thresh_std,
                mean_tpir,
                std_tpir,
                mean_fpir,
                std_fpir,
            ]

        csv_file_writer.writerow(csv_row)

    csv_file.close()


if __name__ == "__main__":
    metrics()
