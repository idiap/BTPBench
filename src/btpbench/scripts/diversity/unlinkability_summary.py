# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

"""Summarize diversity results derived from unlinkability score ranges."""

import re

from pathlib import Path
from typing import Any

import click
import pandas

from btpbench.scripts.diversity.summary import (
    _algorithm_fields,
    _decode_key_bucket,
    _subject_rows,
    build_family_report_columns,
    write_family_summary_files,
)

_RESULT_NAME_RE = re.compile(
    r"^diversity-(?P<database>[^-]+)-unlinkability"
    r"(?:-subjects(?P<subject_limit>\d+))?"
    r"-(?P<algorithm>.+)-(?P<baseline>[^-]+)"
    r"-keys(?P<key_bucket>[^-]+)-(?P<key_count>\d+)"
    r"-threshold(?P<d_local_threshold>[^-]+)$"
)

_OUTPUT_COLUMNS = [
    "algorithm_family",
    "algorithm",
    "database",
    "baseline",
    "normalization",
    "key_scope",
    "key_bucket",
    "key_count",
    "d_local_threshold",
    "score_criterion",
    "biohash_bits",
    "biohash_representation",
    "combined_instances",
    "overlap",
    "coefficients",
    "coefficient_range",
    "subjects",
    "average_diversity",
    "min_diversity",
    "max_diversity",
    "source_file",
]
_COMMON_REPORT_COLUMNS = [
    "algorithm",
    "database",
    "baseline",
    "normalization",
    "key_scope",
    "key_bucket",
    "key_count",
    "d_local_threshold",
    "score_criterion",
    "subjects",
    "average_diversity",
    "min_diversity",
    "max_diversity",
    "source_file",
]
_FAMILY_REPORT_COLUMNS = build_family_report_columns(_COMMON_REPORT_COLUMNS)


def _decode_number_tag(tag: str) -> float:
    """Decode a filename-safe number such as ``0d05`` or ``m0d5``."""
    return float(tag.replace("m", "-", 1).replace("d", "."))


def _result_metadata(result_file: Path) -> dict[str, Any] | None:
    """Parse unlinkability-diversity metadata encoded in a result filename."""
    match = _RESULT_NAME_RE.fullmatch(result_file.stem)
    if not match:
        return None

    values = match.groupdict()
    algorithm_fields = _algorithm_fields(values["algorithm"])
    if algorithm_fields is None:
        return None

    return {
        **algorithm_fields,
        "algorithm": values["algorithm"],
        "database": values["database"],
        "baseline": values["baseline"],
        "key_bucket": _decode_key_bucket(values["key_bucket"]),
        "key_count": int(values["key_count"]),
        "d_local_threshold": _decode_number_tag(values["d_local_threshold"]),
        "source_file": str(result_file.resolve()),
    }


def summarize_result_file(result_file: Path) -> list[dict[str, Any]]:
    """Build summary rows for every score-range criterion in one CSV."""
    metadata = _result_metadata(result_file)
    if metadata is None:
        return []

    results = pandas.read_csv(result_file)
    subjects = _subject_rows(results)
    size_columns = sorted(
        column
        for column in subjects.columns
        if column.startswith(("score_range_", "score_ranges_"))
        and column.endswith("_clique_size")
    )

    rows = []
    for size_column in size_columns:
        clique_sizes = pandas.to_numeric(
            subjects[size_column], errors="coerce"
        ).dropna()
        if clique_sizes.empty:
            continue

        rows.append(
            {
                **metadata,
                "score_criterion": size_column.removesuffix("_clique_size"),
                "subjects": len(clique_sizes),
                "average_diversity": clique_sizes.mean(),
                "min_diversity": int(clique_sizes.min()),
                "max_diversity": int(clique_sizes.max()),
            }
        )
    return rows


def _find_result_files(input_dirs: tuple[Path, ...]) -> list[Path]:
    """Find the newest CSV for each unlinkability-derived configuration."""
    result_files: dict[Path, Path] = {}
    for input_dir in input_dirs:
        patterns = (
            "diversity-*-unlinkability-*-threshold*.csv",
            "*/diversity-*-unlinkability-*-threshold*.csv",
            "results/*/diversity-*-unlinkability-*-threshold*.csv",
        )
        for pattern in patterns:
            for result_file in input_dir.glob(pattern):
                resolved = result_file.resolve()
                result_files[resolved] = resolved

    latest_files: dict[tuple[str, str, str, str, float], Path] = {}
    for result_file in result_files.values():
        metadata = _result_metadata(result_file)
        if metadata is None:
            continue
        identity = (
            metadata["database"],
            metadata["algorithm"],
            metadata["baseline"],
            metadata["key_bucket"],
            metadata["d_local_threshold"],
        )
        previous = latest_files.get(identity)
        if (
            previous is None
            or result_file.stat().st_mtime_ns > previous.stat().st_mtime_ns
        ):
            latest_files[identity] = result_file
    return sorted(latest_files.values())


def build_summary(input_dirs: tuple[Path, ...]) -> pandas.DataFrame:
    """Summarize supported unlinkability-derived diversity result files."""
    rows = [
        row
        for result_file in _find_result_files(input_dirs)
        for row in summarize_result_file(result_file)
    ]
    if not rows:
        return pandas.DataFrame(columns=_OUTPUT_COLUMNS)

    summary = pandas.DataFrame(rows)
    for column in _OUTPUT_COLUMNS:
        if column not in summary:
            summary[column] = pandas.NA

    integer_columns = (
        "key_count",
        "biohash_bits",
        "combined_instances",
        "overlap",
        "coefficients",
        "coefficient_range",
        "subjects",
        "min_diversity",
        "max_diversity",
    )
    for column in integer_columns:
        summary[column] = summary[column].astype("Int64")

    family_order = {"biohash": 0, "polyprotect": 1, "combined_polyprotect": 2}
    summary["_family_order"] = summary["algorithm_family"].map(family_order)
    summary = summary.sort_values(
        [
            "_family_order",
            "algorithm",
            "key_bucket",
            "d_local_threshold",
            "score_criterion",
        ]
    )
    return summary[_OUTPUT_COLUMNS].reset_index(drop=True)


@click.command(name="unlinkability-summary")
@click.option(
    "-i",
    "--input-dir",
    "input_dirs",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Experiment folder or results folder. Repeat for separate runs.",
)
@click.option(
    "-o",
    "--output-prefix",
    "--output-file",
    "output_prefix",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="CSV filename prefix used to create one output per algorithm family.",
)
def unlinkability_summary(input_dirs: tuple[Path, ...], output_prefix: Path):
    """Report diversity statistics derived from unlinkability score ranges."""
    report = build_summary(input_dirs)
    if report.empty:
        raise click.ClickException(
            "No matching BioHash, PolyProtect, or combined PolyProtect "
            "unlinkability-diversity results were found."
        )

    output_files = write_family_summary_files(
        report,
        output_prefix,
        _FAMILY_REPORT_COLUMNS,
    )
    for output_file in output_files:
        click.echo(f"Wrote {output_file}")


if __name__ == "__main__":
    unlinkability_summary()
