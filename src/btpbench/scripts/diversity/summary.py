# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

"""Summarize maximum-clique diversity results across experiment grids."""

import re

from pathlib import Path
from typing import Any

import click
import pandas

_RESULT_NAME_RE = re.compile(
    r"^diversity-(?P<database>[^-]+)"
    r"(?:-subjects(?P<subject_limit>\d+))?"
    r"-(?P<algorithm>.+)-(?P<baseline>[^-]+)"
    r"-keys(?P<key_bucket>[^-]+)-(?P<key_count>\d+)$"
)
_BIOHASH_RE = re.compile(
    r"^(?P<normalization>normalized|unnormalized)_biohash_"
    r"(?P<key_scope>usr|sys)_(?P<representation>binary|real)_"
    r"(?P<bits>\d+)$"
)
_POLYPROTECT_RE = re.compile(
    r"^(?P<normalization>normalized|unnormalized)_polyprotect_"
    r"(?P<key_scope>usr|sys)_(?P<overlap>\d+)_"
    r"(?P<coefficients>\d+)_(?P<coefficient_range>\d+)$"
)
_COMBINED_POLYPROTECT_RE = re.compile(
    r"^(?P<normalization>normalized|unnormalized)_combined_"
    r"(?P<combined_instances>\d+)_"
    r"(?P<inner_normalization>normalized|unnormalized)_polyprotect_"
    r"(?P<key_scope>usr|sys)_(?P<overlap>\d+)_"
    r"(?P<coefficients>\d+)_(?P<coefficient_range>\d+)$"
)


def build_family_report_columns(common_columns: list[str]) -> dict[str, list[str]]:
    """Add algorithm-specific fields to a common summary column layout."""
    return {
        "biohash": [
            *common_columns[:5],
            "biohash_bits",
            "biohash_representation",
            *common_columns[5:],
        ],
        "polyprotect": [
            *common_columns[:5],
            "overlap",
            "coefficients",
            "coefficient_range",
            *common_columns[5:],
        ],
        "combined_polyprotect": [
            *common_columns[:5],
            "combined_instances",
            "overlap",
            "coefficients",
            "coefficient_range",
            *common_columns[5:],
        ],
    }


_OUTPUT_COLUMNS = [
    "algorithm_family",
    "algorithm",
    "database",
    "baseline",
    "normalization",
    "key_scope",
    "key_bucket",
    "key_count",
    "fmr",
    "fmr_percent",
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
    "fmr",
    "fmr_percent",
    "subjects",
    "average_diversity",
    "min_diversity",
    "max_diversity",
    "source_file",
]
_FAMILY_REPORT_COLUMNS = build_family_report_columns(_COMMON_REPORT_COLUMNS)


def _decode_key_bucket(tag: str) -> str:
    """Convert a filename-safe key bucket tag back to its display value."""
    if tag == "random":
        return tag
    return tag.replace("m", "-", 1).replace("d", ".")


def _algorithm_fields(algorithm: str) -> dict[str, Any] | None:
    """Extract configuration fields from a supported algorithm name."""
    match = _COMBINED_POLYPROTECT_RE.fullmatch(algorithm)
    if match:
        values = match.groupdict()
        return {
            "algorithm_family": "combined_polyprotect",
            "normalization": values["normalization"],
            "key_scope": values["key_scope"],
            "combined_instances": int(values["combined_instances"]),
            "overlap": int(values["overlap"]),
            "coefficients": int(values["coefficients"]),
            "coefficient_range": int(values["coefficient_range"]),
        }

    match = _POLYPROTECT_RE.fullmatch(algorithm)
    if match:
        values = match.groupdict()
        return {
            "algorithm_family": "polyprotect",
            "normalization": values["normalization"],
            "key_scope": values["key_scope"],
            "overlap": int(values["overlap"]),
            "coefficients": int(values["coefficients"]),
            "coefficient_range": int(values["coefficient_range"]),
        }

    match = _BIOHASH_RE.fullmatch(algorithm)
    if match:
        values = match.groupdict()
        return {
            "algorithm_family": "biohash",
            "normalization": values["normalization"],
            "key_scope": values["key_scope"],
            "biohash_bits": int(values["bits"]),
            "biohash_representation": values["representation"],
        }

    return None


def _result_metadata(result_file: Path) -> dict[str, Any] | None:
    """Parse experiment metadata encoded in a diversity result filename."""
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
        "source_file": str(result_file.resolve()),
    }


def _subject_rows(results: pandas.DataFrame) -> pandas.DataFrame:
    """Return real subject rows, excluding the pipeline's aggregate row."""
    if "subject_id" not in results:
        return results
    subject_ids = results["subject_id"].astype(str).str.upper()
    return results.loc[subject_ids != "AVERAGE"]


def summarize_result_file(
    result_file: Path,
    fmrs: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Build one summary row per requested FMR for a result CSV."""
    metadata = _result_metadata(result_file)
    if metadata is None:
        return []

    results = pandas.read_csv(result_file)
    subjects = _subject_rows(results)
    rows = []
    for fmr in fmrs:
        fmr_tag = f"{fmr:g}"
        size_column = f"fmr_{fmr_tag}_clique_size"
        if size_column not in subjects:
            continue

        clique_sizes = pandas.to_numeric(
            subjects[size_column], errors="coerce"
        ).dropna()
        if clique_sizes.empty:
            continue

        rows.append(
            {
                **metadata,
                "fmr": fmr,
                "fmr_percent": fmr * 100,
                "subjects": len(clique_sizes),
                "average_diversity": clique_sizes.mean(),
                "min_diversity": int(clique_sizes.min()),
                "max_diversity": int(clique_sizes.max()),
            }
        )
    return rows


def _find_result_files(input_dirs: tuple[Path, ...]) -> list[Path]:
    """Find the newest CSV for each algorithm/configuration/key bucket."""
    result_files: dict[Path, Path] = {}
    for input_dir in input_dirs:
        patterns = (
            "diversity-*.csv",
            "*/diversity-*.csv",
            "results/*/diversity-*.csv",
        )
        for pattern in patterns:
            for result_file in input_dir.glob(pattern):
                resolved = result_file.resolve()
                result_files[resolved] = resolved

    latest_files: dict[tuple[str, str, str, str], Path] = {}
    for result_file in result_files.values():
        metadata = _result_metadata(result_file)
        if metadata is None:
            continue
        identity = (
            metadata["database"],
            metadata["algorithm"],
            metadata["baseline"],
            metadata["key_bucket"],
        )
        previous = latest_files.get(identity)
        if (
            previous is None
            or result_file.stat().st_mtime_ns > previous.stat().st_mtime_ns
        ):
            latest_files[identity] = result_file
    return sorted(latest_files.values())


def build_summary(
    input_dirs: tuple[Path, ...], fmrs: tuple[float, ...]
) -> pandas.DataFrame:
    """Summarize all supported diversity results found below input folders."""
    rows = [
        row
        for result_file in _find_result_files(input_dirs)
        for row in summarize_result_file(result_file, fmrs)
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
            "key_count",
            "fmr",
        ],
        ascending=[True, True, True, True, False],
    )
    return summary[_OUTPUT_COLUMNS].reset_index(drop=True)


def _family_output_file(output_prefix: Path, family: str) -> Path:
    """Build one family-specific CSV path from the requested output prefix."""
    suffix = output_prefix.suffix or ".csv"
    stem = output_prefix.stem if output_prefix.suffix else output_prefix.name
    family_tag = family.replace("_", "-")
    return output_prefix.parent / f"{stem}-{family_tag}{suffix}"


def write_family_summary_files(
    report: pandas.DataFrame,
    output_prefix: Path,
    family_report_columns: dict[str, list[str]],
) -> list[Path]:
    """Write one focused CSV for each configured algorithm family."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_files = []
    for family, columns in family_report_columns.items():
        output_file = _family_output_file(output_prefix, family)
        family_report = report.loc[report["algorithm_family"] == family, columns]
        family_report.to_csv(output_file, index=False)
        output_files.append(output_file)
    return output_files


def write_summary_files(report: pandas.DataFrame, output_prefix: Path) -> list[Path]:
    """Write the standard diversity summary files."""
    return write_family_summary_files(report, output_prefix, _FAMILY_REPORT_COLUMNS)


@click.command()
@click.option(
    "-i",
    "--input-dir",
    "input_dirs",
    multiple=True,
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        path_type=Path,
    ),
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
@click.option(
    "-f",
    "--fmr",
    "fmrs",
    type=click.FloatRange(min=0.0, max=1.0, min_open=True),
    multiple=True,
    default=(0.01, 0.001),
    show_default=True,
    help="FMR fraction to report. Repeat for multiple FMRs.",
)
def summary(input_dirs: tuple[Path, ...], output_prefix: Path, fmrs: tuple[float, ...]):
    """Report mean, minimum, and maximum diversity for experiment results."""
    report = build_summary(input_dirs, fmrs)
    if report.empty:
        raise click.ClickException(
            "No matching BioHash, PolyProtect, or combined PolyProtect FMR "
            "diversity results were found."
        )

    output_files = write_summary_files(report, output_prefix)
    for output_file in output_files:
        click.echo(f"Wrote {output_file}")


if __name__ == "__main__":
    summary()
