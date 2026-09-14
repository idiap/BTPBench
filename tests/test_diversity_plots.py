# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

from pathlib import Path

import pandas
import pytest

from click.testing import CliRunner

from btpbench.scripts.diversity.plots import (
    _criterion_label,
    load_diversity_distributions,
    plots,
)


def _write_result(path: Path) -> None:
    pandas.DataFrame(
        {
            "subject_id": ["1", "2", "3", "AVERAGE"],
            "fmr_0.01_clique_size": [2, 4, 3, 999],
            "fmr_0.01_clique_keys": ["[]", "[]", "[]", ""],
            "fmr_0.001_clique_size": [3, 3, 3, 999],
            "score_range_-2_0_clique_size": [1, 2, 4, 999],
        }
    ).to_csv(path, index=False)


def test_criterion_labels_cover_fmr_and_unlinkability_ranges():
    assert _criterion_label("fmr_0.01_clique_size") == "FMR 1%"
    assert _criterion_label("score_range_-2_0_clique_size") == "Score range [-2, 0]"
    assert (
        _criterion_label("score_ranges_-2_-1__0_1_clique_size")
        == "Score ranges [-2, -1] or [0, 1]"
    )


def test_load_diversity_distributions_excludes_average_row(tmp_path):
    result_file = tmp_path / "diversity.csv"
    _write_result(result_file)

    labels, distributions = load_diversity_distributions(result_file)

    assert labels == ["FMR 1%", "FMR 0.1%", "Score range [-2, 0]"]
    assert [values.tolist() for values in distributions] == [
        [2.0, 4.0, 3.0],
        [3.0, 3.0, 3.0],
        [1.0, 2.0, 4.0],
    ]


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("fmr_0.01_practical_N", [8.0, 10.0]),
        ("fmr_0.01_mean", [8.25, 10.5]),
        ("fmr_0.01_N", [8.5, 10.25]),
    ],
)
def test_load_diversity_distributions_supports_legacy_schemas(
    tmp_path, column, expected
):
    result_file = tmp_path / "legacy-diversity.csv"
    pandas.DataFrame(
        {
            "subject_id": ["1", "2", "AVERAGE"],
            column: [*expected, 999],
        }
    ).to_csv(result_file, index=False)

    labels, distributions = load_diversity_distributions(result_file)

    assert labels == ["FMR 1%"]
    assert distributions[0].tolist() == expected


@pytest.mark.parametrize("suffix", ["png", "pdf"])
def test_plots_cli_writes_violin_plot(tmp_path, suffix):
    result_file = tmp_path / "diversity.csv"
    output_file = tmp_path / f"violin.{suffix}"
    _write_result(result_file)

    result = CliRunner().invoke(
        plots,
        ["-f", str(result_file), "-o", str(output_file)],
    )

    assert result.exit_code == 0, result.output
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_plots_cli_rejects_csv_without_diversity_columns(tmp_path):
    result_file = tmp_path / "not-diversity.csv"
    pandas.DataFrame({"subject_id": ["1"], "value": [2]}).to_csv(
        result_file, index=False
    )

    result = CliRunner().invoke(
        plots,
        ["-f", str(result_file), "-o", str(tmp_path / "plot.png")],
    )

    assert result.exit_code != 0
    assert "No supported diversity columns" in result.output
