# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import os

from pathlib import Path

import pandas

from click.testing import CliRunner

from btpbench.scripts.diversity.summary import build_summary, summary


def _write_result(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pandas.DataFrame(
        {
            "subject_id": ["1", "2", "AVERAGE"],
            "fmr_0.01_clique_size": [2, 4, 999],
            "fmr_0.01_clique_keys": ["[1,2]", "[1,2,3,4]", ""],
            "fmr_0.001_clique_size": [3, 5, 999],
            "fmr_0.001_clique_keys": ["[1,2,3]", "[1,2,3,4,5]", ""],
        }
    ).to_csv(path, index=False)


def test_build_summary_for_all_supported_algorithm_families(tmp_path):
    results_dir = tmp_path / "results" / "soteria"
    stale_result = (
        results_dir / "diversity-soteria-normalized_polyprotect_usr_3_5_50-"
        "edgeface-keysm1d0-104.csv"
    )
    _write_result(stale_result)
    os.utime(stale_result, (1, 1))

    filenames = [
        (
            "diversity-soteria-normalized_biohash_usr_binary_103-"
            "edgeface-keysrandom-111.csv"
        ),
        "diversity-soteria-normalized_polyprotect_usr_3_5_50-edgeface-keysm1d0-102.csv",
        (
            "diversity-soteria-normalized_combined_5_"
            "normalized_polyprotect_usr_3_7_1000-"
            "edgeface-keysm1d0-102.csv"
        ),
    ]
    for filename in filenames:
        _write_result(results_dir / filename)

    report = build_summary((tmp_path,), (0.01, 0.001))

    assert len(report) == 6
    assert set(report["algorithm_family"]) == {
        "biohash",
        "polyprotect",
        "combined_polyprotect",
    }
    assert set(report["average_diversity"]) == {3.0, 4.0}
    assert set(report["min_diversity"]) == {2, 3}
    assert set(report["max_diversity"]) == {4, 5}
    assert set(report["subjects"]) == {2}
    assert 104 not in set(report["key_count"])

    biohash = report.loc[report["algorithm_family"] == "biohash"].iloc[0]
    assert biohash["biohash_bits"] == 103
    assert biohash["key_bucket"] == "random"

    combined = report.loc[report["algorithm_family"] == "combined_polyprotect"].iloc[0]
    assert combined["combined_instances"] == 5
    assert combined["coefficients"] == 7
    assert combined["coefficient_range"] == 1000
    assert combined["key_bucket"] == "-1.0"


def test_summary_cli_writes_one_focused_csv_per_family(tmp_path):
    results_dir = tmp_path / "results" / "soteria"
    filenames = (
        (
            "diversity-soteria-normalized_biohash_usr_binary_103-"
            "edgeface-keysrandom-111.csv"
        ),
        "diversity-soteria-normalized_polyprotect_usr_3_5_50-edgeface-keysm1d0-102.csv",
        (
            "diversity-soteria-normalized_combined_5_"
            "normalized_polyprotect_usr_3_7_1000-"
            "edgeface-keysm1d0-102.csv"
        ),
    )
    for filename in filenames:
        _write_result(results_dir / filename)
    output_prefix = tmp_path / "summary.csv"

    result = CliRunner().invoke(
        summary,
        ["-i", str(tmp_path), "-o", str(output_prefix)],
    )

    assert result.exit_code == 0, result.output
    assert not output_prefix.exists()

    biohash = pandas.read_csv(tmp_path / "summary-biohash.csv")
    polyprotect = pandas.read_csv(tmp_path / "summary-polyprotect.csv")
    combined = pandas.read_csv(tmp_path / "summary-combined-polyprotect.csv")
    assert list(biohash["fmr"]) == [0.01, 0.001]
    assert list(polyprotect["fmr"]) == [0.01, 0.001]
    assert list(combined["fmr"]) == [0.01, 0.001]
    assert "biohash_bits" in biohash
    assert "combined_instances" not in biohash
    assert "coefficients" in polyprotect
    assert "biohash_bits" not in polyprotect
    assert "combined_instances" in combined
