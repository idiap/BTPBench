# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

from pathlib import Path

import pandas

from click.testing import CliRunner

from btpbench.scripts.diversity.unlinkability_summary import (
    build_summary,
    unlinkability_summary,
)


def _write_result(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pandas.DataFrame(
        {
            "subject_id": ["1", "2", "AVERAGE"],
            "score_range_-2_0_clique_size": [2, 4, 999],
            "score_range_-2_0_clique_keys": ["[1,2]", "[1,2,3,4]", ""],
        }
    ).to_csv(path, index=False)


def _result_filenames() -> tuple[str, ...]:
    return (
        (
            "diversity-soteria-unlinkability-normalized_biohash_usr_binary_103-"
            "edgeface-keysrandom-60-threshold0d5.csv"
        ),
        (
            "diversity-soteria-unlinkability-normalized_polyprotect_usr_3_5_50-"
            "edgeface-keysm1d0-60-threshold0d5.csv"
        ),
        (
            "diversity-soteria-unlinkability-normalized_combined_5_"
            "normalized_polyprotect_usr_3_7_1000-"
            "edgeface-keysm1d0-60-threshold0d5.csv"
        ),
    )


def test_build_unlinkability_summary_for_all_families(tmp_path):
    results_dir = tmp_path / "results" / "soteria"
    for filename in _result_filenames():
        _write_result(results_dir / filename)

    report = build_summary((tmp_path,))

    assert len(report) == 3
    assert set(report["algorithm_family"]) == {
        "biohash",
        "polyprotect",
        "combined_polyprotect",
    }
    assert "density_epsilon" not in report
    assert set(report["d_local_threshold"]) == {0.5}
    assert set(report["score_criterion"]) == {"score_range_-2_0"}
    assert set(report["average_diversity"]) == {3.0}
    assert set(report["min_diversity"]) == {2}
    assert set(report["max_diversity"]) == {4}
    assert set(report["subjects"]) == {2}


def test_build_unlinkability_summary_ignores_legacy_epsilon_results(tmp_path):
    legacy_file = (
        tmp_path / "diversity-soteria-unlinkability-normalized_biohash_usr_binary_103-"
        "edgeface-keysrandom-60-epsilon0d05-threshold0d5.csv"
    )
    _write_result(legacy_file)

    report = build_summary((tmp_path,))

    assert report.empty


def test_unlinkability_summary_cli_writes_one_csv_per_family(tmp_path):
    results_dir = tmp_path / "results" / "soteria"
    for filename in _result_filenames():
        _write_result(results_dir / filename)
    output_prefix = tmp_path / "unlinkability-summary.csv"

    result = CliRunner().invoke(
        unlinkability_summary,
        ["-i", str(tmp_path), "-o", str(output_prefix)],
    )

    assert result.exit_code == 0, result.output
    assert not output_prefix.exists()

    biohash = pandas.read_csv(tmp_path / "unlinkability-summary-biohash.csv")
    polyprotect = pandas.read_csv(tmp_path / "unlinkability-summary-polyprotect.csv")
    combined = pandas.read_csv(
        tmp_path / "unlinkability-summary-combined-polyprotect.csv"
    )
    assert len(biohash) == 1
    assert len(polyprotect) == 1
    assert len(combined) == 1
    assert "biohash_bits" in biohash
    assert "combined_instances" not in biohash
    assert "coefficients" in polyprotect
    assert "biohash_bits" not in polyprotect
    assert "combined_instances" in combined
