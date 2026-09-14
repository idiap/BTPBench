# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import json

import numpy
import pandas
import pytest

from click.testing import CliRunner

from btpbench.scripts.unlinkability import metrics, plots


def test_load_optimized_mated_scores_from_diversity_csv(tmp_path):
    score_file = tmp_path / "diversity.csv"
    pandas.DataFrame(
        [
            {
                "subject_id": "1",
                "selected_keys": json.dumps([10, 11, 12]),
                "selected_key_scores": json.dumps([0.1, None, -0.2]),
            },
            {
                "subject_id": "2",
                "selected_keys": json.dumps([12, 13]),
                "selected_key_scores": json.dumps([-0.4]),
            },
            {
                "subject_id": "AVERAGE",
                "selected_keys": "[]",
                "selected_key_scores": "[]",
            },
        ]
    ).to_csv(score_file, index=False)

    scores, n_unique_keys = plots._load_optimized_mated_scores(score_file)

    numpy.testing.assert_array_equal(scores, numpy.array([0.1, -0.2, -0.4]))
    assert n_unique_keys == 4


def test_load_optimized_mated_scores_from_clique_diversity_csv(tmp_path):
    score_file = tmp_path / "diversity.csv"
    pandas.DataFrame(
        [
            {
                "subject_id": "1",
                "score_range_clique_size": 3,
                "score_range_clique_keys": json.dumps([10, 11, 12]),
                "score_range_clique_scores": json.dumps([0.1, None, -0.2]),
            },
            {
                "subject_id": "2",
                "score_range_clique_size": 2,
                "score_range_clique_keys": json.dumps([12, 13]),
                "score_range_clique_scores": json.dumps([-0.4]),
            },
            {
                "subject_id": "AVERAGE",
                "score_range_clique_size": 2.5,
                "score_range_clique_keys": "",
                "score_range_clique_scores": "",
            },
        ]
    ).to_csv(score_file, index=False)

    scores, n_unique_keys = plots._load_optimized_mated_scores(score_file)

    numpy.testing.assert_array_equal(scores, numpy.array([0.1, -0.2, -0.4]))
    assert n_unique_keys == 4


def test_unlinkability_metric_keeps_small_positive_density_difference():
    mated_scores = numpy.concatenate([numpy.full(1001, 0.5), numpy.full(999, 1.5)])
    non_mated_scores = numpy.concatenate([numpy.full(1000, 0.5), numpy.full(1000, 1.5)])
    edges = numpy.array([0.0, 1.0, 2.0])

    _, d_local, _ = metrics.unlinkability_metric(
        mated_scores,
        non_mated_scores,
        edges,
        omega=1.0,
    )

    numpy.testing.assert_allclose(d_local, numpy.array([1 / 2001, 0.0]))


def test_compute_unlinkability_metric_needs_no_density_tolerance():
    metric = metrics.compute_unlinkability_metric(
        numpy.array([0.2, 0.3, 1.2, 1.3, 1.4]),
        numpy.array([0.2, 1.2, 1.4, 2.2, 2.4]),
        metric_bins=3,
        omega=1.0,
        x_min=0.0,
        x_max=3.0,
    )

    numpy.testing.assert_allclose(metric.d_local, numpy.array([1 / 3, 0.2, 0.0]))


@pytest.mark.parametrize(
    "epsilon_option",
    ["-e", "--epsilon", "--density-epsilon"],
)
def test_unlinkability_plots_reject_epsilon_options(epsilon_option):
    result = CliRunner().invoke(plots.main, [epsilon_option, "0"])

    assert result.exit_code == 2
    assert "No such option" in result.output
    assert epsilon_option in result.output


def test_d_local_transition_scores_mark_at_or_below_threshold_boundaries():
    scores = numpy.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3])
    d_local = numpy.array([0.3, 0.1, 0.2, 0.4, 0.2, 0.5])

    transition_scores = metrics.d_local_transition_scores(
        scores,
        d_local,
        threshold=0.2,
    )

    numpy.testing.assert_array_equal(
        transition_scores,
        numpy.array([-0.7, -0.6, -0.4]),
    )


def test_d_local_transition_scores_only_mark_above_threshold_transitions():
    scores = numpy.array([0.1, 0.2, 0.3, 0.4])
    d_local = numpy.array([0.5, 0.5, 0.1, 0.1])

    transition_scores = metrics.d_local_transition_scores(
        scores,
        d_local,
        threshold=0.5,
    )

    numpy.testing.assert_array_equal(transition_scores, numpy.array([]))


def test_d_local_above_threshold_regions_count_mated_samples():
    score_edges = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    d_local = numpy.array([0.2, 0.4, 0.6, 0.1, 1.0])
    mated_scores = numpy.array([-0.1, 0.2, 1.1, 1.9, 2.5, 4.9, 5.0])

    regions = metrics.d_local_above_threshold_regions(
        score_edges,
        d_local,
        threshold=0.2,
        mated_scores=mated_scores,
    )

    assert regions == [(1.0, 3.0, 3), (4.0, 5.0, 2)]


def test_d_local_at_or_below_threshold_ranges():
    score_edges = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    d_local = numpy.array([0.2, 0.4, 0.6, 0.1, 1.0])

    regions = metrics.d_local_at_or_below_threshold_ranges(
        score_edges,
        d_local,
        threshold=0.2,
    )

    assert regions == [(0.0, 1.0), (3.0, 4.0)]


def test_unlinkability_plots_can_write_interactive_html(tmp_path):
    pytest.importorskip("plotly")

    rng = numpy.random.default_rng(0)
    mated_file = tmp_path / "unlinkability-mated.csv"
    non_mated_file = tmp_path / "unlinkability-non-mated.csv"
    optimized_mated_file = tmp_path / "diversity.csv"
    pandas.DataFrame({"score": rng.normal(-1.0, 0.15, 1000)}).to_csv(
        mated_file,
        index=False,
    )
    pandas.DataFrame({"score": rng.normal(-0.8, 0.2, 1000)}).to_csv(
        non_mated_file,
        index=False,
    )
    pandas.DataFrame(
        [
            {
                "subject_id": "1",
                "score_range_clique_size": 3,
                "score_range_clique_keys": json.dumps([10, 11, 12]),
                "score_range_clique_scores": json.dumps([-1.2, -1.1, -1.0]),
            },
            {
                "subject_id": "2",
                "score_range_clique_size": 2,
                "score_range_clique_keys": json.dumps([12, 13]),
                "score_range_clique_scores": json.dumps([-0.9]),
            },
        ]
    ).to_csv(optimized_mated_file, index=False)
    out_file = tmp_path / "unlinkability.html"

    result = CliRunner().invoke(
        plots.main,
        [
            "-m",
            str(mated_file),
            "-n",
            str(non_mated_file),
            "--optimized-mated-score-file",
            str(optimized_mated_file),
            "-t",
            "Unlinkability",
            "-o",
            str(out_file),
            "--bins",
            "64",
            "--smooth-sigma",
            "1.0",
        ],
    )

    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert out_file.exists()
    html = out_file.read_text(encoding="utf-8").lower()
    assert "<html" in html
    assert "mated (selected, 4 keys)" in html
