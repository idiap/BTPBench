# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import csv
import io
import json

import click
import numpy
import pandas
import pytest

from click.testing import CliRunner

from btpbench.baselines import Template
from btpbench.btps import ProtectedTemplate
from btpbench.scripts.diversity import pipeline as diversity_pipeline


def _template(subject_id: str, idx: int) -> Template:
    return Template(subject_id, f"{subject_id}_{idx}", numpy.asarray([idx]))


class _DistanceAlgorithm:
    def compare(self, left: ProtectedTemplate, right: ProtectedTemplate) -> float:
        return float(abs(left.get_template()[0] - right.get_template()[0]))


def _protected_templates(subject_id: str, key_offset: int) -> list[ProtectedTemplate]:
    return [
        ProtectedTemplate(
            subject_id,
            f"{subject_id}_{value}",
            numpy.asarray([value]),
            key_offset + value,
        )
        for value in (1, 2, 4)
    ]


def test_unlinkability_protection_batches_match_templates_to_keys():
    templates = [
        _template("1", 0),
        _template("1", 1),
        _template("1", 2),
        _template("2", 0),
        _template("2", 1),
    ]

    templates_by_subject, keys, batches = (
        diversity_pipeline._unlinkability_protection_batches(
            templates,
            [10, 11, 12, 13],
        )
    )

    assert list(templates_by_subject) == ["1", "2"]
    assert keys == [10, 11, 12]
    assert [(key, [t.template_id for t in batch]) for key, batch in batches] == [
        (10, ["1_0", "2_0"]),
        (11, ["1_1", "2_1"]),
        (12, ["1_2"]),
    ]


def test_unlinkability_protection_batches_skip_missing_templates():
    templates = [
        Template("1", "1_missing", None),
        _template("1", 0),
        _template("2", 0),
        Template("2", "2_missing", None),
    ]

    templates_by_subject, keys, batches = (
        diversity_pipeline._unlinkability_protection_batches(templates, [10, 11])
    )

    assert [t.template_id for t in templates_by_subject["1"]] == ["1_0"]
    assert [t.template_id for t in templates_by_subject["2"]] == ["2_0"]
    assert keys == [10]
    assert [(key, [t.template_id for t in batch]) for key, batch in batches] == [
        (10, ["1_0", "2_0"]),
    ]


def test_unlinkability_protection_batches_trim_templates_to_available_keys():
    templates = [
        _template("1", 0),
        _template("1", 1),
        _template("2", 0),
        _template("2", 1),
        _template("2", 2),
    ]

    templates_by_subject, keys, batches = (
        diversity_pipeline._unlinkability_protection_batches(templates, [10, 11])
    )

    assert [t.template_id for t in templates_by_subject["1"]] == ["1_0", "1_1"]
    assert [t.template_id for t in templates_by_subject["2"]] == ["2_0", "2_1"]
    assert keys == [10, 11]
    assert [(key, [t.template_id for t in batch]) for key, batch in batches] == [
        (10, ["1_0", "2_0"]),
        (11, ["1_1", "2_1"]),
    ]


def test_unlinkability_protection_batches_require_keys():
    templates = [_template("1", 0), _template("1", 1)]

    with pytest.raises(ValueError, match="at least one key"):
        diversity_pipeline._unlinkability_protection_batches(templates, [])


def test_select_first_subjects_preserves_all_items_for_selected_subjects():
    templates = [
        _template("1", 0),
        _template("2", 0),
        _template("1", 1),
        _template("3", 0),
    ]

    selected = diversity_pipeline._select_first_subjects(templates, 2)

    assert [template.template_id for template in selected] == ["1_0", "2_0", "1_1"]


def test_select_first_subjects_minus_one_keeps_every_subject():
    templates = [_template("1", 0), _template("2", 0)]

    assert diversity_pipeline._select_first_subjects(templates, -1) == templates


def test_select_first_subjects_rejects_zero():
    with pytest.raises(ValueError, match="-1 or greater"):
        diversity_pipeline._select_first_subjects([_template("1", 0)], 0)


def test_build_key_systems_preserves_single_key_systems():
    rng = numpy.random.default_rng(0)

    assert diversity_pipeline._build_key_systems([10, 11, 12], 2, 1, rng) == [10, 11]


def test_build_key_systems_samples_multiple_keys_per_system():
    rng = numpy.random.default_rng(0)

    key_systems = diversity_pipeline._build_key_systems([10, 11, 12], 4, 2, rng)

    assert len(key_systems) == 4
    assert all(len(key_system) == 2 for key_system in key_systems)
    assert all(len(set(key_system)) == 2 for key_system in key_systems)
    assert all(key in {10, 11, 12} for key_system in key_systems for key in key_system)


def test_build_key_systems_rejects_too_small_combined_bucket():
    rng = numpy.random.default_rng(0)

    with pytest.raises(ValueError, match="needs 3 key"):
        diversity_pipeline._build_key_systems([10, 11], 4, 3, rng)


def test_score_range_thresholds_qualify_inclusive_ranges():
    [threshold] = diversity_pipeline._score_range_thresholds(((-0.9, -0.7),))
    scores = numpy.asarray([-1.0, -0.9, -0.8, -0.7, -0.6])

    assert threshold.tag == "score_range_-0.9_-0.7"
    numpy.testing.assert_array_equal(
        threshold.qualifies(scores),
        numpy.asarray([False, True, True, True, False]),
    )


def test_score_range_thresholds_qualify_any_range():
    [threshold] = diversity_pipeline._score_range_thresholds(
        ((-1.0, -0.9), (-0.7, -0.6)),
    )
    scores = numpy.asarray([-1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5])

    assert threshold.tag == "score_ranges_-1_-0.9__-0.7_-0.6"
    numpy.testing.assert_array_equal(
        threshold.qualifies(scores),
        numpy.asarray([False, True, True, False, True, True, False]),
    )


def test_score_range_thresholds_reject_inverted_ranges():
    with pytest.raises(click.ClickException, match="MIN"):
        diversity_pipeline._score_range_thresholds(((-0.7, -0.9),))


def test_unlinkability_metric_filename_tag_includes_threshold():
    assert (
        diversity_pipeline._unlinkability_metric_filename_tag(
            True,
            d_local_threshold=0.5,
        )
        == "-threshold0d5"
    )


def test_unlinkability_metric_filename_tag_is_empty_for_other_diversity_runs():
    assert (
        diversity_pipeline._unlinkability_metric_filename_tag(
            False,
            d_local_threshold=0.5,
        )
        == ""
    )


@pytest.mark.parametrize(
    "epsilon_option",
    ["--epsilon", "--density-epsilon"],
)
def test_diversity_pipeline_rejects_epsilon_options(epsilon_option):
    result = CliRunner().invoke(
        diversity_pipeline.pipeline,
        [epsilon_option, "0"],
    )

    assert result.exit_code == 2
    assert "No such option" in result.output
    assert epsilon_option in result.output


def test_unlinkability_score_range_thresholds_derive_d_local_ranges(tmp_path):
    mated_file = tmp_path / "unlinkability-mated.csv"
    non_mated_file = tmp_path / "unlinkability-non-mated.csv"
    pandas.DataFrame({"score": [0.2, 0.3, 1.2, 1.3, 1.4]}).to_csv(
        mated_file,
        index=False,
    )
    pandas.DataFrame({"score": [0.2, 1.2, 1.4, 2.2, 2.4]}).to_csv(
        non_mated_file,
        index=False,
    )

    [threshold] = diversity_pipeline._unlinkability_score_range_thresholds(
        mated_file,
        non_mated_file,
        metric_bins=3,
        omega=1.0,
        d_local_threshold=0.0,
        x_min=0.0,
        x_max=3.0,
    )

    assert threshold.tag == "score_range_2_3"
    numpy.testing.assert_array_equal(
        threshold.qualifies(numpy.asarray([1.99, 2.0, 2.5, 3.0, 3.01])),
        numpy.asarray([False, True, True, True, False]),
    )


def test_unlinkability_score_range_thresholds_require_score_file_pair(tmp_path):
    mated_file = tmp_path / "unlinkability-mated.csv"
    pandas.DataFrame({"score": [0.2]}).to_csv(mated_file, index=False)

    with pytest.raises(click.ClickException, match="must be passed together"):
        diversity_pipeline._unlinkability_score_range_thresholds(
            mated_file,
            None,
            metric_bins=3,
            omega=1.0,
            d_local_threshold=0.0,
            x_min=0.0,
            x_max=3.0,
        )


def test_fmr_thresholds_require_protected_score_file():
    with pytest.raises(click.ClickException, match="protected-score-file"):
        diversity_pipeline._fmr_thresholds(None, (0.01,))


def test_format_clique_keys_is_single_csv_safe_json_cell():
    clique_keys = diversity_pipeline._json_cell([numpy.int64(10), 11, "12"])

    assert clique_keys == '[10,11,"12"]'
    assert json.loads(clique_keys) == [10, 11, "12"]

    buffer = io.StringIO()
    pandas.DataFrame(
        [
            {
                "subject_id": "1",
                "fmr_0.01_clique_size": 3,
                "fmr_0.01_clique_keys": clique_keys,
            }
        ]
    ).to_csv(buffer, index=False)

    buffer.seek(0)
    row = next(csv.DictReader(buffer))

    assert row["fmr_0.01_clique_keys"] == clique_keys
    assert len(row) == 3


def test_append_average_row_averages_sizes_and_leaves_keys_blank():
    threshold = diversity_pipeline._DiversityThreshold(
        tag="test",
        label="test threshold",
        max_score=0.5,
    )
    subject_results = pandas.DataFrame(
        [
            {
                "subject_id": "1",
                "test_clique_size": 2,
                "test_clique_keys": "[10,11]",
                "test_clique_scores": "[-0.5]",
            },
            {
                "subject_id": "2",
                "test_clique_size": 3,
                "test_clique_keys": "[20,21,22]",
                "test_clique_scores": "[-0.4,-0.3,-0.2]",
            },
        ]
    )

    results = diversity_pipeline._append_average_row(subject_results, [threshold])

    assert results.iloc[-1].to_dict() == {
        "subject_id": "AVERAGE",
        "test_clique_size": 2.5,
        "test_clique_keys": "",
        "test_clique_scores": "",
    }

    buffer = io.StringIO()
    results.to_csv(buffer, index=False)
    rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))
    assert rows[0]["test_clique_size"] == "2"
    assert rows[-1]["test_clique_size"] == "2.5"
    assert rows[-1]["test_clique_keys"] == ""
    assert rows[-1]["test_clique_scores"] == ""


def test_maximum_clique_returns_exact_largest_complete_subgraph():
    # Upper-triangle order for five nodes. Nodes 0, 2, and 3 form the unique
    # maximum clique; the extra 0--1 edge cannot extend it.
    qualifying_pairs = numpy.asarray(
        [
            True,  # 0--1
            True,  # 0--2
            True,  # 0--3
            False,  # 0--4
            False,  # 1--2
            False,  # 1--3
            False,  # 1--4
            True,  # 2--3
            False,  # 2--4
            False,  # 3--4
        ]
    )

    assert diversity_pipeline._maximum_clique(qualifying_pairs, 5) == (0, 2, 3)


def test_maximum_clique_returns_one_of_multiple_maxima():
    qualifying_pairs = numpy.asarray([True, False, False, False, False, True])

    assert diversity_pipeline._maximum_clique(qualifying_pairs, 4) in {
        (0, 1),
        (2, 3),
    }


def test_maximum_clique_includes_a_single_template():
    assert diversity_pipeline._maximum_clique(numpy.asarray([]), 1) == (0,)


def test_subject_clique_results_parallelize_subjects_and_preserve_order():
    threshold = diversity_pipeline._DiversityThreshold(
        tag="test",
        label="test threshold",
        max_score=2.5,
    )
    subject_protected = {
        "subject-2": _protected_templates("subject-2", 10),
        "subject-1": _protected_templates("subject-1", 0),
    }

    results = diversity_pipeline._subject_clique_results(
        subject_protected,
        [threshold],
        _DistanceAlgorithm(),
        _DistanceAlgorithm,
        (),
        n_workers=2,
    )

    assert [row["subject_id"] for row in results] == ["subject-2", "subject-1"]
    assert [row["test_clique_size"] for row in results] == [2, 2]
    assert json.loads(results[0]["test_clique_keys"]) in ([11, 12], [12, 14])
    assert json.loads(results[1]["test_clique_keys"]) in ([1, 2], [2, 4])
    assert json.loads(results[0]["test_clique_scores"]) in ([1.0], [2.0])
    assert json.loads(results[1]["test_clique_scores"]) in ([1.0], [2.0])
