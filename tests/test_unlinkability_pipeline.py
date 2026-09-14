# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import json

from pathlib import Path

import click
import numpy
import pytest

from btpbench.btps import ProtectedTemplate
from btpbench.sample import Sample
from btpbench.scripts.unlinkability import pipeline as unlink_pipeline


def _sample(subject_id: str, idx: int) -> Sample:
    return Sample(
        subject_id,
        f"{subject_id}_{idx}",
        Path("unused"),
        {"frame_idx": idx},
        lambda: numpy.zeros((1, 1, 3)),
    )


def _protected(subject_id: str, idx: int) -> ProtectedTemplate:
    template = ProtectedTemplate(
        subject_id,
        f"{subject_id}_{idx}",
        numpy.asarray([idx]),
        idx,
    )
    template.metadata = {"frame_idx": idx}
    return template


def test_select_samples_per_subject_preserves_order_and_limits():
    samples = [
        _sample("1", 0),
        _sample("1", 1),
        _sample("1", 2),
        _sample("2", 0),
        _sample("2", 1),
        _sample("2", 2),
    ]

    selected = unlink_pipeline._select_samples_per_subject(samples, 2)

    assert list(selected) == ["1", "2"]
    assert [sample.template_id for sample in selected["1"]] == ["1_0", "1_1"]
    assert [sample.template_id for sample in selected["2"]] == ["2_0", "2_1"]


def test_select_samples_per_subject_rejects_short_subjects():
    samples = [_sample("1", 0), _sample("2", 0), _sample("2", 1)]

    with pytest.raises(ValueError, match="subject 1 only has 1"):
        unlink_pipeline._select_samples_per_subject(samples, 2)


def test_mated_and_non_mated_pair_generation():
    grouped = {
        "1": [_protected("1", 0), _protected("1", 1), _protected("1", 2)],
        "2": [_protected("2", 0), _protected("2", 1)],
    }

    _, subject_indices = unlink_pipeline._flatten_grouped_templates(grouped)

    assert list(unlink_pipeline._mated_pairs(subject_indices)) == [
        (0, 1),
        (0, 2),
        (1, 2),
        (3, 4),
    ]
    assert list(unlink_pipeline._non_mated_pairs(subject_indices)) == [
        (0, 3),
        (0, 4),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
    ]


def test_load_bucket_keys_accepts_dict_and_list(tmp_path):
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"keys": {"dict_bucket": {"a": 10, "b": 11}, "list_bucket": [20]}})
    )

    assert unlink_pipeline._load_bucket_keys(keys_file, "dict_bucket") == [10, 11]
    assert unlink_pipeline._load_bucket_keys(keys_file, "list_bucket") == [20]


def test_load_bucket_keys_rejects_missing_bucket(tmp_path):
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"keys": {}}))

    with pytest.raises(click.ClickException, match="missing"):
        unlink_pipeline._load_bucket_keys(keys_file, "missing")


def test_build_key_systems_preserves_single_key_systems():
    rng = numpy.random.default_rng(0)

    assert unlink_pipeline._build_key_systems([10, 11, 12], 2, 1, rng) == [10, 11]


def test_build_key_systems_samples_multiple_keys_per_system():
    rng = numpy.random.default_rng(0)

    key_systems = unlink_pipeline._build_key_systems([10, 11, 12], 4, 2, rng)

    assert len(key_systems) == 4
    assert all(len(key_system) == 2 for key_system in key_systems)
    assert all(len(set(key_system)) == 2 for key_system in key_systems)
    assert all(key in {10, 11, 12} for key_system in key_systems for key in key_system)


def test_build_key_systems_rejects_too_few_single_keys():
    rng = numpy.random.default_rng(0)

    with pytest.raises(ValueError, match="3 key system"):
        unlink_pipeline._build_key_systems([10, 11], 3, 1, rng)
