# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import logging
import random

from pathlib import Path

import numpy
import pytest
import yaml

from btpbench.baselines import Template
from btpbench.btps import ProtectedTemplate
from btpbench.scripts import pipeline_utils
from btpbench.utils import Distribution


@pytest.fixture
def system_config_file():
    return Path(__file__).parent / "assets" / "test_config" / "system_config.yaml"


@pytest.fixture
def exp_config_file():
    return Path(__file__).parent / "assets" / "test_config" / "experiment_config.yaml"


def test_utils(system_config_file, exp_config_file):
    system_conf, exp_conf = pipeline_utils.load_config(
        system_config_file, exp_config_file
    )

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    pipeline_utils.setup_logger(logging.INFO)

    assert "baselines" in system_conf
    assert isinstance(system_conf, dict)
    assert isinstance(exp_conf, dict)
    assert "database" in exp_conf
    assert exp_conf["database"] == "test"
    assert exp_conf["splits"] == "dev"

    assert exp_conf["detector"] == "mediapipe"
    pipeline_utils.validate_detector(exp_conf["detector"])

    pipeline_utils.check_dir(output_dir, exp_conf)
    assert (output_dir / "exp_info.yaml").exists()
    with (output_dir / "exp_info.yaml").open("r") as f:
        saved_exp_conf = yaml.safe_load(f)
    assert saved_exp_conf == exp_conf

    baseline, has_protection, save, compliant, detector, n_worker = (
        pipeline_utils.load_common_parameters(system_conf, exp_conf)
    )

    assert baseline == exp_conf["bio_alg"]
    assert has_protection == ("btps" in exp_conf)
    assert save == exp_conf["save"]
    assert compliant == exp_conf["compliant"]
    assert detector == exp_conf["detector"]
    assert n_worker == exp_conf["num_processes"]

    _ = pipeline_utils.create_dataset(system_conf, exp_conf)

    (output_dir / "exp_info.yaml").unlink()
    output_dir.rmdir()


def test_get_n_keys_per_system():
    assert pipeline_utils.get_n_keys_per_system({"type": "polyprotect"}) == 1
    assert (
        pipeline_utils.get_n_keys_per_system(
            {"type": "combined", "nb_algs": 5},
        )
        == 5
    )


def test_inversion_evaluation_parameters_support_legacy_name():
    assert pipeline_utils.inversion_evaluation_parameters({}) == (10, 42)
    assert pipeline_utils.inversion_evaluation_parameters(
        {"n_validations": 4, "attack_seed": None}
    ) == (4, None)
    assert pipeline_utils.inversion_evaluation_parameters(
        {"n_validations": 4, "n_attack_trials": 7, "attack_seed": "9"}
    ) == (7, 9)

    with pytest.raises(ValueError, match="at least 1"):
        pipeline_utils.inversion_evaluation_parameters({"n_attack_trials": 0})

    assert pipeline_utils.key_sampling_seed({}) == 42
    assert pipeline_utils.key_sampling_seed({"key_sampling_seed": "7"}) == 7
    assert pipeline_utils.key_sampling_seed({"key_sampling_seed": None}) is None


def test_load_key_pool_and_sample_user_assignments(tmp_path):
    keys_file = tmp_path / "keys.json"
    keys_file.write_text('{"keys": {"-0.9": {"a": 11, "b": 12, "c": 13}}}')

    key_pool, source, tag = pipeline_utils.load_key_pool(
        {
            "sampling_mode": "keys",
            "keys_file": str(keys_file),
            "keys_bucket": "-0.9",
        }
    )
    assignments = pipeline_utils.sample_user_key_dictionaries(
        ["alice", "bob"],
        key_pool,
        n_key_systems=2,
        n_keys_per_subject=1,
        rng=random.Random(7),
    )

    assert source == "bucket:-0.9"
    assert tag == "keysm0d9"
    assert len(assignments) == 2
    assert all(set(assignment) == {"alice", "bob"} for assignment in assignments)
    assert all(len(set(assignment.values())) == 2 for assignment in assignments)
    assert pipeline_utils.key_bucket_tag("-0.9") == "m0d9"


def test_load_key_pool_rejects_missing_bucket(tmp_path):
    keys_file = tmp_path / "keys.json"
    keys_file.write_text('{"keys": {}}')

    with pytest.raises(ValueError, match="does not exist"):
        pipeline_utils.load_key_pool(
            {
                "sampling_mode": "keys",
                "keys_file": str(keys_file),
                "keys_bucket": "missing",
            }
        )


@pytest.mark.parametrize(
    ("total_items", "n_workers", "min_chunks_per_worker", "expected"),
    [
        (0, 4, 4, 1),
        (4, 4, 4, 1),
        (8, 2, 4, 1),
        (100, 4, 4, 6),
        (100, 4, 5, 5),
    ],
)
def test_get_optimal_chunksize(total_items, n_workers, min_chunks_per_worker, expected):
    assert (
        pipeline_utils.get_optimal_chunksize(
            total_items,
            n_workers,
            min_chunks_per_worker,
        )
        == expected
    )


def test_iter_protocol_splits_resolves_all_per_protocol():
    class DatasetWithDifferentSplits:
        def protocols(self):
            return ["first", "second"]

        def protocol_splits(self, protocol):
            return {
                "first": ["dev"],
                "second": ["eval"],
            }[protocol]

    selected = list(
        pipeline_utils.iter_protocol_splits(
            DatasetWithDifferentSplits(),
            protocols="all",
            splits="all",
            database_name="test",
        )
    )

    assert selected == [("first", "dev"), ("second", "eval")]


def test_protected_verification_matching_splits_work_between_processes(monkeypatch):
    observed_chunksizes = []

    class FakeExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, pairs, chunksize):
            list(pairs)
            observed_chunksizes.append(chunksize)
            return []

    monkeypatch.setattr(pipeline_utils, "ProcessPoolExecutor", FakeExecutor)

    pipeline_utils.processed_protected_verification_matching(
        prot_templates=[object()] * 10,
        btp_alg_cls=object,
        btp_alg_args=(),
        compliant=True,
        n_worker=2,
        batch_size=20,
        protected_csv=object(),
    )

    assert observed_chunksizes == [5, 1]


def test_evaluate_inversion_trials_records_distinct_trials(monkeypatch):
    original = Template("subject", "original", numpy.array([1.0]), {"camera": "a"})
    protected = ProtectedTemplate("subject", "protected", numpy.array([2.0]), 17)
    observed_calls = []

    def fake_invert(*args):
        observed_calls.append((args[-2], args[-1]))
        trial = args[-1]
        return [(Template("subject", f"inverted-{trial}", numpy.array([1.0])), 0)]

    class Writer:
        def __init__(self):
            self.rows = []

        def write_score(self, score, reference, probe):
            self.rows.append((score, reference, probe))

    writer = Writer()
    monkeypatch.setattr(pipeline_utils, "processed_inversion_no_mem", fake_invert)

    pipeline_utils.evaluate_inversion_trials(
        [original],
        [protected],
        object,
        (),
        True,
        2,
        Distribution(
            means=numpy.array([1.0]),
            mins=numpy.array([1.0]),
            maxs=numpy.array([1.0]),
            probs=[(numpy.array([1.0]), numpy.array([1.0]))],
        ),
        1,
        3,
        1234,
        lambda reference, probe: float(
            reference.get_template()[0] * probe.get_template()[0]
        ),
        writer,
        "bucket:-0.9",
        "system",
    )

    assert observed_calls == [(1234, 0), (1234, 1), (1234, 2)]
    assert [row[0] for row in writer.rows] == [1.0, 1.0, 1.0]
    assert [row[1].metadata["attack_trial"] for row in writer.rows] == [0, 1, 2]
    assert all(row[1].metadata["attack_solved"] for row in writer.rows)
    assert all(row[1].metadata["key"] == 17 for row in writer.rows)
    assert all(row[1].metadata["key_source"] == "bucket:-0.9" for row in writer.rows)
    assert all(row[1].metadata["key_scope"] == "system" for row in writer.rows)


def test_evaluate_inversion_trials_rejects_invalid_inputs():
    distribution = Distribution(
        means=numpy.array([1.0]),
        mins=numpy.array([1.0]),
        maxs=numpy.array([1.0]),
        probs=[(numpy.array([1.0]), numpy.array([1.0]))],
    )

    with pytest.raises(ValueError, match="at least 1"):
        pipeline_utils.evaluate_inversion_trials(
            [],
            [],
            object,
            (),
            True,
            1,
            distribution,
            1,
            0,
            None,
            lambda a, b: 0.0,
            object(),
            "configured",
            "user",
        )

    with pytest.raises(ValueError, match="counts must match"):
        pipeline_utils.evaluate_inversion_trials(
            [Template("s", "t", numpy.array([1.0]))],
            [],
            object,
            (),
            True,
            1,
            distribution,
            1,
            1,
            None,
            lambda a, b: 0.0,
            object(),
            "configured",
            "user",
        )
