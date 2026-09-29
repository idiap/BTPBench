# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import csv

import numpy
import pandas
import pytest

from click.testing import CliRunner

from btpbench.baselines import Template
from btpbench.scripts.keyselection import pipeline_user_cost


def _write_verification_scores(path, extra_scores=()):
    rows = [
        {"probe_subject_id": "a", "bio_ref_subject_id": "a", "score": -0.9},
        {"probe_subject_id": "a", "bio_ref_subject_id": "b", "score": -0.3},
        {"probe_subject_id": "b", "bio_ref_subject_id": "a", "score": -0.2},
        *extra_scores,
    ]
    pandas.DataFrame(rows).to_csv(path, index=False)


def test_pipeline_writes_subject_costs_and_forwards_seed(tmp_path, monkeypatch):
    system_file = tmp_path / "system.yaml"
    experiment_file = tmp_path / "experiment.yaml"
    verification_file = tmp_path / "verification.csv"
    system_file.touch()
    experiment_file.touch()
    _write_verification_scores(verification_file)

    experiment = {
        "output_dir": str(tmp_path),
        "key_sampling_seed": 17,
        "btps": {
            "algs": [
                {
                    "type": "dummy",
                    "system_specific": False,
                    "ks_method": "multiple_guesses",
                    "ks_n_elements": 5,
                }
            ]
        },
    }
    valid = Template("subject-a", "valid", numpy.array([0.25, 0.75]))
    duplicate = Template("subject-a", "duplicate", numpy.array([0.5, 0.5]))
    missing = Template("subject-b", "missing", None)

    class Dataset:
        def samples(self):
            return [valid, missing, duplicate]

    class BioAlgorithm:
        def __init__(self, *args):
            pass

        def compare(self, reference, probe):
            return 0.0

    class BTPAlgorithm:
        def get_inversion_config(self):
            return {"precision": 3}

        def get_key_selection_tag(self):
            return "selection"

        def get_inversion_config_tag(self):
            return "inversion"

        def get_alg_name(self):
            return "dummy"

        def is_system_specific(self):
            return False

    class FakeWorker:
        @classmethod
        def key_selection_usr_cost(cls, index):
            raise AssertionError("FakeExecutor supplies results directly")

    executor_calls = []

    class FakeExecutor:
        def __init__(self, **kwargs):
            executor_calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, indexes, chunksize):
            assert function == FakeWorker.key_selection_usr_cost
            assert list(indexes) == [0, 1]
            assert chunksize == 1
            return iter(
                [
                    {
                        "subject_id": "subject-a",
                        "template_id": "valid",
                        "elapsed_seconds": 1.25,
                        "n_trials": 4,
                        "status": "selected",
                    },
                    {
                        "subject_id": "subject-b",
                        "template_id": "missing",
                        "elapsed_seconds": 0.0,
                        "n_trials": 0,
                        "status": "missing_template",
                    },
                ]
            )

    monkeypatch.setattr(
        pipeline_user_cost.pipeline_utils,
        "load_config",
        lambda *args: ({}, experiment),
    )
    monkeypatch.setattr(
        pipeline_user_cost.pipeline_utils,
        "load_common_parameters",
        lambda *args: ("dummybio", True, False, True, "mediapipe", 2),
    )
    monkeypatch.setattr(
        pipeline_user_cost.pipeline_utils,
        "create_dataset",
        lambda *args: ("database", Dataset()),
    )
    monkeypatch.setattr(
        pipeline_user_cost.pipeline_utils,
        "threaded_feature_extraction",
        lambda samples, *args: list(samples),
    )
    monkeypatch.setattr(
        pipeline_user_cost.pipeline_utils,
        "build_btp_alg",
        lambda *args: (
            BTPAlgorithm(),
            BTPAlgorithm,
            (tmp_path, False, experiment["btps"]["algs"][0]),
            tmp_path,
        ),
    )
    monkeypatch.setattr(
        pipeline_user_cost.pipeline_utils, "validate_detector", lambda *_: None
    )
    monkeypatch.setattr(
        pipeline_user_cost,
        "get_baseline_dict",
        lambda: {"dummybio": BioAlgorithm},
    )
    monkeypatch.setattr(
        pipeline_user_cost,
        "get_protected_baseline_dict",
        lambda: {"dummy": BTPAlgorithm},
    )
    monkeypatch.setattr(
        pipeline_user_cost.bob.measure, "far_threshold", lambda *args: -0.25
    )
    monkeypatch.setattr(pipeline_user_cost, "BTPWorker", FakeWorker)
    monkeypatch.setattr(pipeline_user_cost, "ProcessPoolExecutor", FakeExecutor)

    result = CliRunner().invoke(
        pipeline_user_cost.pipeline,
        [
            "-s",
            str(system_file),
            "-e",
            str(experiment_file),
            "-v",
            str(verification_file),
            "-f",
            "0.05",
        ],
    )

    assert result.exit_code == 0, result.output
    output_file = (
        tmp_path
        / "key_selection_cost_fmr0.05-selection-inversion-database-dummy-dummybio.csv"
    )
    rows = list(csv.DictReader(output_file.open()))
    assert [row["row_type"] for row in rows] == ["subject", "subject", "mean"]
    assert rows[0]["elapsed_seconds"] == "1.25"
    assert rows[0]["n_trials"] == "4"
    assert rows[1]["subject_id"] == "subject-b"
    assert rows[1]["status"] == "missing_template"
    assert rows[2]["status"] == "complete"
    assert rows[2]["n_subjects"] == "2"
    assert rows[2]["n_selected"] == "1"
    assert rows[2]["n_missing_templates"] == "1"
    assert rows[2]["elapsed_seconds"] == "1.25"
    assert rows[2]["n_trials"] == "4.0"
    assert {row["key_sampling_seed"] for row in rows} == {"17"}
    init_extra_args = executor_calls[0]["initargs"][-1]
    assert init_extra_args["key_sampling_seed"] == 17
    assert [template.subject_id for template in executor_calls[0]["initargs"][3]] == [
        "subject-a",
        "subject-b",
    ]
    provenance_file = output_file.with_suffix(".csv.provenance.json")
    provenance = pipeline_user_cost.json.loads(provenance_file.read_text())
    assert provenance["key_sampling_seed"] == 17
    assert provenance["num_processes"] == 2
    assert provenance["threshold"] == -0.25
    assert provenance["verification_file_sha256"]
    assert provenance["output_sha256"]
    assert not list(tmp_path.glob("*.partial"))

    repeated = CliRunner().invoke(
        pipeline_user_cost.pipeline,
        [
            "-s",
            str(system_file),
            "-e",
            str(experiment_file),
            "-v",
            str(verification_file),
            "-f",
            "0.05",
        ],
    )
    assert repeated.exit_code == 0, repeated.output
    assert len(executor_calls) == 1

    experiment["key_sampling_seed"] = 18
    changed = CliRunner().invoke(
        pipeline_user_cost.pipeline,
        [
            "-s",
            str(system_file),
            "-e",
            str(experiment_file),
            "-v",
            str(verification_file),
            "-f",
            "0.05",
        ],
    )
    assert changed.exit_code == 0, changed.output
    assert len(executor_calls) == 2
    assert (
        pipeline_user_cost.json.loads(provenance_file.read_text())["key_sampling_seed"]
        == 18
    )


def test_verification_scores_drop_nan_and_reject_infinity(tmp_path, monkeypatch):
    scores_file = tmp_path / "scores.csv"
    _write_verification_scores(
        scores_file,
        [
            {
                "probe_subject_id": "c",
                "bio_ref_subject_id": "d",
                "score": numpy.nan,
            }
        ],
    )
    observed = {}

    def fake_threshold(negative_scores, positive_scores, fmr):
        observed["negative"] = negative_scores
        observed["positive"] = positive_scores
        observed["fmr"] = fmr
        return -0.4

    monkeypatch.setattr(pipeline_user_cost.bob.measure, "far_threshold", fake_threshold)
    assert pipeline_user_cost._thresholds_from_scores(scores_file, (0.1,)) == {
        0.1: -0.4
    }
    assert len(observed["negative"]) == 2
    assert len(observed["positive"]) == 1

    for invalid_score in (numpy.inf, -numpy.inf, "not-a-score"):
        _write_verification_scores(
            scores_file,
            [
                {
                    "probe_subject_id": "c",
                    "bio_ref_subject_id": "d",
                    "score": invalid_score,
                }
            ],
        )
        with pytest.raises(pipeline_user_cost.click.ClickException):
            pipeline_user_cost._thresholds_from_scores(scores_file, (0.1,))


def test_verification_scores_allow_no_mated_scores_and_reject_missing_ids(
    tmp_path, monkeypatch
):
    scores_file = tmp_path / "scores.csv"
    pandas.DataFrame(
        [
            {"probe_subject_id": 1, "bio_ref_subject_id": 2, "score": -0.3},
            {"probe_subject_id": 2, "bio_ref_subject_id": 1, "score": -0.2},
        ]
    ).to_csv(scores_file, index=False)
    monkeypatch.setattr(
        pipeline_user_cost.bob.measure, "far_threshold", lambda *args: -0.25
    )

    assert pipeline_user_cost._thresholds_from_scores(scores_file, (0.1,)) == {
        0.1: -0.25
    }

    scores_file.write_text(
        "probe_subject_id,bio_ref_subject_id,score\n001,1,-0.3\n1,001,-0.2\n"
    )
    assert pipeline_user_cost._thresholds_from_scores(scores_file, (0.1,)) == {
        0.1: -0.25
    }

    pandas.DataFrame(
        [
            {"probe_subject_id": "a", "bio_ref_subject_id": "b", "score": -0.3},
            {
                "probe_subject_id": numpy.nan,
                "bio_ref_subject_id": "a",
                "score": -0.2,
            },
        ]
    ).to_csv(scores_file, index=False)
    with pytest.raises(
        pipeline_user_cost.click.ClickException, match="must not contain missing"
    ):
        pipeline_user_cost._thresholds_from_scores(scores_file, (0.1,))


@pytest.mark.parametrize("fmr", ["nan", "inf", "0", "1", "-0.1"])
def test_cli_rejects_invalid_fmr(tmp_path, fmr):
    system_file = tmp_path / "system.yaml"
    experiment_file = tmp_path / "experiment.yaml"
    verification_file = tmp_path / "verification.csv"
    system_file.touch()
    experiment_file.touch()
    verification_file.touch()

    result = CliRunner().invoke(
        pipeline_user_cost.pipeline,
        [
            "-s",
            str(system_file),
            "-e",
            str(experiment_file),
            "-v",
            str(verification_file),
            "-f",
            fmr,
        ],
    )

    assert result.exit_code == 2
    assert "FMRs must be finite fractions" in result.output


@pytest.mark.parametrize(
    ("baseline", "n_worker", "configs"),
    [
        ("bio", 0, [{"type": "btp"}]),
        ("bio", True, [{"type": "btp"}]),
        ("bio", 1, None),
        ("bio", 1, {}),
        ("bio", 1, [{}]),
        ("bio", 1, [{"type": []}]),
        (["bio"], 1, [{"type": "btp"}]),
    ],
)
def test_experiment_validation_rejects_malformed_config(baseline, n_worker, configs):
    with pytest.raises(pipeline_user_cost.click.ClickException):
        pipeline_user_cost._validate_experiment(
            baseline,
            True,
            n_worker,
            configs,
            {"bio": object},
            {"btp": object},
        )


def test_provenance_prevents_reusing_stale_or_modified_results(tmp_path):
    verification_file = tmp_path / "verification.csv"
    result_file = tmp_path / "cost.csv"
    _write_verification_scores(verification_file)
    result_file.write_text("row_type,status\nmean,complete\n")
    verification_samples_file = tmp_path / "samples.csv"
    verification_samples_file.write_text("subject_id,template_id,path\n1,1,image\n")
    system_config = {
        "databases": {
            "database": {
                "dataset_dir": str(tmp_path / "dataset"),
                "proto_dir": str(tmp_path / "protocols"),
                "verification_samples": str(tmp_path / "samples.csv"),
            }
        }
    }
    exp_config = {"protocol": "verification", "key_sampling_seed": 42}
    btp_config = {"type": "dummy", "system_specific": False}
    runtime = {
        "threshold": -0.25,
        "detector": "mediapipe",
        "num_processes": 2,
    }

    def provenance():
        return pipeline_user_cost._build_provenance(
            system_config,
            exp_config,
            btp_config,
            verification_file,
            0.05,
            runtime["threshold"],
            "database",
            "baseline",
            runtime["detector"],
            runtime["num_processes"],
            exp_config["key_sampling_seed"],
        )

    original = provenance()
    pipeline_user_cost._publish_provenance(result_file, original)
    assert pipeline_user_cost._result_is_current(result_file, original)

    exp_config["key_sampling_seed"] = 7
    assert not pipeline_user_cost._result_is_current(result_file, provenance())
    exp_config["key_sampling_seed"] = 42

    runtime["threshold"] = -0.3
    assert not pipeline_user_cost._result_is_current(result_file, provenance())
    runtime["threshold"] = -0.25

    runtime["num_processes"] = 4
    assert not pipeline_user_cost._result_is_current(result_file, provenance())
    runtime["num_processes"] = 2

    runtime["detector"] = "mtcnn"
    assert not pipeline_user_cost._result_is_current(result_file, provenance())
    runtime["detector"] = "mediapipe"

    exp_config["protocol"] = "other"
    assert not pipeline_user_cost._result_is_current(result_file, provenance())
    exp_config["protocol"] = "verification"

    system_config["databases"]["database"]["dataset_dir"] = str(
        tmp_path / "different-dataset"
    )
    assert not pipeline_user_cost._result_is_current(result_file, provenance())
    system_config["databases"]["database"]["dataset_dir"] = str(tmp_path / "dataset")

    verification_samples_file.write_text(
        verification_samples_file.read_text() + "2,2,other-image\n"
    )
    assert not pipeline_user_cost._result_is_current(result_file, provenance())
    verification_samples_file.write_text("subject_id,template_id,path\n1,1,image\n")

    verification_file.write_text(verification_file.read_text() + "\n")
    assert not pipeline_user_cost._result_is_current(result_file, provenance())

    pipeline_user_cost._publish_provenance(result_file, provenance())
    result_file.write_text(result_file.read_text() + "subject,selected\n")
    assert not pipeline_user_cost._result_is_current(result_file, provenance())

    pipeline_user_cost._invalidate_provenance(result_file)
    assert not result_file.with_suffix(".csv.provenance.json").exists()


def test_write_cost_file_keeps_partial_output_on_worker_failure(tmp_path, monkeypatch):
    output_file = tmp_path / "cost.csv"
    template = Template("subject", "template", numpy.array([1.0]))

    class FakeWorker:
        @classmethod
        def key_selection_usr_cost(cls, index):
            raise AssertionError("FakeExecutor supplies results directly")

    class FailingExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, indexes, chunksize):
            def results():
                yield {
                    "subject_id": "subject",
                    "template_id": "template",
                    "elapsed_seconds": 0.5,
                    "n_trials": 2,
                    "status": "selected",
                }
                raise RuntimeError("worker failed")

            return results()

    monkeypatch.setattr(pipeline_user_cost, "BTPWorker", FakeWorker)
    monkeypatch.setattr(pipeline_user_cost, "ProcessPoolExecutor", FailingExecutor)

    with pytest.raises(RuntimeError, match="worker failed"):
        pipeline_user_cost._write_cost_file(
            output_file,
            object,
            (),
            True,
            [template],
            object(),
            -0.2,
            0.05,
            lambda *_: 0.0,
            42,
            1,
        )

    assert not output_file.exists()
    partial_file = tmp_path / "cost.csv.partial"
    assert partial_file.exists()
    rows = list(csv.DictReader(partial_file.open()))
    assert len(rows) == 1
    assert rows[0]["row_type"] == "subject"


def test_write_cost_file_handles_no_selected_subjects(tmp_path, monkeypatch):
    output_file = tmp_path / "cost.csv"
    template = Template("subject", "missing", None)

    class FakeWorker:
        @classmethod
        def key_selection_usr_cost(cls, index):
            raise AssertionError("FakeExecutor supplies results directly")

    class MissingExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, indexes, chunksize):
            return iter(
                [
                    {
                        "subject_id": "subject",
                        "template_id": "missing",
                        "elapsed_seconds": 0.0,
                        "n_trials": 0,
                        "status": "missing_template",
                    }
                ]
            )

    monkeypatch.setattr(pipeline_user_cost, "BTPWorker", FakeWorker)
    monkeypatch.setattr(pipeline_user_cost, "ProcessPoolExecutor", MissingExecutor)

    pipeline_user_cost._write_cost_file(
        output_file,
        object,
        (),
        True,
        [template],
        object(),
        -0.2,
        0.05,
        lambda *_: 0.0,
        None,
        1,
    )

    rows = list(csv.DictReader(output_file.open()))
    assert rows[-1]["status"] == "no_selection"
    assert rows[-1]["n_selected"] == "0"
    assert rows[-1]["elapsed_seconds"] == ""
    assert rows[-1]["n_trials"] == ""
