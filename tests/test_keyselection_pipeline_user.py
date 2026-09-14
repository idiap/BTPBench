# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import json

import numpy

from click.testing import CliRunner

from btpbench.baselines import Template
from btpbench.btps import ProtectedTemplate
from btpbench.scripts.keyselection import pipeline_user


def test_pipeline_user_stops_after_writing_selected_keys(tmp_path, monkeypatch):
    system_file = tmp_path / "system.yaml"
    experiment_file = tmp_path / "experiment.yaml"
    system_file.touch()
    experiment_file.touch()

    experiment = {
        "output_dir": str(tmp_path / "output"),
        "key_sampling_seed": 7,
        "btps": {"algs": [{"type": "dummy", "system_specific": False}]},
    }
    selection_template = Template(
        "subject",
        "template",
        numpy.array([0.25, 0.75]),
    )

    class Dataset:
        def samples(self):
            return [selection_template]

        def metadata_names(self):
            return []

    class BioAlgorithm:
        def __init__(self, *args):
            pass

        def compare(self, reference, probe):
            return float(numpy.dot(reference.get_template(), probe.get_template()))

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

    selected = ProtectedTemplate(
        "subject",
        "template",
        numpy.array([1.0]),
        123,
    )
    reconstructed = Template("subject", "reconstructed", numpy.array([0.5]))

    class FakeExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, indexes, chunksize):
            assert list(indexes) == [0]
            return [(selected, reconstructed, -0.75)]

    monkeypatch.setattr(
        pipeline_user.pipeline_utils,
        "load_config",
        lambda *args: ({}, experiment),
    )
    monkeypatch.setattr(
        pipeline_user.pipeline_utils,
        "load_common_parameters",
        lambda *args: ("dummybio", True, False, True, None, 1),
    )
    monkeypatch.setattr(
        pipeline_user.pipeline_utils,
        "create_dataset",
        lambda *args: ("database", Dataset()),
    )
    monkeypatch.setattr(
        pipeline_user.pipeline_utils,
        "threaded_feature_extraction",
        lambda samples, *args: list(samples),
    )
    monkeypatch.setattr(
        pipeline_user.pipeline_utils,
        "build_btp_alg",
        lambda *args: (
            BTPAlgorithm(),
            BTPAlgorithm,
            (tmp_path, False, experiment["btps"]["algs"][0]),
            tmp_path,
        ),
    )
    monkeypatch.setattr(pipeline_user.pipeline_utils, "check_dir", lambda *_: None)
    monkeypatch.setattr(
        pipeline_user.pipeline_utils, "validate_detector", lambda *_: None
    )
    monkeypatch.setattr(
        pipeline_user.pipeline_utils,
        "processed_protection",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("post-selection protection must not run")
        ),
    )
    monkeypatch.setattr(
        pipeline_user.pipeline_utils,
        "evaluate_inversion_trials",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("post-selection attack validation must not run")
        ),
    )
    monkeypatch.setattr(
        pipeline_user,
        "get_baseline_dict",
        lambda: {"dummybio": BioAlgorithm},
    )
    monkeypatch.setattr(pipeline_user, "get_protected_baseline_dict", lambda: {})
    monkeypatch.setattr(pipeline_user, "ProcessPoolExecutor", FakeExecutor)

    result = CliRunner().invoke(
        pipeline_user.pipeline,
        [
            "-s",
            str(system_file),
            "-e",
            str(experiment_file),
            "--thresh",
            "-0.5",
        ],
    )

    assert result.exit_code == 0, result.output
    keys_files = list(tmp_path.glob("keys_select_*.json"))
    selection_scores = list(tmp_path.glob("ref_keys_select_*.csv"))
    assert len(keys_files) == 1
    assert len(selection_scores) == 1
    assert json.loads(keys_files[0].read_text()) == {"subject": 123}
    assert not list(tmp_path.glob("keys_validation_*.csv"))
