# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import numpy

from click.testing import CliRunner

from btpbench.baselines import Template
from btpbench.btps import ProtectedTemplate
from btpbench.scripts.irreversibility import pipeline as irreversibility


def test_irreversibility_evaluates_multiple_bucket_keys_and_trials(
    tmp_path, monkeypatch
):
    system_file = tmp_path / "system.yaml"
    experiment_file = tmp_path / "experiment.yaml"
    keys_file = tmp_path / "keys.json"
    system_file.touch()
    experiment_file.touch()
    keys_file.write_text('{"keys": {"-0.9": {"one": 11, "two": 12}}}')

    experiment = {
        "output_dir": str(tmp_path / "output"),
        "n_attack_trials": 3,
        "attack_seed": 91,
        "key_sampling_seed": 7,
        "sampling_mode": "keys",
        "n_keys": 2,
        "keys_file": str(keys_file),
        "keys_bucket": "-0.9",
        "btps": {"algs": [{"type": "dummy", "system_specific": True}]},
    }
    reference = Template("background", "reference", numpy.array([0.0, 1.0]))
    target = Template("target", "probe", numpy.array([1.0, 0.0]))

    class Loader:
        def references(self):
            return [reference]

        def probes(self):
            return [target]

        def metadata_names(self):
            return ["camera"]

    class Dataset:
        def load_irreversibility(self):
            return Loader()

    class BioAlgorithm:
        def __init__(self, *args):
            pass

        def compare(self, original, inverted):
            return float(numpy.dot(original.get_template(), inverted.get_template()))

    class BTPAlgorithm:
        def __init__(self):
            self.key = 42

        def get_inversion_config(self):
            return {"precision": 3}

        def get_inversion_config_tag(self):
            return "attack"

        def get_alg_name(self):
            return "dummy"

        def has_key_dictionary(self):
            return False

        def has_key_distribution(self):
            return False

        def is_system_specific(self):
            return True

        def set_key(self, key):
            self.key = key

    class ScoreWriter:
        def __init__(self, path, metadata_names):
            self.path = path
            self.metadata_names = metadata_names

        def close(self):
            pass

    btp = BTPAlgorithm()
    evaluated = []
    writer = ScoreWriter(tmp_path / "unused.csv", [])

    monkeypatch.setattr(
        irreversibility.pipeline_utils,
        "load_config",
        lambda *args: ({}, experiment),
    )
    monkeypatch.setattr(
        irreversibility.pipeline_utils,
        "load_common_parameters",
        lambda *args: ("dummy", True, False, True, None, 1),
    )
    monkeypatch.setattr(
        irreversibility.pipeline_utils,
        "create_dataset",
        lambda *args: ("database", Dataset()),
    )
    monkeypatch.setattr(
        irreversibility.pipeline_utils, "validate_detector", lambda *_: None
    )
    monkeypatch.setattr(irreversibility.pipeline_utils, "check_dir", lambda *_: None)
    monkeypatch.setattr(
        irreversibility.pipeline_utils,
        "threaded_feature_extraction",
        lambda samples, *args: list(samples),
    )
    monkeypatch.setattr(
        irreversibility.pipeline_utils,
        "build_btp_alg",
        lambda *args: (
            btp,
            BTPAlgorithm,
            (tmp_path, False, experiment["btps"]["algs"][0]),
            tmp_path,
        ),
    )

    def protect(templates, _cls, args, *_rest):
        key = args[2]["key"]
        return [
            ProtectedTemplate(
                template.subject_id,
                template.template_id,
                template.get_template(),
                key,
            )
            for template in templates
        ]

    def evaluate(
        originals,
        protected,
        _cls,
        _args,
        _compliant,
        _workers,
        _distribution,
        _chunksize,
        trials,
        seed,
        _compare,
        _writer,
        source,
        scope,
        key_system,
    ):
        evaluated.append(
            (
                originals,
                protected[0].get_keys(),
                trials,
                seed,
                source,
                scope,
                key_system,
            )
        )

    monkeypatch.setattr(irreversibility.pipeline_utils, "processed_protection", protect)
    monkeypatch.setattr(
        irreversibility.pipeline_utils, "evaluate_inversion_trials", evaluate
    )
    monkeypatch.setattr(
        irreversibility, "get_baseline_dict", lambda: {"dummy": BioAlgorithm}
    )
    monkeypatch.setattr(irreversibility, "get_protected_baseline_dict", lambda: {})

    def make_writer(path, metadata_names):
        writer.path = path
        writer.metadata_names = metadata_names
        return writer

    monkeypatch.setattr(irreversibility, "CSVScoreWriter", make_writer)

    result = CliRunner().invoke(
        irreversibility.pipeline,
        ["-s", str(system_file), "-e", str(experiment_file)],
    )

    assert result.exit_code == 0, result.output
    assert {entry[1] for entry in evaluated} == {11, 12}
    assert all(entry[2:6] == (3, 91, "bucket:-0.9", "system") for entry in evaluated)
    assert [entry[6] for entry in evaluated] == [0, 1]
    assert writer.metadata_names == [
        "camera",
        "attack_trial",
        "attack_solved",
        "key",
        "key_system",
        "key_source",
        "key_scope",
    ]
