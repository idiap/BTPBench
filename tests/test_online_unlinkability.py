# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import json

import numpy
import pandas
import pytest

from click import ClickException
from click.testing import CliRunner

from btpbench.baselines import Template
from btpbench.btps import ProtectedTemplate
from btpbench.scripts import online_keyselection
from btpbench.scripts.unlinkability import online_pipeline


class DummyBTP:
    """Small process-safe BTP used to exercise the real worker path."""

    def __init__(self, *args):
        pass

    def protect(self, template, key=None):
        result = ProtectedTemplate(
            template.subject_id,
            f"{template.template_id}_k{key}",
            numpy.asarray([float(key % 101)]),
            key,
        )
        result.metadata = template.metadata
        return result

    def compare(self, reference, probe):
        return -float(abs(reference.get_template()[0] - probe.get_template()[0]))


class DummyDataset:
    def metadata_names(self, verification=True):
        return ["sample_index"]


class DummyRun:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.dataset = DummyDataset()
        self.alg = DummyBTP()
        self.alg_cls = DummyBTP
        self.alg_args = ()
        self.keys = None
        self.threshold = None
        self.ks_fmr = 0.05
        self.key_sampling_seed = 42
        self.random_keys = True
        self.n_workers = 2
        self.compliant = False
        self.keys_file = None
        self.verification_file = None

    @property
    def assignment(self):
        return "random"

    def stem(self, samples_per_subject, n_subjects):
        return f"dummy-random-keyseed42-n{samples_per_subject}-subjects{n_subjects}"

    def metadata(self, samples_per_subject, n_subjects):
        return {
            "schema_version": 1,
            "assignment": "random",
            "key_sampling_seed": 42,
            "samples_per_subject": samples_per_subject,
            "n_subjects_requested": n_subjects,
        }


class SelectedDummyBTP(DummyBTP):
    def key_selection_usr_with_stats(
        self,
        template,
        distribution,
        threshold,
        compare,
        seed=None,
        candidate_keys=None,
        excluded_keys=(),
    ):
        del distribution, threshold, compare, seed
        key = next(key for key in candidate_keys if key not in excluded_keys)
        protected = self.protect(template, key)
        return protected, template, -0.75, len(excluded_keys) + 1


def _template(subject_id, index):
    return Template(
        subject_id,
        f"{subject_id}_{index}",
        numpy.asarray([float(index)]),
        {"sample_index": index},
    )


def test_fmr_threshold_uses_only_non_mated_scores(tmp_path):
    score_file = tmp_path / "scores.csv"
    pandas.DataFrame(
        {
            "probe_subject_id": ["a", "a", "a", "b", "c"],
            "bio_ref_subject_id": ["a", "b", "c", "c", "a"],
            "score": [-0.1, -0.9, -0.8, -0.7, -0.6],
        }
    ).to_csv(score_file, index=False)

    threshold = online_keyselection.fmr_threshold(score_file, 0.5)

    assert numpy.isfinite(threshold)
    assert -0.9 <= threshold <= -0.6


def test_fmr_threshold_rejects_blank_subject_ids(tmp_path):
    score_file = tmp_path / "scores.csv"
    pandas.DataFrame(
        {
            "probe_subject_id": ["a", " ", "b"],
            "bio_ref_subject_id": ["b", "c", "a"],
            "score": [-0.9, -0.8, -0.7],
        }
    ).to_csv(score_file, index=False)

    with pytest.raises(ClickException, match="missing subject IDs"):
        online_keyselection.fmr_threshold(score_file, 0.5)


@pytest.mark.parametrize("btps", [None, [], {"algs": [{}]}])
def test_prepare_run_rejects_malformed_btp_configuration(
    tmp_path,
    monkeypatch,
    btps,
):
    system_file = tmp_path / "system.yaml"
    experiment_file = tmp_path / "experiment.yaml"
    system_file.touch()
    experiment_file.touch()
    experiment = {"btps": btps}
    monkeypatch.setattr(
        online_keyselection.pipeline_utils,
        "load_config",
        lambda *args: ({}, experiment),
    )
    monkeypatch.setattr(
        online_keyselection.pipeline_utils,
        "load_common_parameters",
        lambda *args: ("dummy", True, False, False, "mediapipe", 1),
    )

    with pytest.raises(ClickException, match="btps.algs|needs a type"):
        online_keyselection.prepare_run(
            system_file,
            experiment_file,
            None,
            None,
            0.05,
            2,
            -1,
            None,
            True,
        )


def test_load_templates_rejects_duplicate_ids_within_subject(tmp_path):
    duplicate = Template("subject", "duplicate", numpy.asarray([0.0]))

    class DuplicateDataset(DummyDataset):
        def samples(self, verification=True):
            return [duplicate, duplicate]

    run = DummyRun(tmp_path)
    run.dataset = DuplicateDataset()

    with pytest.raises(ClickException, match="duplicate template IDs"):
        online_keyselection.load_templates(run, 2, -1)


def test_random_assignment_is_repeatable_and_distinct_within_subject():
    grouped = {
        "a": [_template("a", index) for index in range(3)],
        "b": [_template("b", index) for index in range(3)],
    }
    initargs = (DummyBTP, (), tuple(range(20)), None, None, None, 42, True)

    online_keyselection._init_worker(*initargs)
    first = [online_keyselection._select_subject(item) for item in grouped.items()]
    online_keyselection._init_worker(*initargs)
    second = [online_keyselection._select_subject(item) for item in grouped.items()]

    first_keys = [[row["key"] for row in result[2]] for result in first]
    second_keys = [[row["key"] for row in result[2]] for result in second]
    assert first_keys == second_keys
    assert all(len(keys) == len(set(keys)) for keys in first_keys)
    assert first_keys[0] != first_keys[1]


def test_selected_assignment_passes_pool_and_exclusions_to_algorithm():
    templates = [_template("subject", index) for index in range(3)]
    distribution = object()
    compare = object()
    online_keyselection._init_worker(
        SelectedDummyBTP,
        (),
        (11, 12, 13),
        distribution,
        -0.5,
        compare,
        42,
        False,
    )

    _, protected, audit = online_keyselection._select_subject(("subject", templates))

    assert [template.get_keys() for template in protected] == [11, 12, 13]
    assert [row["key"] for row in audit] == [11, 12, 13]
    assert [row["n_trials"] for row in audit] == [1, 2, 3]


def test_random_assignment_reports_finite_pool_exhaustion():
    templates = [_template("subject", index) for index in range(3)]
    online_keyselection._init_worker(
        DummyBTP,
        (),
        (11, 12),
        None,
        None,
        None,
        42,
        True,
    )

    with pytest.raises(RuntimeError, match="after assigning 2/3.*pool exhausted"):
        online_keyselection._select_subject(("subject", templates))


def test_completed_outputs_requires_complete_matching_provenance(tmp_path):
    mated = tmp_path / "mated.csv"
    non_mated = tmp_path / "non-mated.csv"
    metadata = tmp_path / "metadata.json"
    expected = {"assignment": "selected", "key_sampling_seed": 42}
    mated.touch()

    with pytest.raises(ClickException, match="Incomplete online outputs"):
        online_keyselection.completed_outputs(
            [mated, non_mated, metadata],
            False,
            expected,
        )

    non_mated.touch()
    metadata.write_text(json.dumps(expected))
    assert online_keyselection.completed_outputs(
        [mated, non_mated, metadata],
        False,
        expected,
    )

    metadata.write_text(json.dumps({**expected, "key_sampling_seed": 7}))
    with pytest.raises(ClickException, match="different settings"):
        online_keyselection.completed_outputs(
            [mated, non_mated, metadata],
            False,
            expected,
        )
    assert not online_keyselection.completed_outputs(
        [mated, non_mated, metadata],
        True,
        expected,
    )


def test_online_pipeline_real_processes_write_scores_and_key_audit(
    tmp_path,
    monkeypatch,
):
    system_file = tmp_path / "system.yaml"
    experiment_file = tmp_path / "experiment.yaml"
    system_file.touch()
    experiment_file.touch()
    run = DummyRun(tmp_path)
    grouped = {
        "a": [_template("a", 0), _template("a", 1)],
        "b": [_template("b", 0), _template("b", 1)],
    }
    monkeypatch.setattr(online_pipeline, "prepare_run", lambda *args: run)
    monkeypatch.setattr(
        online_pipeline,
        "load_templates",
        lambda *args: (grouped, None, None),
    )

    result = CliRunner().invoke(
        online_pipeline.online_pipeline,
        [
            "-s",
            str(system_file),
            "-e",
            str(experiment_file),
            "--random",
            "--samples-per-subject",
            "2",
            "--n-subjects",
            "2",
            "--non-mated-samples-per-subject",
            "1",
            "-o",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    non_mated_file = next(tmp_path.glob("*-non-mated.csv"))
    mated_file = next(
        path for path in tmp_path.glob("*-mated.csv") if path != non_mated_file
    )
    metadata_file = next(tmp_path.glob("*.json"))
    assert len(pandas.read_csv(mated_file)) == 2
    assert len(pandas.read_csv(non_mated_file)) == 1

    metadata = json.loads(metadata_file.read_text())
    assert metadata["assignment"] == "random"
    assert metadata["n_subjects"] == 2
    assert len(metadata["selections"]) == 4
    for subject in ("a", "b"):
        keys = [
            row["key"] for row in metadata["selections"] if row["subject_id"] == subject
        ]
        assert len(keys) == len(set(keys)) == 2


def test_override_keeps_completed_outputs_when_staging_fails(tmp_path, monkeypatch):
    system_file = tmp_path / "system.yaml"
    experiment_file = tmp_path / "experiment.yaml"
    system_file.touch()
    experiment_file.touch()
    run = DummyRun(tmp_path)
    grouped = {
        "a": [_template("a", 0), _template("a", 1)],
        "b": [_template("b", 0), _template("b", 1)],
    }
    protected = {
        subject: [
            run.alg.protect(template, key=index + 1)
            for index, template in enumerate(group)
        ]
        for subject, group in grouped.items()
    }
    stem = "online-unlinkability-dummy-random-keyseed42-n2-subjects2-nm1-seed42"
    mated_file = tmp_path / f"{stem}-mated.csv"
    non_mated_file = tmp_path / f"{stem}-non-mated.csv"
    metadata_file = tmp_path / f"{stem}.json"
    mated_file.write_text("old mated")
    non_mated_file.write_text("old non-mated")
    metadata_file.write_text("old metadata")

    monkeypatch.setattr(online_pipeline, "prepare_run", lambda *args: run)
    monkeypatch.setattr(
        online_pipeline,
        "load_templates",
        lambda *args: (grouped, None, None),
    )
    monkeypatch.setattr(
        online_pipeline,
        "protect_online",
        lambda *args: (protected, []),
    )
    calls = 0

    def fail_second_score_file(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("comparison failed")

    monkeypatch.setattr(
        online_pipeline,
        "_write_protected_scores",
        fail_second_score_file,
    )

    result = CliRunner().invoke(
        online_pipeline.online_pipeline,
        [
            "-s",
            str(system_file),
            "-e",
            str(experiment_file),
            "--random",
            "--samples-per-subject",
            "2",
            "--n-subjects",
            "2",
            "--non-mated-samples-per-subject",
            "1",
            "-o",
            str(tmp_path),
            "--override",
        ],
    )

    assert result.exit_code != 0
    assert mated_file.read_text() == "old mated"
    assert non_mated_file.read_text() == "old non-mated"
    assert metadata_file.read_text() == "old metadata"
    assert not list(tmp_path.glob("*.replacement"))


def test_cli_rejects_invalid_sample_counts_before_loading_configs(tmp_path):
    system_file = tmp_path / "system.yaml"
    experiment_file = tmp_path / "experiment.yaml"
    system_file.touch()
    experiment_file.touch()

    result = CliRunner().invoke(
        online_pipeline.online_pipeline,
        [
            "-s",
            str(system_file),
            "-e",
            str(experiment_file),
            "--random",
            "--samples-per-subject",
            "2",
            "--non-mated-samples-per-subject",
            "3",
        ],
    )

    assert result.exit_code != 0
    assert "Cannot exceed --samples-per-subject" in result.output
