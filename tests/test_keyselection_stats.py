# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import numpy
import pytest

from btpbench.baselines import Template
from btpbench.btps import (
    BaselineBTP,
    KeySelectionStats,
    ProtectedTemplate,
    _load_candidate_keys,
)
from btpbench.scripts import workers
from btpbench.scripts.workers import BTPWorker
from btpbench.utils import Distribution, derive_seed


class ControlledBTP(BaselineBTP):
    def __init__(self, work_dir, scores, method="legacy"):
        super().__init__(
            work_dir,
            save=False,
            config={
                "system_specific": False,
                "normalize_input": False,
                "ks_method": method,
                "ks_n_elements": 1,
            },
        )
        self.scores = iter(scores)
        self.inversion_seeds = []

    def get_secret(self, key):
        return {}

    def _protect(self, feat_vec, key):
        return numpy.asarray([key], dtype=float)

    def compare(self, ref, probe):
        return float(probe.get_template()[0])

    def get_alg_name(self):
        return "controlled"

    def invert(self, protected_template, template_dist, dont_save=False, seed=None):
        self.inversion_seeds.append(seed)
        score = next(self.scores)
        value = None if score is None else numpy.asarray([score])
        return Template(
            protected_template.subject_id,
            protected_template.template_id,
            value,
        )

    def _invert_atomic(self, protected_template, key, initial_guess, seed=None):
        self.inversion_seeds.append(seed)
        score = next(self.scores)
        return None if score is None else numpy.asarray([score])


@pytest.fixture()
def template():
    return Template("subject", "template", numpy.asarray([1.0]))


@pytest.fixture()
def distribution():
    values = numpy.asarray([1.0])
    return Distribution(values, values, values, [(values, values)])


def compare_score(_reference, probe):
    return float(probe.get_template()[0])


def test_load_candidate_keys_supported_formats_and_validation(tmp_path):
    list_path = tmp_path / "list.json"
    list_path.write_text("[7, 3, 7]", encoding="utf-8")
    assert _load_candidate_keys(list_path) == (7, 3)

    map_path = tmp_path / "map.json"
    map_path.write_text('{"subject-a": 8, "subject-b": 2}', encoding="utf-8")
    assert _load_candidate_keys(map_path) == (8, 2)

    text_path = tmp_path / "keys.txt"
    text_path.write_text("9, 4\n9", encoding="utf-8")
    assert _load_candidate_keys(text_path) == (9, 4)

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("[true]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain integers"):
        _load_candidate_keys(invalid_path)


def test_legacy_stats_count_distinct_trials_and_preserve_tuple_api(
    tmp_path,
    template,
    distribution,
):
    stats = KeySelectionStats(n_trials=99)
    alg = ControlledBTP(tmp_path, [0.9, 0.1])
    result = alg.key_selection_usr(
        template,
        distribution,
        0.5,
        compare_score,
        seed=42,
        stats=stats,
        candidate_keys=(10, 20),
    )

    assert len(result) == 3
    assert isinstance(result[0], ProtectedTemplate)
    assert result[2] == 0.1
    assert stats.n_trials == 2


def test_stats_path_preserves_default_seed_sequence(tmp_path, template, distribution):
    expected_rng = numpy.random.default_rng(42)
    expected_key = int(expected_rng.integers(0, 2_000_000))
    expected_inversion_seed = int(expected_rng.integers(1, 2**32))

    alg = ControlledBTP(tmp_path, [0.1])
    protected, _, _, n_trials = alg.key_selection_usr_with_stats(
        template,
        distribution,
        0.5,
        compare_score,
        seed=42,
    )

    assert protected.get_keys() == expected_key
    assert alg.inversion_seeds == [expected_inversion_seed]
    assert n_trials == 1


def test_legacy_accepts_failed_inversion_and_respects_exclusions(
    tmp_path,
    template,
    distribution,
):
    alg = ControlledBTP(tmp_path, [None])
    protected, _, score, n_trials = alg.key_selection_usr_with_stats(
        template,
        distribution,
        0.5,
        compare_score,
        seed=7,
        candidate_keys=(10, 20),
        excluded_keys={10},
    )

    assert protected.get_keys() == 20
    assert numpy.isnan(score)
    assert n_trials == 1


def test_multiple_guesses_counts_trials_and_exhausts_pool(
    tmp_path,
    template,
    distribution,
):
    alg = ControlledBTP(tmp_path, [0.9, 0.1], method="multiple_guesses")
    protected, _, score, n_trials = alg.key_selection_usr_with_stats(
        template,
        distribution,
        0.5,
        compare_score,
        seed=42,
        candidate_keys=(10, 20),
    )
    assert protected.get_keys() in {10, 20}
    assert score == 0.1
    assert n_trials == 2

    exhausted = ControlledBTP(tmp_path, [], method="multiple_guesses")
    with pytest.raises(RuntimeError, match="exhausting all options"):
        exhausted.key_selection_usr_with_stats(
            template,
            distribution,
            0.5,
            compare_score,
            seed=42,
            candidate_keys=(10, 20),
            excluded_keys={10, 20},
        )


class WorkerAlgorithm:
    def __init__(self):
        self.seeds = []

    def key_selection_usr(self, template, dist, thresh, compare_f, seed):
        self.seeds.append(seed)
        return (
            ProtectedTemplate(
                template.subject_id,
                template.template_id,
                template.get_template(),
                1,
            ),
            template,
            0.0,
        )

    def key_selection_usr_with_stats(self, template, dist, thresh, compare_f, seed):
        self.seeds.append(seed)
        return (
            ProtectedTemplate(
                template.subject_id,
                template.template_id,
                template.get_template(),
                1,
            ),
            template,
            0.0,
            3,
        )


def test_worker_cost_row_timing_status_and_seed(monkeypatch, template):
    missing = Template("missing", "missing-template", None)
    BTPWorker.init(
        WorkerAlgorithm,
        (),
        True,
        [template, missing],
        None,
        None,
        {
            "ref_distribution": object(),
            "thresh": 0.5,
            "compare_f": object(),
            "key_sampling_seed": 42,
        },
    )
    clock = iter((10.0, 10.25))
    monkeypatch.setattr(workers, "perf_counter", lambda: next(clock))

    selected = BTPWorker.key_selection_usr_cost(0)
    assert selected == {
        "subject_id": "subject",
        "template_id": "template",
        "elapsed_seconds": 0.25,
        "n_trials": 3,
        "status": "selected",
    }
    expected_seed = derive_seed(42, "key-selection", "subject", "template")
    assert BTPWorker._instance.alg.seeds == [expected_seed]

    assert BTPWorker.key_selection_usr_cost(1) == {
        "subject_id": "missing",
        "template_id": "missing-template",
        "elapsed_seconds": 0.0,
        "n_trials": 0,
        "status": "missing_template",
    }
