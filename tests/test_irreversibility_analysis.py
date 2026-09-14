# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import pandas
import pytest

from btpbench.scripts.irreversibility.analysis import (
    calculate_inversion_rates,
)


def test_inversion_rates_count_repeated_attack_trials_directly(tmp_path):
    score_file = tmp_path / "irreversibility.csv"
    pandas.DataFrame(
        [
            _score_row(0.9, True, 0),
            _score_row(0.4, True, 1),
            _score_row(0.8, True, 2),
            _score_row(float("nan"), False, 3),
        ]
    ).to_csv(score_file, index=False)

    rates = calculate_inversion_rates(score_file, [0.5])

    assert rates.solution_rate == pytest.approx(0.75)
    assert rates.match_rates[0] == pytest.approx(2 / 3)
    assert rates.success_rates[0] == pytest.approx(0.5)


def test_inversion_rates_remain_compatible_with_legacy_scores(tmp_path):
    score_file = tmp_path / "legacy.csv"
    rows = [_score_row(0.9, True, 0), _score_row(float("nan"), False, 1)]
    for row in rows:
        row.pop("probe_attack_solved")
    pandas.DataFrame(rows).to_csv(score_file, index=False)

    rates = calculate_inversion_rates(score_file, [0.5])

    assert rates.solution_rate == pytest.approx(0.5)
    assert rates.match_rates[0] == pytest.approx(1.0)
    assert rates.success_rates[0] == pytest.approx(0.5)


def _score_row(score, solved, trial):
    return {
        "probe_template_id": f"inverted-{trial}",
        "probe_subject_id": "subject",
        "bio_ref_template_id": "original",
        "bio_ref_subject_id": "subject",
        "score": score,
        "probe_attack_solved": solved,
    }
