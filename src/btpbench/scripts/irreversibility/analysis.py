# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import logging

from dataclasses import dataclass
from pathlib import Path

import numpy
import pandas

import btpbench.metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InversionRates:
    """Aggregate inversion rates for one score file."""

    solution_rate: float
    match_rates: numpy.ndarray
    success_rates: numpy.ndarray


def _boolean_series(values: pandas.Series) -> pandas.Series:
    """Parse booleans written by the CSV score writer."""
    if pandas.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"1", "true", "yes"})


def calculate_inversion_rates(
    score_file: Path,
    thresholds: list[float] | numpy.ndarray,
) -> InversionRates:
    """Calculate solution, conditional-match, and overall success rates."""
    columns = pandas.read_csv(score_file, nrows=0).columns
    solved_column = "probe_attack_solved"
    columns_of_interest = [
        "probe_subject_id",
        "bio_ref_subject_id",
        "score",
    ]
    has_explicit_trials = solved_column in columns
    if has_explicit_trials:
        columns_of_interest.append(solved_column)

    score_df = pandas.read_csv(score_file, usecols=columns_of_interest)
    total_attempts = len(score_df)
    if total_attempts == 0:
        return InversionRates(
            solution_rate=0.0,
            match_rates=numpy.zeros(len(thresholds)),
            success_rates=numpy.zeros(len(thresholds)),
        )

    if has_explicit_trials:
        solved = _boolean_series(score_df[solved_column])
        solved_count = int(solved.sum())
        solution_rate = solved_count / total_attempts
        clean_df = score_df[solved & score_df["score"].notna()]
    else:
        fta, clean_df = btpbench.metrics.remove_nans(score_df)
        solution_rate = 1 - fta
        solved_count = 0

    negative_scores, positive_scores = btpbench.metrics.neg_pos_scores(clean_df)
    if len(negative_scores) > 0:
        logger.warning(
            "Ignoring %d non-mated comparison(s) in irreversibility scores.",
            len(negative_scores),
        )
    if not has_explicit_trials:
        solved_count = len(positive_scores)

    matches = numpy.asarray(
        [numpy.sum(positive_scores >= threshold) for threshold in thresholds],
        dtype=float,
    )
    match_rates = (
        matches / solved_count if solved_count else numpy.zeros(len(thresholds))
    )
    success_rates = (
        matches / total_attempts if has_explicit_trials else solution_rate * match_rates
    )
    return InversionRates(solution_rate, match_rates, success_rates)
