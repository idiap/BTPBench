# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import hashlib
import json

from dataclasses import dataclass
from typing import Any

import numpy

from btpbench.baselines import Template


@dataclass
class Distribution:
    """Dataclass to store distribution information."""

    means: numpy.ndarray
    mins: numpy.ndarray
    maxs: numpy.ndarray
    probs: list[tuple[numpy.ndarray, numpy.ndarray]]  # Values, probabilities


def generate_guesses_from_distribution(
    template_dist: Distribution,
    num_guesses: int,
    seed: int | None = None,
) -> numpy.ndarray:
    """Generate reproducible guesses from a template distribution."""
    tmpl_size = template_dist.means.shape[0]
    guesses = numpy.zeros((num_guesses, tmpl_size))
    rng = numpy.random.default_rng(seed)
    for i in range(tmpl_size):
        guesses[:, i] = rng.choice(
            template_dist.probs[i][0],
            size=num_guesses,
            p=template_dist.probs[i][1],
        )

    return guesses


def derive_seed(base_seed: int, *components: Any) -> int:
    """Derive a stable, non-zero 32-bit seed from an experiment seed."""
    payload = json.dumps(
        [base_seed, *components],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") % (2**32 - 1) + 1


def templates_to_single_per_subject(
    templates: list[Template],
) -> tuple[list[Template], list[Template]]:
    """Keep only one template per subject.

    The first template found is kept.
    """
    seen = set()
    unique_templates = []
    remaining_templates = []
    for t in templates:
        if t.subject_id not in seen and t.get_template() is not None:
            unique_templates.append(t)
            seen.add(t.subject_id)
        else:
            remaining_templates.append(t)

    return unique_templates, remaining_templates


def templates_to_matrix(
    templates: list[Template], precision: int = 2
) -> tuple[numpy.ndarray, list[str]]:
    """Transform a list of templates to a matrix.

    Each row correspond to a different template.
    """
    raw_templates = [
        t.get_template() for t in templates if t.get_template() is not None
    ]
    subjects = [t.subject_id for t in templates if t.get_template() is not None]

    mat = numpy.vstack(raw_templates)
    if precision > 0:
        mat = numpy.around(mat, decimals=precision)

    return mat, subjects


def templates_matrix_distribution(templates_matrix: numpy.ndarray) -> Distribution:
    """Compute the distribution of a template matrix."""

    means = numpy.mean(templates_matrix, axis=0)
    mins = numpy.min(templates_matrix, axis=0)
    maxs = numpy.max(templates_matrix, axis=0)

    probabilities = []
    for i in range(templates_matrix.shape[1]):
        vals, counts = numpy.unique(templates_matrix[:, i], return_counts=True)
        probabilities.append((vals, counts / templates_matrix.shape[0]))

    return Distribution(means, mins, maxs, probabilities)
