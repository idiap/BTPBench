# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import numpy

from btpbench.baselines import Template
from btpbench.utils import (
    derive_seed,
    generate_guesses_from_distribution,
    templates_matrix_distribution,
    templates_to_matrix,
)


def test_utils():
    templates = [Template(str(i), str(i), numpy.ones(10) * i) for i in range(5)]

    mat, subjects = templates_to_matrix(templates, precision=1)

    assert numpy.array_equal(
        subjects,
        ["0", "1", "2", "3", "4"],
    )

    assert numpy.array_equal(
        mat,
        numpy.vstack(
            (
                numpy.ones(10) * 0,
                numpy.ones(10) * 1,
                numpy.ones(10) * 2,
                numpy.ones(10) * 3,
                numpy.ones(10) * 4,
            )
        ),
    )

    dist = templates_matrix_distribution(mat)
    assert numpy.array_equal(dist.mins, numpy.zeros(10))
    assert numpy.array_equal(dist.maxs, numpy.ones(10) * 4)

    for i in range(mat.shape[1]):
        vals, probs = dist.probs[i]
        assert numpy.array_equal([0, 1, 2, 3, 4], vals)
        assert numpy.array_equal(probs, [0.2] * 5)


def test_inversion_randomness_is_reproducible_and_trial_specific():
    templates = [Template(str(i), str(i), numpy.array([i, i + 1])) for i in range(5)]
    matrix, _ = templates_to_matrix(templates)
    distribution = templates_matrix_distribution(matrix)

    first_seed = derive_seed(42, "inversion", 0, "subject", "template", 7)
    repeated_seed = derive_seed(42, "inversion", 0, "subject", "template", 7)
    next_trial_seed = derive_seed(42, "inversion", 1, "subject", "template", 7)

    assert first_seed == repeated_seed
    assert first_seed != next_trial_seed
    assert numpy.array_equal(
        generate_guesses_from_distribution(distribution, 20, first_seed),
        generate_guesses_from_distribution(distribution, 20, repeated_seed),
    )
    assert not numpy.array_equal(
        generate_guesses_from_distribution(distribution, 20, first_seed),
        generate_guesses_from_distribution(distribution, 20, next_trial_seed),
    )
