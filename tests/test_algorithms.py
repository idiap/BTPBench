# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

from btpbench.algorithms import get_baseline_dict, get_protected_baseline_dict


def test_algorithm():
    baselines = get_baseline_dict()
    protected_baselines = get_protected_baseline_dict()

    for baseline in ["iresnet50", "iresnet100", "edgeface", "edgefacexs", "facenet"]:
        assert baseline in baselines

    assert set(protected_baselines) == {"biohash", "polyprotect", "combined"}
