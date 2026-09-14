# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

from btpbench.baselines.edgeface import EdgeFaceBase, EdgeFaceXS
from btpbench.baselines.facenet import FaceNet
from btpbench.baselines.iresnet import IResNet50, IResNet100
from btpbench.baselines.random import (
    Random,
)
from btpbench.btps.biohash import BioHash
from btpbench.btps.combined import CombinedBTPAlgs
from btpbench.btps.polyprotect import PolyProtect


def get_baseline_dict():
    """Return baselines dictionary."""
    baseline_dict: dict[str, type] = dict()
    baseline_dict["iresnet100"] = IResNet100
    baseline_dict["iresnet50"] = IResNet50
    baseline_dict["edgeface"] = EdgeFaceBase
    baseline_dict["edgefacexs"] = EdgeFaceXS
    baseline_dict["facenet"] = FaceNet
    baseline_dict["random"] = Random
    return baseline_dict


def get_protected_baseline_dict():
    """Return protected baselines dictionary."""
    protected_baseline_dict: dict[str, type] = dict()
    protected_baseline_dict["biohash"] = BioHash
    protected_baseline_dict["polyprotect"] = PolyProtect
    protected_baseline_dict["combined"] = CombinedBTPAlgs
    return protected_baseline_dict
