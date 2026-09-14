# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy


@dataclass
class Sample:
    """Dataclass containing sample information."""

    subject_id: str
    template_id: str
    path: Path
    metadata: dict[str, Any]
    data: Callable[[], numpy.ndarray]
