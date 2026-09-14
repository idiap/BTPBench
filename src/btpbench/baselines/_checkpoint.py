# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import hashlib

from pathlib import Path
from typing import Any

import torch


def load_verified_torch_checkpoint(
    checkpoint_url: str,
    expected_sha256: str,
    checkpoint_file: Path,
) -> Any:
    """Download a checkpoint when needed, verify it, and load it safely."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    actual_hash = None
    if checkpoint_file.exists():
        with checkpoint_file.open("rb") as handle:
            actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()

    if actual_hash != expected_sha256:
        torch.hub.download_url_to_file(
            checkpoint_url,
            str(checkpoint_file),
            hash_prefix=expected_sha256,
        )

    return torch.load(checkpoint_file, map_location="cpu", weights_only=True)
