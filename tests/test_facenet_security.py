# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import hashlib

from pathlib import Path

import pytest
import torch

from btpbench.baselines import facenet as facenet_module
from btpbench.baselines.facenet import FaceNet


def _make_checkpoint(path: Path) -> str:
    torch.save({"weight": torch.tensor([1.0])}, path)
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def test_facenet_loads_a_verified_cached_checkpoint(tmp_path, monkeypatch):
    hub_dir = tmp_path / "torch" / "hub"
    checkpoint_file = hub_dir.parent / "checkpoints" / "20180402-114759-vggface2.pt"
    checkpoint_file.parent.mkdir(parents=True)
    expected_hash = _make_checkpoint(checkpoint_file)

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(hub_dir))
    monkeypatch.setattr(FaceNet, "CHECKPOINT_SHA256", expected_hash)

    def unexpected_download(*args, **kwargs):
        pytest.fail("a valid cached checkpoint must not be downloaded again")

    monkeypatch.setattr(torch.hub, "download_url_to_file", unexpected_download)

    state_dict = FaceNet._load_checkpoint()

    assert torch.equal(state_dict["weight"], torch.tensor([1.0]))


def test_facenet_download_uses_the_full_expected_hash(tmp_path, monkeypatch):
    hub_dir = tmp_path / "torch" / "hub"
    source_file = tmp_path / "source.pt"
    expected_hash = _make_checkpoint(source_file)
    observed = {}

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(hub_dir))
    monkeypatch.setattr(FaceNet, "CHECKPOINT_SHA256", expected_hash)
    monkeypatch.setattr(FaceNet, "CHECKPOINT_URL", "https://example.invalid/model.pt")

    def verified_download(url, destination, hash_prefix):
        observed.update(url=url, hash_prefix=hash_prefix)
        Path(destination).write_bytes(source_file.read_bytes())

    monkeypatch.setattr(torch.hub, "download_url_to_file", verified_download)

    state_dict = FaceNet._load_checkpoint()

    assert observed == {
        "url": "https://example.invalid/model.pt",
        "hash_prefix": expected_hash,
    }
    assert torch.equal(state_dict["weight"], torch.tensor([1.0]))


def test_facenet_discards_unused_classification_weights(tmp_path, monkeypatch):
    observed = {}

    class FakeModel:
        def load_state_dict(self, state_dict):
            observed["state_dict"] = state_dict

        def to(self, device):
            observed["device"] = device

        def eval(self):
            observed["evaluated"] = True

    def fake_model(*, pretrained):
        observed["pretrained"] = pretrained
        return FakeModel()

    monkeypatch.setattr(facenet_module, "InceptionResnetV1", fake_model)
    monkeypatch.setattr(
        FaceNet,
        "_load_checkpoint",
        classmethod(
            lambda cls: {
                "embedding.weight": torch.tensor([1.0]),
                "logits.weight": torch.tensor([2.0]),
                "logits.bias": torch.tensor([3.0]),
            }
        ),
    )
    monkeypatch.setattr(facenet_module, "create_detector", lambda *args: object())

    FaceNet(tmp_path / "work", save=False)

    assert observed["pretrained"] is None
    assert observed["state_dict"] == {"embedding.weight": torch.tensor([1.0])}
    assert observed["evaluated"] is True
