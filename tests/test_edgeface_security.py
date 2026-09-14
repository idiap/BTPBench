# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import hashlib

from pathlib import Path

import pytest
import torch

from torch import nn

from btpbench.baselines import edgeface as edgeface_module
from btpbench.baselines.edgeface import EdgeFace


def _make_checkpoint(path: Path) -> str:
    torch.save({"weight": torch.tensor([1.0])}, path)
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def test_edgeface_loads_a_verified_cached_checkpoint(tmp_path, monkeypatch):
    hub_dir = tmp_path / "hub"
    checkpoint_file = hub_dir / "checkpoints" / "test_model.pt"
    checkpoint_file.parent.mkdir(parents=True)
    expected_hash = _make_checkpoint(checkpoint_file)

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(hub_dir))
    monkeypatch.setattr(
        EdgeFace,
        "CHECKPOINTS",
        {"test_model": ("https://example.invalid/test_model.pt", expected_hash)},
    )

    def unexpected_download(*args, **kwargs):
        pytest.fail("a valid cached checkpoint must not be downloaded again")

    monkeypatch.setattr(torch.hub, "download_url_to_file", unexpected_download)

    state_dict = EdgeFace._load_checkpoint("test_model")

    assert torch.equal(state_dict["weight"], torch.tensor([1.0]))


def test_edgeface_download_uses_the_full_expected_hash(tmp_path, monkeypatch):
    hub_dir = tmp_path / "hub"
    source_file = tmp_path / "source.pt"
    expected_hash = _make_checkpoint(source_file)
    observed = {}

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(hub_dir))
    monkeypatch.setattr(
        EdgeFace,
        "CHECKPOINTS",
        {"test_model": ("https://example.invalid/test_model.pt", expected_hash)},
    )

    def verified_download(url, destination, hash_prefix):
        observed.update(url=url, hash_prefix=hash_prefix)
        Path(destination).write_bytes(source_file.read_bytes())

    monkeypatch.setattr(torch.hub, "download_url_to_file", verified_download)

    state_dict = EdgeFace._load_checkpoint("test_model")

    assert observed == {
        "url": "https://example.invalid/test_model.pt",
        "hash_prefix": expected_hash,
    }
    assert torch.equal(state_dict["weight"], torch.tensor([1.0]))


def test_edgeface_loads_hugging_face_checkpoint_without_github(tmp_path, monkeypatch):
    observed = {}

    class FakeModel:
        def load_state_dict(self, state_dict):
            observed["state_dict"] = state_dict

        def to(self, device):
            observed["device"] = device

        def eval(self):
            observed["evaluated"] = True

    def fake_create_model(model):
        observed["model"] = model
        return FakeModel()

    def unexpected_hub_load(*args, **kwargs):
        pytest.fail("EdgeFace must not load executable model code from GitHub")

    monkeypatch.setattr(edgeface_module, "_create_edgeface_model", fake_create_model)
    monkeypatch.setattr(torch.hub, "load", unexpected_hub_load)
    monkeypatch.setattr(
        EdgeFace,
        "_load_checkpoint",
        classmethod(lambda cls, model: {"verified": model}),
    )
    monkeypatch.setattr(edgeface_module, "create_detector", lambda detector: object())

    EdgeFace("edgeface_base", tmp_path / "work", save=False)

    assert observed["model"] == "edgeface_base"
    assert EdgeFace.CHECKPOINTS["edgeface_base"][0] == (
        "https://huggingface.co/Idiap/EdgeFace-Base/resolve/"
        "2944a63d6a73efb3754c553e119b57fe7e54a179/edgeface_base.pt"
    )
    assert EdgeFace.CHECKPOINTS["edgeface_xs_gamma_06"][0] == (
        "https://huggingface.co/Idiap/EdgeFace-XS-GAMMA/resolve/"
        "735c1b59bdc798260e56f12533fddfe8a8c4c568/"
        "edgeface_xs_gamma_06.pt"
    )
    assert observed["state_dict"] == {"verified": "edgeface_base"}
    assert observed["evaluated"] is True


def test_edgeface_builds_the_published_timm_architectures(monkeypatch):
    observed = []

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(10, 20)

        def reset_classifier(self, output_size):
            observed.append(output_size)
            self.head = nn.Linear(20, output_size)

    def fake_timm_create_model(architecture):
        observed.append(architecture)
        return FakeBackbone()

    monkeypatch.setattr(edgeface_module.timm, "create_model", fake_timm_create_model)

    base = edgeface_module._create_edgeface_model("edgeface_base")
    extra_small = edgeface_module._create_edgeface_model("edgeface_xs_gamma_06")

    assert observed == ["edgenext_base", 512, "edgenext_x_small", 512]
    assert isinstance(base.model.projection, nn.Linear)
    assert isinstance(extra_small.model.projection, nn.Sequential)
    assert extra_small.model.projection.linear1.out_features == 6
    assert extra_small.model.projection.linear2.in_features == 6
    assert isinstance(extra_small.model.head, nn.Linear)
