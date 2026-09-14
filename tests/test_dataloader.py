# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

from pathlib import Path

import numpy
import pytest

from btpbench import dataloader


@pytest.fixture()
def protocol_dir():
    return (
        Path(__file__).parent.resolve()
        / "assets"
        / "test_protocols"
        / "grandtest"
        / "dev"
    )


@pytest.fixture()
def protocols_dir():
    return Path(__file__).parent.resolve() / "assets" / "test_protocols"


@pytest.fixture()
def video_file():
    return Path(__file__).parent.resolve() / "assets" / "test_loader" / "output.mp4"


@pytest.fixture()
def dataset_dir():
    return Path(__file__).parent.resolve() / "assets" / "test_dataset"


@pytest.fixture()
def protcol_files(protocol_dir):
    return ((protocol_dir / "for_enrolling.csv"), (protocol_dir / "for_probing.csv"))


def test_assets_exists(protocol_dir, dataset_dir):
    assert protocol_dir.exists()
    assert dataset_dir.exists()

    assert (protocol_dir / "for_enrolling.csv").exists()
    assert (protocol_dir / "for_probing.csv").exists()

    assert len(list(dataset_dir.glob("*.png"))) == 4


def test_video_load(video_file):
    img = dataloader.cv_loader_video(video_file)
    assert numpy.sum(img) == 4770000


def test_dataset(protocols_dir, dataset_dir):
    dataset = dataloader.Dataset(
        protocols_dir,
        dataset_dir,
        protocols_dir / "all-samples.csv",
        protocols_dir / "all-samples.csv",
        extension=".png",
        compliant=False,
    )

    protocols = dataset.protocols()
    assert len(protocols) == 1
    assert protocols[0] == "grandtest"

    ref_metadata = dataset.metadata_names()

    assert len(dataset.samples()) == 4
    assert len(dataset.samples(1)) == 2
    assert len(dataset.samples(2)) == 4

    assert "metadata_1" in ref_metadata
    assert "metadata_0" in ref_metadata

    splits = dataset.protocol_splits("grandtest")
    assert len(splits) == 1
    assert splits[0] == "dev"

    loader = dataset.load_protocol("grandtest", "dev")
    refs = loader.references()
    probes = loader.probes()
    refs_l = list(refs)
    probes_l = list(probes)

    assert len(refs_l) == 2
    assert len(probes_l) == 2
    assert loader.metadata_names() == ["metadata_1", "metadata_0"]


def test_dataload(protcol_files, dataset_dir):
    loader = dataloader.Dataloader(
        protcol_files,
        dataset_dir,
        extension=".png",
        compliant=False,
    )

    refs = loader.references()
    probes = loader.probes()
    refs_l = list(refs)
    probes_l = list(probes)
    assert len(refs_l) == 2
    assert len(probes_l) == 2

    for r in refs_l:
        data = r.data()
        h, w, c = data.shape

        assert h == 100
        assert w == 100
        assert c == 3

        assert data[0][0][0] == 214
        assert data[0][0][1] == 45
        assert data[0][0][2] == 222

        assert "metadata_0" in r.metadata
        assert "metadata_1" in r.metadata
        assert "path" not in r.metadata
        assert "subject_id" not in r.metadata
        assert "template_id" not in r.metadata

        assert r.path.exists()

        assert r.metadata["metadata_1"] == f"x_{r.subject_id}"


def test_dataloader_rejects_mismatched_metadata_columns(tmp_path, dataset_dir):
    enroll_file = tmp_path / "enroll.csv"
    probe_file = tmp_path / "probe.csv"
    enroll_file.write_text("path,subject_id,template_id,session\n1_0,1,0,one\n")
    probe_file.write_text("path,subject_id,template_id,camera\n1_1,1,1,front\n")

    with pytest.raises(ValueError, match="same metadata columns"):
        dataloader.Dataloader(
            (enroll_file, probe_file),
            dataset_dir,
            extension=".png",
            compliant=False,
        )
