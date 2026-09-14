# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import copy

from functools import partial
from math import isclose
from pathlib import Path

import numpy
import pytest

from btpbench.baselines.edgeface import EdgeFaceBase, EdgeFaceXS
from btpbench.baselines.facenet import FaceNet
from btpbench.baselines.iresnet import IResNet50, IResNet100
from btpbench.baselines.random import Random
from btpbench.dataloader import cv_loader
from btpbench.preprocessor import (
    MediaPipeDetector,
    chanel_first,
    normalize,
)
from btpbench.sample import Sample


@pytest.fixture()
def test_dir():
    return Path(__file__).parent / "assets" / "test_preprocessor"


@pytest.fixture()
def enroll_sample(test_dir):
    img_path = test_dir / "enroll.JPG"
    load_f = partial(cv_loader, img_path)
    metadata: dict[str, int] = dict()
    metadata["leye_x"] = 213
    metadata["leye_y"] = 247
    metadata["reye_x"] = 111
    metadata["reye_y"] = 255
    metadata["present"] = True
    return Sample(1, 0, img_path, metadata, load_f)


@pytest.fixture()
def probe_sample(test_dir):
    img_path = test_dir / "probe.JPG"
    load_f = partial(cv_loader, img_path)
    metadata: dict[str, int] = dict()
    metadata["leye_x"] = 275
    metadata["leye_y"] = 274
    metadata["reye_x"] = 183
    metadata["reye_y"] = 281
    metadata["present"] = True
    return Sample(1, 1, img_path, metadata, load_f)


def test_iresnet_no_backup(enroll_sample, probe_sample):
    output_dir = Path(__file__).parent / "output"

    model = IResNet100(output_dir, False)
    enroll_sample = model.preprocessor(enroll_sample)
    probe_sample = model.preprocessor(probe_sample)
    enroll_template = model.feature_extraction(enroll_sample)
    probe_template = model.feature_extraction(probe_sample)

    assert enroll_template.metadata["present"]
    assert probe_template.metadata["present"]

    score = model.compare(enroll_template, probe_template)
    assert score > -0.5

    score = model.compare(enroll_template, enroll_template)
    assert isclose(score, 0, abs_tol=1e-5)

    assert not (output_dir / "1_0.npy").exists()
    assert not (output_dir / "1_1.npy").exists()

    output_dir.rmdir()


def test_iresnet50_no_backup(enroll_sample, probe_sample):
    output_dir = Path(__file__).parent / "output"

    model = IResNet50(output_dir, False)
    enroll_sample = model.preprocessor(enroll_sample)
    probe_sample = model.preprocessor(probe_sample)

    enroll_template = model.feature_extraction(enroll_sample)
    probe_template = model.feature_extraction(probe_sample)

    score = model.compare(enroll_template, probe_template)
    assert score > -0.5

    score = model.compare(enroll_template, enroll_template)
    assert isclose(score, 0, abs_tol=1e-5)
    output_dir.rmdir()


def test_edgefacexs_no_backup(enroll_sample, probe_sample):
    output_dir = Path(__file__).parent / "output"

    model = EdgeFaceXS(output_dir, False)
    enroll_sample = model.preprocessor(enroll_sample)
    probe_sample = model.preprocessor(probe_sample)

    enroll_template = model.feature_extraction(enroll_sample)
    probe_template = model.feature_extraction(probe_sample)

    score = model.compare(enroll_template, probe_template)
    assert score > -0.6

    score = model.compare(enroll_template, enroll_template)
    assert isclose(score, 0, abs_tol=1e-5)
    output_dir.rmdir()


def test_facenet_no_backup(enroll_sample, probe_sample):
    output_dir = Path(__file__).parent / "output"

    model = FaceNet(output_dir, False)
    enroll_sample = model.preprocessor(enroll_sample)
    probe_sample = model.preprocessor(probe_sample)

    enroll_template = model.feature_extraction(enroll_sample)
    probe_template = model.feature_extraction(probe_sample)

    score = model.compare(enroll_template, probe_template)
    assert score > -0.5

    score = model.compare(enroll_template, enroll_template)
    assert isclose(score, 0, abs_tol=1e-5)
    output_dir.rmdir()


def test_iresnet(enroll_sample, probe_sample):
    backup_sample = copy.deepcopy(enroll_sample)

    detector = MediaPipeDetector()

    enroll_sample = detector(enroll_sample)
    enroll_sample = normalize(enroll_sample)
    enroll_sample = chanel_first(enroll_sample)

    probe_sample = detector(probe_sample)
    probe_sample = normalize(probe_sample)
    probe_sample = chanel_first(probe_sample)

    output_dir = Path(__file__).parent / "output"

    model = IResNet100(output_dir, True)

    assert numpy.array_equal(
        enroll_sample.data(), model.preprocessor(backup_sample).data()
    )

    enroll_template = model.feature_extraction(enroll_sample)
    probe_template = model.feature_extraction(probe_sample)

    score = model.compare(enroll_template, probe_template)
    assert score > -0.6

    score = model.compare(enroll_template, enroll_template)
    assert isclose(score, 0, abs_tol=1e-5)

    assert (output_dir / "1" / "1_0.npy").exists()
    assert (output_dir / "1" / "1_1.npy").exists()

    (output_dir / "1" / "1_0.npy").unlink()
    (output_dir / "1" / "1_1.npy").unlink()
    (output_dir / "1").rmdir()
    output_dir.rmdir()


def test_edgeface(enroll_sample, probe_sample):
    backup_sample = copy.deepcopy(enroll_sample)
    detector = MediaPipeDetector()

    enroll_sample = detector(enroll_sample)
    enroll_sample = normalize(enroll_sample)
    enroll_sample = chanel_first(enroll_sample)

    probe_sample = detector(probe_sample)
    probe_sample = normalize(probe_sample)
    probe_sample = chanel_first(probe_sample)

    output_dir = Path(__file__).parent / "output"

    model = EdgeFaceBase(output_dir, True)

    assert numpy.array_equal(
        enroll_sample.data(), model.preprocessor(backup_sample).data()
    )

    enroll_template = model.feature_extraction(enroll_sample)
    probe_template = model.feature_extraction(probe_sample)

    score = model.compare(enroll_template, probe_template)
    assert score > -0.5

    score = model.compare(enroll_template, enroll_template)
    assert isclose(score, 0, abs_tol=1e-5)

    assert (output_dir / "1" / "1_0.npy").exists()
    assert (output_dir / "1" / "1_1.npy").exists()

    (output_dir / "1" / "1_0.npy").unlink()
    (output_dir / "1" / "1_1.npy").unlink()
    (output_dir / "1").rmdir()
    output_dir.rmdir()


def test_random_no_backup(enroll_sample, probe_sample):
    output_dir = Path(__file__).parent / "output"

    model = Random(output_dir, False)

    # Random model doesn't need preprocessing, but we still call it
    enroll_sample = model.preprocessor(enroll_sample)
    probe_sample = model.preprocessor(probe_sample)

    enroll_template = model.feature_extraction(enroll_sample)
    probe_template = model.feature_extraction(probe_sample)

    # Verify template shapes
    assert enroll_template.get_template().shape == (512,)
    assert probe_template.get_template().shape == (512,)

    # Verify templates are within expected range
    assert numpy.all(enroll_template.get_template() >= -5)
    assert numpy.all(enroll_template.get_template() <= 5)
    assert numpy.all(probe_template.get_template() >= -5)
    assert numpy.all(probe_template.get_template() <= 5)

    # Verify comparison works
    score = model.compare(enroll_template, probe_template)
    assert isinstance(score, float | numpy.floating)

    # Verify self-comparison gives 0 (cosine distance of identical vectors)
    score = model.compare(enroll_template, enroll_template)
    assert isclose(score, 0, abs_tol=1e-5)

    # Verify no files were saved
    assert not (output_dir / "1_0.npy").exists()
    assert not (output_dir / "1_1.npy").exists()

    output_dir.rmdir()


def test_random_with_backup(enroll_sample, probe_sample):
    output_dir = Path(__file__).parent / "output"

    model = Random(output_dir, True)

    enroll_sample = model.preprocessor(enroll_sample)
    probe_sample = model.preprocessor(probe_sample)

    enroll_template = model.feature_extraction(enroll_sample)
    probe_template = model.feature_extraction(probe_sample)

    # Verify templates are saved
    assert (output_dir / "1" / "1_0.npy").exists()
    assert (output_dir / "1" / "1_1.npy").exists()

    # Load and verify saved templates
    saved_enroll = numpy.load(output_dir / "1" / "1_0.npy")
    saved_probe = numpy.load(output_dir / "1" / "1_1.npy")

    assert numpy.array_equal(saved_enroll, enroll_template.get_template())
    assert numpy.array_equal(saved_probe, probe_template.get_template())

    # Cleanup
    (output_dir / "1" / "1_0.npy").unlink()
    (output_dir / "1" / "1_1.npy").unlink()
    (output_dir / "1").rmdir()
    output_dir.rmdir()


def test_random_deterministic():
    """Test that Random model produces deterministic results with same generator state."""
    output_dir = Path(__file__).parent / "output"

    # Create a Random model
    model = Random(output_dir, False)

    # Create a dummy sample
    img_path = Path(__file__).parent / "assets" / "test_preprocessor" / "enroll.JPG"
    load_f = partial(cv_loader, img_path)
    metadata = {"present": True}
    sample = Sample(1, 0, img_path, metadata, load_f)

    # Extract features twice - they should be DIFFERENT because generator advances
    template1 = model.feature_extraction(sample)
    template2 = model.feature_extraction(sample)

    # Verify that templates are different (generator state advanced)
    assert not numpy.array_equal(template1.get_template(), template2.get_template())

    # But create two NEW Random instances with same seed - they should produce same FIRST vector
    model_a = Random(output_dir, False)
    model_b = Random(output_dir, False)

    template_a = model_a.feature_extraction(sample)
    template_b = model_b.feature_extraction(sample)

    # First extraction from each should not be identical (seed is different)
    assert not numpy.array_equal(template_a.get_template(), template_b.get_template())

    output_dir.rmdir()
