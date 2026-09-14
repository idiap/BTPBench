# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

from functools import partial
from pathlib import Path

import numpy
import pytest

from btpbench.dataloader import cv_loader
from btpbench.preprocessor import (
    MTCNNDetector,
    chanel_first,
    normalize,
    reshape,
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
    return Sample(1, 1, img_path, metadata, load_f)


@pytest.fixture()
def probe_sample(test_dir):
    img_path = test_dir / "probe.JPG"
    load_f = partial(cv_loader, img_path)
    metadata: dict[str, int] = dict()
    metadata["leye_x"] = 275
    metadata["leye_y"] = 274
    metadata["reye_x"] = 183
    metadata["reye_y"] = 281
    return Sample(1, 1, img_path, metadata, load_f)


def test_reshape(enroll_sample):
    reshaped_image = reshape(enroll_sample, 301, 605)
    img = reshaped_image.data()

    assert img.shape[0] == 301
    assert img.shape[1] == 605


def test_chanel_first(enroll_sample):
    normalized_image = chanel_first(reshape(enroll_sample, 100, 100))
    img = normalized_image.data()

    assert len(img.shape) == 4
    assert img.shape[0] == 1
    assert img.shape[1] == 3
    assert img.shape[2] == 100
    assert img.shape[3] == 100


def test_normalize(enroll_sample):
    normalized_image = normalize(enroll_sample)
    img = normalized_image.data()

    assert numpy.min(img) >= -1 and numpy.min(img) <= 1
    assert numpy.max(img) >= -1 and numpy.max(img) <= 1


def test_mtcnn_detector(enroll_sample):
    detector = MTCNNDetector((145, 145))
    new_sample = detector(enroll_sample)

    img = new_sample.data()
    h, w, c = img.shape

    assert c == 3
    assert h == w
    assert h == 145

    assert img.max() > 1
    assert img.min() >= 0
