# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import copy

from functools import partial
from pathlib import Path

import numpy
import pytest

from btpbench.baselines.iresnet import IResNet100
from btpbench.btps.biohash import BioHash
from btpbench.dataloader import Sample, cv_loader
from btpbench.scripts.workers import BTPWorker, FRWorker


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
    return Sample("1", "0", img_path, metadata, load_f)


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
    return Sample("1", "1", img_path, metadata, load_f)


@pytest.fixture()
def samples(enroll_sample, probe_sample):
    return [copy.deepcopy(enroll_sample), copy.deepcopy(probe_sample)]


@pytest.fixture()
def templates(enroll_sample, probe_sample):
    output_dir = Path(__file__).parent / "output"
    model = IResNet100(output_dir, False)

    enroll_sample = model.preprocessor(copy.deepcopy(enroll_sample))
    probe_sample = model.preprocessor(copy.deepcopy(probe_sample))

    enroll_template = model.feature_extraction(enroll_sample)
    probe_template = model.feature_extraction(probe_sample)
    output_dir.rmdir()
    return [enroll_template, probe_template]


@pytest.fixture()
def protected_templates(templates):
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    protector = BioHash(output_dir, False)
    output_dir.rmdir()
    return [protector.protect(template) for template in templates]


def test_fr_worker(samples, templates):
    output_dir = Path(__file__).parent / "output"
    bio_alg_cls = IResNet100
    bio_alg_args = (output_dir, False)

    FRWorker.init(bio_alg_cls, bio_alg_args, False, samples, None, None)

    worker_templates = [FRWorker.feature_extraction(i) for i in range(len(samples))]

    assert len(worker_templates) == len(templates)
    assert worker_templates[0].template_id == templates[0].template_id
    assert worker_templates[1].template_id == templates[1].template_id
    assert numpy.array_equal(
        worker_templates[1].get_template(),
        templates[1].get_template(),
    )
    assert numpy.array_equal(
        worker_templates[0].get_template(),
        templates[0].get_template(),
    )

    FRWorker.init(
        bio_alg_cls, bio_alg_args, False, None, [templates[0]], [templates[1]]
    )
    score, t1, t2 = FRWorker.compare((0, 0))

    FRWorker.init(bio_alg_cls, bio_alg_args, False, None, templates, None)
    score2, t3, t4 = FRWorker.compare_verification((0, 1))

    assert score == score2
    assert numpy.array_equal(t1.get_template(), t3.get_template())
    assert numpy.array_equal(t2.get_template(), t4.get_template())

    output_dir.rmdir()


def test_btp_worker(templates, protected_templates):
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    btp_alg_cls = BioHash
    btp_alg_args = (output_dir, False)

    BTPWorker.init(btp_alg_cls, btp_alg_args, False, templates, None, None)

    worker_prot_templates = [BTPWorker.protect(i) for i in range(len(templates))]
    assert len(worker_prot_templates) == len(protected_templates)
    assert worker_prot_templates[0].template_id == protected_templates[0].template_id
    assert worker_prot_templates[1].template_id == protected_templates[1].template_id
    assert numpy.array_equal(
        worker_prot_templates[1].get_template(),
        protected_templates[1].get_template(),
    )
    assert numpy.array_equal(
        worker_prot_templates[0].get_template(),
        protected_templates[0].get_template(),
    )

    BTPWorker.init(
        btp_alg_cls, btp_alg_args, False, None, protected_templates, protected_templates
    )

    score1, t1, t2 = BTPWorker.compare((0, 0))
    score2, t3, t4 = BTPWorker.compare((0, 1))
    score3, t5, t6 = BTPWorker.compare((1, 0))
    score4, t7, t8 = BTPWorker.compare((1, 0))

    assert t7.template_id == t5.template_id
    assert t7.template_id == t4.template_id
    assert score2 == score3

    output_dir.rmdir()
