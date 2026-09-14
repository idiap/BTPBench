# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import json

from functools import partial
from pathlib import Path

import numpy
import pytest

from btpbench.baselines import Template
from btpbench.baselines.iresnet import IResNet100
from btpbench.btps import BaselineBTP, ProtectedTemplate
from btpbench.btps.biohash import BioHash
from btpbench.btps.combined import CombinedBTPAlgs
from btpbench.btps.polyprotect import PolyProtect
from btpbench.dataloader import cv_loader
from btpbench.exceptions import InvalidInputSampleError
from btpbench.sample import Sample
from btpbench.utils import Distribution, templates_matrix_distribution


class DummyBioHash(BaselineBTP):
    def __init__(self, work_dir, save=False, config=..., **kwargs):
        super().__init__(work_dir, save, config, **kwargs)

    def _protect(self, feat_vec, key):
        return feat_vec

    def get_secret(self, key):
        return dict()

    def compare(self, ref, probe):
        return 42

    def get_alg_name(self):
        return "toto"


class DummyBTP(BaselineBTP):
    def __init__(self, work_dir, save=False, config=dict(), **kwargs):
        super().__init__(work_dir, save, config, **kwargs)

    def get_alg_name(self):
        return f"toto_{self._key_offset}"

    def get_secret(self, key):
        return dict()

    def _protect(self, feat_vec, key):
        return feat_vec + self._key_offset

    def compare(self, ref, probe):
        return self._key_offset


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
def templates(enroll_sample, probe_sample):
    output_dir = Path(__file__).parent / "output"
    model = IResNet100(output_dir, False)
    enroll_sample = model.preprocessor(enroll_sample)
    probe_sample = model.preprocessor(probe_sample)
    enroll_template = model.feature_extraction(enroll_sample)
    probe_template = model.feature_extraction(probe_sample)
    output_dir.rmdir()
    return enroll_template, probe_template


def test_biohash(templates):
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    protector = BioHash(output_dir, False)

    protected_templates = [protector.protect(template) for template in templates]

    assert protected_templates[0].metadata["present"]
    assert protected_templates[1].metadata["present"]

    protected_templates[0].save(output_dir / "1_1_sys_1.npy")
    assert (output_dir / "1_1_sys_1.npy").exists()

    assert protector.get_alg_name() == "normalized_biohash_usr_binary_64"

    ok, loaded_template = ProtectedTemplate.load("1", "0", output_dir / "1_1_sys_1.npy")

    assert loaded_template.get_keys() == protected_templates[0].get_keys()
    assert loaded_template.subject_id == protected_templates[0].subject_id
    assert loaded_template.template_id == protected_templates[0].template_id
    assert loaded_template.get_keys() == 1

    assert ok
    assert protector.compare(
        loaded_template, protected_templates[0]
    ) == protector.compare(protected_templates[0], protected_templates[0])
    assert protector.compare(protected_templates[0], protected_templates[0]) == 0
    assert protector.compare(protected_templates[0], protected_templates[1]) >= -0.3

    (output_dir / "1_1_sys_1.npy").unlink()
    output_dir.rmdir()


def test_biohash_real_compare_uses_cosine(tmp_path):
    protector = BioHash(
        tmp_path,
        False,
        {
            "binarize": False,
            "normalize_input": False,
            "num_bits": 2,
            "num_features": 2,
        },
    )
    ref = ProtectedTemplate("1", "0", numpy.array([1.0, 2.0]), 1)
    probe = ProtectedTemplate("2", "0", numpy.array([2.0, 4.0]), 2)

    assert protector.get_alg_name() == "unnormalized_biohash_usr_real_2"
    assert protector.compare(ref, probe) == pytest.approx(0.0)


def test_biohash_surrogate_gradient_and_inversion(tmp_path):
    protector = BioHash(
        tmp_path,
        False,
        {
            "method": "minimize_surrogate",
            "num_features": 6,
            "num_bits": 4,
            "binarize": True,
            "normalize_input": True,
        },
    )
    key = 7
    target_source = numpy.array([0.6, -0.4, 0.2, 0.7, -0.3, 0.1])
    target_source /= numpy.linalg.norm(target_source)
    raw_protected_template = protector._protect(target_source, key)  # noqa: SLF001
    protected_template = protector._postprocess_protected_template(  # noqa: SLF001
        raw_protected_template
    )
    assert not numpy.array_equal(raw_protected_template, protected_template)
    initial_guess = numpy.array([-0.2, 0.5, -0.1, 0.3, 0.4, -0.6])

    _, analytical_gradient = protector._surrogate_loss_and_gradient(  # noqa: SLF001
        initial_guess,
        protected_template,
        key,
        initial_guess,
    )
    epsilon = 1e-6
    numerical_gradient = numpy.empty_like(initial_guess)
    for index in range(len(initial_guess)):
        upper = initial_guess.copy()
        lower = initial_guess.copy()
        upper[index] += epsilon
        lower[index] -= epsilon
        upper_loss = protector._surrogate_loss_and_gradient(  # noqa: SLF001
            upper,
            protected_template,
            key,
            initial_guess,
        )[0]
        lower_loss = protector._surrogate_loss_and_gradient(  # noqa: SLF001
            lower,
            protected_template,
            key,
            initial_guess,
        )[0]
        numerical_gradient[index] = (upper_loss - lower_loss) / (2 * epsilon)

    assert numpy.allclose(
        analytical_gradient,
        numerical_gradient,
        rtol=1e-5,
        atol=1e-7,
    )

    inverted_template = protector._invert_atomic(  # noqa: SLF001
        protected_template,
        key,
        initial_guess,
    )
    assert inverted_template is not None

    inverted_template /= numpy.linalg.norm(inverted_template)
    inverted_biohash = protector._protect_postprocessed(  # noqa: SLF001
        inverted_template,
        key,
    )
    assert numpy.array_equal(inverted_biohash, protected_template)
    assert protector.get_inversion_config() == {
        "method": "minimize_surrogate",
        "num_guesses": 10,
        "precision": 3,
        "surrogate_temperature": 0.1,
        "surrogate_regularization": 0.001,
        "surrogate_maxiter": 500,
    }


def test_biohash_surrogate_requires_binary_output(tmp_path):
    with pytest.raises(ValueError, match="requires binarize"):
        BioHash(
            tmp_path,
            False,
            {
                "method": "minimize_surrogate",
                "num_features": 4,
                "num_bits": 2,
                "binarize": False,
            },
        )


def test_biohash_hard_constraints_and_inversion(tmp_path):
    protector = BioHash(
        tmp_path,
        False,
        {
            "method": "minimize_hard_constraints",
            "num_features": 6,
            "num_bits": 4,
            "binarize": True,
            "normalize_input": True,
            "hard_constraint_epsilon": 1e-8,
            "hard_constraint_maxiter": 1000,
            "hard_constraint_ftol": 1e-10,
        },
    )
    key = 7
    target_source = numpy.array([0.6, -0.4, 0.2, 0.7, -0.3, 0.1])
    raw_protected_template = protector._protect(target_source, key)  # noqa: SLF001
    protected_template = protector._postprocess_protected_template(  # noqa: SLF001
        raw_protected_template
    )
    assert not numpy.array_equal(raw_protected_template, protected_template)
    initial_guess = numpy.array([-0.2, 0.5, -0.1, 0.3, 0.4, -0.6])

    inverted_template = protector._invert_atomic(  # noqa: SLF001
        protected_template,
        key,
        initial_guess,
    )
    assert inverted_template is not None

    constraint_values = protector._hard_constraint_values(  # noqa: SLF001
        inverted_template,
        protected_template,
        key,
    )
    assert numpy.min(constraint_values) >= -1e-8

    inverted_biohash = protector._protect_postprocessed(  # noqa: SLF001
        inverted_template,
        key,
    )
    assert numpy.array_equal(inverted_biohash, protected_template)
    assert protector.get_inversion_config() == {
        "method": "minimize_hard_constraints",
        "num_guesses": 10,
        "precision": 3,
        "hard_constraint_epsilon": 1e-8,
        "hard_constraint_maxiter": 1000,
        "hard_constraint_ftol": 1e-10,
    }


def test_biohash_hard_constraints_requires_binary_output(tmp_path):
    with pytest.raises(ValueError, match="requires binarize"):
        BioHash(
            tmp_path,
            False,
            {
                "method": "minimize_hard_constraints",
                "num_features": 4,
                "num_bits": 2,
                "binarize": False,
            },
        )


def test_polyprotect_invert():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    protector = PolyProtect(
        output_dir,
        False,
        {
            "system_specific": True,
            "overlap": 4,
            "nb_coef": 5,
            "coef_range": 50,
            "key": 42,
            "num_guesses": 20,
            "precision": 3,
            "method": "root",
        },
    )

    assert protector.get_inversion_config_tag() == "root-3-20"
    assert protector.get_inversion_config() == {
        "method": "root",
        "precision": 3,
        "num_guesses": 20,
    }

    unprotected_template = Template("1", "1", numpy.ones(100))

    protected_template = protector.protect(unprotected_template)

    estimated_unprotected_template = protector.invert(
        protected_template,
        templates_matrix_distribution(
            numpy.random.default_rng().normal(1.0, 0.02, (20, 100)),
        ),
    )

    assert (
        protector.compare(unprotected_template, estimated_unprotected_template) >= -0.8
    )


def test_polyprotect(templates):
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    protector = PolyProtect(output_dir, False, {"system_specific": True})

    protected_templates = [protector.protect(template) for template in templates]

    protected_templates[0].save(output_dir / "1_1_sys_42.npy")
    assert (output_dir / "1_1_sys_42.npy").exists()

    assert protector.get_alg_name() == "normalized_polyprotect_sys_2_4_10"

    ok, loaded_template = ProtectedTemplate.load(
        "1", "0", output_dir / "1_1_sys_42.npy"
    )

    assert protected_templates[0].get_keys() == 42
    assert loaded_template.get_keys() == protected_templates[0].get_keys()
    assert loaded_template.subject_id == protected_templates[0].subject_id
    assert loaded_template.template_id == protected_templates[0].template_id

    assert ok
    assert protector.compare(
        loaded_template, protected_templates[0]
    ) == protector.compare(protected_templates[0], protected_templates[0])
    assert protector.compare(protected_templates[0], protected_templates[0]) >= -1e-5
    assert protector.compare(protected_templates[0], protected_templates[1]) >= -0.5

    (output_dir / "1_1_sys_42.npy").unlink()
    output_dir.rmdir()


def test_polyprotect_binarize_and_hamming_compare(tmp_path):
    protector = PolyProtect(
        tmp_path,
        False,
        {
            "binarize": True,
            "normalize_input": False,
            "system_specific": True,
            "overlap": 2,
            "nb_coef": 4,
            "coef_range": 10,
        },
    )
    template = Template("1", "0", numpy.arange(1, 9, dtype=float))
    protected_template = protector.protect(template)

    assert protector.get_alg_name() == "unnormalized_polyprotect_sys_binary_2_4_10"
    assert numpy.all(numpy.isin(protected_template.get_template(), (0, 1)))

    ref = ProtectedTemplate("1", "0", numpy.array([1, 0, 1, 0]), 42)
    probe = ProtectedTemplate("2", "0", numpy.array([1, 1, 0, 0]), 42)

    assert protector.compare(ref, probe) == pytest.approx(-0.5)


def test_polyprotect_surrogate_gradient_and_inversion(tmp_path):
    protector = PolyProtect(
        tmp_path,
        False,
        {
            "method": "minimize_surrogate",
            "binarize": True,
            "normalize_input": True,
            "overlap": 2,
            "nb_coef": 3,
            "coef_range": 5,
        },
    )
    key = 11
    target_source = numpy.array([0.5, -0.4, 0.3, 0.8, -0.2, 0.1])
    target_source /= numpy.linalg.norm(target_source)
    raw_protected_template = protector._protect(target_source, key)  # noqa: SLF001
    protected_template = protector._postprocess_protected_template(  # noqa: SLF001
        raw_protected_template
    )
    assert not numpy.array_equal(raw_protected_template, protected_template)
    initial_guess = numpy.array([-0.2, 0.4, -0.3, 0.5, 0.1, -0.6])

    _, analytical_gradient = protector._surrogate_loss_and_gradient(  # noqa: SLF001
        initial_guess,
        protected_template,
        key,
        initial_guess,
    )
    epsilon = 1e-6
    numerical_gradient = numpy.empty_like(initial_guess)
    for index in range(len(initial_guess)):
        upper = initial_guess.copy()
        lower = initial_guess.copy()
        upper[index] += epsilon
        lower[index] -= epsilon
        upper_loss = protector._surrogate_loss_and_gradient(  # noqa: SLF001
            upper,
            protected_template,
            key,
            initial_guess,
        )[0]
        lower_loss = protector._surrogate_loss_and_gradient(  # noqa: SLF001
            lower,
            protected_template,
            key,
            initial_guess,
        )[0]
        numerical_gradient[index] = (upper_loss - lower_loss) / (2 * epsilon)

    assert numpy.allclose(
        analytical_gradient,
        numerical_gradient,
        rtol=1e-4,
        atol=1e-6,
    )

    inverted_template = protector._invert_atomic(  # noqa: SLF001
        protected_template,
        key,
        target_source.copy(),
    )
    assert inverted_template is not None

    inverted_template /= numpy.linalg.norm(inverted_template)
    inverted_polyprotect = protector._protect_postprocessed(  # noqa: SLF001
        inverted_template,
        key,
    )
    assert numpy.array_equal(inverted_polyprotect, protected_template)
    assert protector.get_inversion_config()["surrogate_maxiter"] == 500


def test_polyprotect_surrogate_requires_binary_output(tmp_path):
    with pytest.raises(ValueError, match="requires binarize"):
        PolyProtect(
            tmp_path,
            False,
            {
                "method": "minimize_surrogate",
                "binarize": False,
            },
        )


def test_polyprotect_hard_constraints_and_inversion(tmp_path):
    protector = PolyProtect(
        tmp_path,
        False,
        {
            "method": "minimize_hard_constraints",
            "binarize": True,
            "normalize_input": True,
            "overlap": 2,
            "nb_coef": 3,
            "coef_range": 5,
            "hard_constraint_epsilon": 1e-8,
            "hard_constraint_maxiter": 200,
            "hard_constraint_ftol": 1e-10,
        },
    )
    key = 11
    target_source = numpy.array([0.5, -0.4, 0.3, 0.8, -0.2, 0.1])
    target_source /= numpy.linalg.norm(target_source)
    raw_protected_template = protector._protect(target_source, key)  # noqa: SLF001
    protected_template = protector._postprocess_protected_template(  # noqa: SLF001
        raw_protected_template
    )
    assert not numpy.array_equal(raw_protected_template, protected_template)

    inverted_template = protector._invert_atomic(  # noqa: SLF001
        protected_template,
        key,
        target_source.copy(),
    )
    assert inverted_template is not None

    constraint_values = protector._hard_constraint_values(  # noqa: SLF001
        inverted_template,
        protected_template,
        key,
    )
    assert numpy.min(constraint_values) >= -1e-8

    inverted_template /= numpy.linalg.norm(inverted_template)
    inverted_polyprotect = protector._protect_postprocessed(  # noqa: SLF001
        inverted_template,
        key,
    )
    assert numpy.array_equal(inverted_polyprotect, protected_template)
    assert protector.get_inversion_config()["hard_constraint_maxiter"] == 200


def test_biohash_combined():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    btp_alg_dict = dict()
    btp_alg_dict["biohash"] = DummyBioHash

    template1 = Template("1", "0", numpy.random.randint(0, 2, 5))  # noqa: NPY002
    template2 = Template("2", "0", numpy.random.randint(0, 2, 5))  # noqa: NPY002

    for val in [True, False]:
        conf = {
            "type": "combined",
            "normalize_input": False,
            "normalized": True,
            "inner_type": "biohash",
            "nb_algs": 5,
            "algs_config": {"binarize": val, "system_specific": True},
            "system_specific": True,
        }

        btp_alg = CombinedBTPAlgs(
            output_dir,
            save=False,
            config=conf,
            prot_baseline_dict=btp_alg_dict,
        )

        prot_template_1 = btp_alg.protect(template1)
        prot_template_2 = btp_alg.protect(template2)

        assert numpy.array_equal(
            template1.get_template(), prot_template_1.get_template()
        )
        assert numpy.array_equal(
            template2.get_template(), prot_template_2.get_template()
        )


def test_combined_btp():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    template1 = Template("1", "0", numpy.arange(1024))
    template2 = Template("2", "0", numpy.arange(1024) + 1)

    btp_alg_dict = dict()
    btp_alg_dict["dummy"] = DummyBTP

    conf = {
        "type": "combined",
        "inner_type": "dummy",
        "normalize_input": False,
        "normalized": True,
        "nb_algs": 5,
        "algs_config": {"key": 42, "normalize_input": False, "system_specific": True},
        "system_specific": True,
    }

    btp_alg = CombinedBTPAlgs(
        output_dir,
        save=False,
        config=conf,
        prot_baseline_dict=btp_alg_dict,
    )

    prot_template_1 = btp_alg.protect(template1)
    prot_template_2 = btp_alg.protect(template2)

    assert numpy.array_equal(prot_template_1.get_template(), numpy.arange(1024) + 600)
    assert numpy.array_equal(prot_template_2.get_template(), numpy.arange(1024) + 601)
    score = btp_alg.compare(prot_template_1, prot_template_2)

    assert score == 1200
    assert btp_alg.get_alg_name() == "unnormalized_combined_5_toto_0"

    output_dir.rmdir()


def test_combined_polyprotect_system_specific_key_list(tmp_path):
    keys = [1, 2, 3, 4, 5]
    config = {
        "type": "combined",
        "inner_type": "polyprotect",
        "nb_algs": 5,
        "key": keys,
        "normalized": True,
        "system_specific": True,
        "normalize_input": True,
        "num_guesses": 5,
        "precision": 3,
        "method": "minimize_cos",
        "ks_method": "legacy",
        "algs_config": {
            "overlap": 3,
            "nb_coef": 5,
            "coef_range": 50,
            "system_specific": True,
            "normalize_input": True,
        },
    }

    btp_alg = CombinedBTPAlgs(
        tmp_path,
        save=True,
        config=config,
        prot_baseline_dict={"polyprotect": PolyProtect},
    )

    assert btp_alg.get_system_secret() == {}
    assert (
        btp_alg.get_alg_name()
        == "normalized_combined_5_normalized_polyprotect_sys_3_5_50"
    )

    template = Template("1", "0", numpy.arange(1, 12, dtype=float))
    protected_template = btp_alg.protect(template)

    expected_templates = [
        PolyProtect(
            tmp_path / "expected" / str(i),
            save=False,
            config=config["algs_config"],
            offset=i * 300,
        )
        .protect(template, key=key)
        .get_template()
        for i, key in enumerate(keys)
    ]
    expected_template = numpy.mean(numpy.vstack(expected_templates), axis=0)

    assert protected_template.get_keys() == keys
    numpy.testing.assert_allclose(
        protected_template.get_template(),
        expected_template,
    )

    saved_template_path = (
        tmp_path / btp_alg.get_alg_name() / "1" / "1_0_sys_1_2_3_4_5.npy"
    )
    ok, loaded_template = ProtectedTemplate.load("1", "0", saved_template_path)

    assert ok
    assert loaded_template.get_keys() == keys
    assert loaded_template.template_id == "0_k1_2_3_4_5"


def test_combined_binarized_polyprotect_stays_binary_and_uses_hamming(tmp_path):
    config = {
        "type": "combined",
        "inner_type": "polyprotect",
        "nb_algs": 3,
        "key": 7,
        "normalized": True,
        "system_specific": True,
        "normalize_input": False,
        "algs_config": {
            "binarize": True,
            "overlap": 2,
            "nb_coef": 4,
            "coef_range": 10,
            "system_specific": True,
            "normalize_input": False,
        },
    }
    btp_alg = CombinedBTPAlgs(
        tmp_path,
        save=False,
        config=config,
        prot_baseline_dict={"polyprotect": PolyProtect},
    )
    template = Template("1", "0", numpy.arange(1, 12, dtype=float))
    protected_template = btp_alg.protect(template)

    assert (
        btp_alg.get_alg_name()
        == "unnormalized_combined_3_unnormalized_polyprotect_sys_binary_2_4_10"
    )
    assert numpy.all(numpy.isin(protected_template.get_template(), (0, 1)))

    ref = ProtectedTemplate("1", "0", numpy.array([1, 0, 1]), [1, 2, 3])
    probe = ProtectedTemplate("2", "0", numpy.array([0, 0, 1]), [1, 2, 3])

    assert btp_alg.compare(ref, probe) == pytest.approx(-1 / 3)


def test_protected_template_save_load_integer_key(tmp_path):
    template_array = numpy.array([0.25, 0.5, 0.75])
    protected_template = ProtectedTemplate("1", "0", template_array, 42)
    template_path = tmp_path / "1_0_sys_42.npy"

    protected_template.save(template_path)
    assert template_path.exists()

    ok, loaded_template = ProtectedTemplate.load("1", "0", template_path)

    assert ok
    assert loaded_template.subject_id == protected_template.subject_id
    assert loaded_template.template_id == "0_k42"
    assert loaded_template.get_keys() == 42
    assert numpy.array_equal(loaded_template.get_template(), template_array)


def test_protected_template_save_load_multiple_keys(tmp_path):
    keys = [1, 2, 3, 4, 5]
    template_array = numpy.array([0.25, 0.5, 0.75])
    protected_template = ProtectedTemplate("1", "0", template_array, keys)
    template_path = tmp_path / "1_0_sys_1_2_3_4_5.npy"

    protected_template.save(template_path)
    assert template_path.exists()

    ok, loaded_template = ProtectedTemplate.load("1", "0", template_path)

    assert ok
    assert loaded_template.subject_id == protected_template.subject_id
    assert loaded_template.template_id == "0_k1_2_3_4_5"
    assert loaded_template.get_keys() == keys
    assert numpy.array_equal(loaded_template.get_template(), template_array)


def test_protected_template_load_with_underscores_in_template_id(tmp_path):
    keys = [642655, 332263, 1645669]
    template_array = numpy.array([0.25, 0.5, 0.75])
    template_id = "SOTERIA_Front_frame_0001"
    protected_template = ProtectedTemplate("1", template_id, template_array, keys)
    template_path = (
        tmp_path / "1_SOTERIA_Front_frame_0001_usr_642655_332263_1645669.npy"
    )

    protected_template.save(template_path)
    assert template_path.exists()

    ok, loaded_template = ProtectedTemplate.load("1", template_id, template_path)

    assert ok
    assert loaded_template.template_id == f"{template_id}_k642655_332263_1645669"
    assert loaded_template.get_keys() == keys
    assert numpy.array_equal(loaded_template.get_template(), template_array)


def test_protected_template_load_missing_key_file(tmp_path):
    # When file does not exist, load should return (False, None)
    missing_file = tmp_path / "does_not_exist_1.npy"
    ok, pt = ProtectedTemplate.load("1", "0", missing_file)
    assert ok is False
    assert pt is None


def test_baselinebtp_invert_none_template():
    class NoopBTP(BaselineBTP):
        def get_secret(self, key: int) -> dict:
            return {}

        def _protect(self, feat_vec, key):  # pragma: no cover - not used here
            return feat_vec

        def compare(self, ref, probe):  # pragma: no cover - not used here
            return 0.0

        def get_alg_name(self):  # pragma: no cover - trivial
            return "noop"

        def _invert(self, *args, **kwargs):  # pragma: no cover - not used
            raise AssertionError("Should not be called when template is None")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    btp = NoopBTP(
        output_dir,
        save=False,
        config={
            "system_specific": True,
            "key": 1,
            "num_guesses": 1,
            "precision": 1,
            "method": "root",
        },
    )

    pt = ProtectedTemplate("1", "0", None, keys=1)

    from btpbench.utils import Distribution

    dist = Distribution(
        means=numpy.zeros(1),
        mins=numpy.zeros(1),
        maxs=numpy.zeros(1),
        probs=[(numpy.zeros(1), numpy.ones(1))],
    )
    inv = btp.invert(pt, dist)

    assert isinstance(inv, Template)
    assert inv.get_template() is None

    output_dir.rmdir()


def test_baselinebtp_key_dictionary_and_name(tmp_path):
    class DictBTP(BaselineBTP):
        def get_secret(self, key: int) -> dict:
            return {}

        def _protect(self, feat_vec, key):
            return feat_vec

        def compare(self, ref, probe):
            return 0.0

        def get_alg_name(self):
            return "dictbtp"

    key_dict_path = tmp_path / "keys_select_custom.json"
    key_dict_path.write_text(json.dumps({"1": 123}))

    btp = DictBTP(
        tmp_path,
        save=False,
        config={
            "system_specific": False,
            "key_dictionary_file": str(key_dict_path),
            "normalize_input": False,
        },
    )

    assert btp.has_key_dictionary()
    assert btp.get_key_dictionary_name() == "custom"

    t = Template("1", "0", numpy.ones(4))
    prot = btp.protect(t)
    assert isinstance(prot, ProtectedTemplate)
    assert prot.get_keys() == 123


def test_baselinebtp_protect_invalid_input_raises():
    class NoopBTP(BaselineBTP):
        def get_secret(self, key: int) -> dict:
            return {}

        def _protect(self, feat_vec, key):
            return feat_vec

        def compare(self, ref, probe):
            return 0.0

        def get_alg_name(self):
            return "noop"

    btp = NoopBTP(Path(), save=False, config={"system_specific": True, "key": 1})

    t = Template("1", "0", None)
    with pytest.raises(InvalidInputSampleError):
        btp.protect(t)


def test_baselinebtp_system_specific_save_and_load(tmp_path):
    class SimpleBTP(BaselineBTP):
        def get_secret(self, key: int) -> dict:
            return {}

        def _protect(self, feat_vec, key):
            return feat_vec + key

        def compare(self, ref, probe):
            return 0.0

        def get_alg_name(self):
            return "simple"

    work_dir = tmp_path
    btp = SimpleBTP(
        work_dir,
        save=True,
        config={"system_specific": True, "key": 7, "normalize_input": False},
    )

    t = Template("1", "0", numpy.ones(3))
    prot1 = btp.protect(t)
    assert isinstance(prot1, ProtectedTemplate)
    assert numpy.array_equal(prot1.get_template(), numpy.ones(3) + 7)

    prot2 = btp.protect(t)
    assert numpy.array_equal(prot2.get_template(), prot1.get_template())


def test_baselinebtp_key_selection_usr(tmp_path):
    class SimpleBTP(BaselineBTP):
        def get_secret(self, key: int) -> dict:
            return {}

        def _protect(self, feat_vec, key):
            return feat_vec + key

        def compare(self, ref, probe):
            # Cosine-like similarity using underlying arrays
            ref_vec = ref.get_template().reshape(1, -1)
            probe_vec = probe.get_template().reshape(1, -1)
            num = numpy.sum(ref_vec * probe_vec)
            denom = numpy.linalg.norm(ref_vec) * numpy.linalg.norm(probe_vec)
            return float(num / denom)

        def get_alg_name(self):
            return "simple"

    work_dir = tmp_path
    btp = SimpleBTP(
        work_dir,
        save=True,
        config={
            "system_specific": False,
            "normalize_input": False,
            "num_guesses": 1,
            "precision": 1,
            "method": "root",
        },
    )

    # Simple 1D template so Distribution is easy to build
    t = Template("1", "0", numpy.ones(1))

    # Distribution with a single possible value 1.0
    dist = Distribution(
        means=numpy.ones(1),
        mins=numpy.ones(1),
        maxs=numpy.ones(1),
        probs=[(numpy.array([1.0]), numpy.array([1.0]))],
    )

    # compare_f uses cosine-like similarity between unprotected templates
    def compare_f(u: Template, v: Template) -> float:
        u_vec = u.get_template().reshape(1, -1)
        v_vec = v.get_template().reshape(1, -1)
        num = numpy.sum(u_vec * v_vec)
        denom = numpy.linalg.norm(u_vec) * numpy.linalg.norm(v_vec)
        return float(num / denom)

    # key_selection_usr: small num_guesses, precision; threshold high so first key accepted
    results_key = btp.key_selection_usr(
        t,
        dist,
        thresh=2.0,
        compare_f=compare_f,
    )

    prot_t_key, inv_t, score = results_key
    assert isinstance(prot_t_key, ProtectedTemplate)
    # inversion might fail and return template with None, just ensure type consistency
    assert isinstance(inv_t, Template)
    assert isinstance(score, float) or numpy.isnan(score)


def test_baselinebtp_invert_helpers_and_unknown_method():
    class SimpleBTP(BaselineBTP):
        def get_secret(self, key: int) -> dict:
            return {}

        def _protect(self, feat_vec, key):
            return feat_vec + key

        def compare(self, ref, probe):  # pragma: no cover - not used here
            return 0.0

        def get_alg_name(self):  # pragma: no cover - trivial
            return "simple"

    btp = SimpleBTP(
        Path(),
        save=False,
        config={
            "system_specific": True,
            "key": 0,
            "num_guesses": 1,
            "method": "unknown",
        },
    )

    # Set normalization on to exercise normalization branches
    btp._normalize_input = True  # noqa: SLF001

    fv = numpy.array([1.0, 0.0])
    residuals = numpy.array([0.0, 1.0])

    # _invert_alg_lsq with cosine=True and non-zero norm
    lsq_res = btp._invert_alg_lsq(fv.copy(), residuals.copy(), key=0, cosine=True)  # noqa: SLF001
    assert lsq_res.shape == fv.shape

    # _invert_alg_minimize with cosine=True (cdist path)
    min_res = btp._invert_alg_minimize(fv.copy(), residuals.copy(), key=0, cosine=True)  # noqa: SLF001
    float_val = float(min_res)
    assert isinstance(float_val, float)

    # _invert_alg_root with cosine=True and zero norm triggers penalty path
    # Use _protect to generate zero vector by overriding temporarily
    def zero_protect(_, __):
        return numpy.zeros_like(fv)

    btp._protect_backup = btp._protect  # noqa: SLF001
    btp._protect = zero_protect  # noqa: SLF001
    root_res = btp._invert_alg_root(fv.copy(), residuals.copy(), key=0, cosine=True)  # noqa: SLF001
    assert root_res.shape[0] >= fv.shape[0]
    btp._protect = btp._protect_backup  # noqa: SLF001

    # _invert success path (will likely fail and return None, but we can still
    # exercise the unknown-method branch separately)
    dist = Distribution(
        means=numpy.ones(1),
        mins=numpy.ones(1),
        maxs=numpy.ones(1),
        probs=[(numpy.array([1.0]), numpy.array([1.0]))],
    )

    with pytest.raises(ValueError):
        btp._invert(  # noqa: SLF001
            protected_template=numpy.ones(1),
            key=0,
            template_dist=dist,
        )


def test_baselinebtp_hamming_invert_helpers():
    class SimpleBTP(BaselineBTP):
        def get_secret(self, key: int) -> dict:
            return {}

        def _protect(self, feat_vec, key):
            return feat_vec

        def compare(self, ref, probe):  # pragma: no cover - not used here
            return 0.0

        def get_alg_name(self):  # pragma: no cover - trivial
            return "simple"

    btp = SimpleBTP(Path(), save=False, config={"normalize_input": False})
    feature_vector = numpy.array([0.0, 1.0])
    residuals = numpy.array([0.0, 0.0])

    # Hamming takes precedence over the default cosine distance.
    minimize_result = btp._invert_alg_minimize(  # noqa: SLF001
        feature_vector, residuals, key=0, hamming=True
    )
    assert minimize_result == pytest.approx(0.5)

    root_result = btp._invert_alg_root(  # noqa: SLF001
        feature_vector, residuals, key=0, hamming=True
    )
    assert numpy.array_equal(root_result, numpy.array([0.0, 1.0]))


def test_baselinebtp_cmaes_hamming(monkeypatch):
    import cma

    class SimpleBTP(BaselineBTP):
        def get_secret(self, key: int) -> dict:
            return {}

        def _protect(self, feat_vec, key):
            return feat_vec

        def compare(self, ref, probe):  # pragma: no cover - not used here
            return 0.0

        def get_alg_name(self):  # pragma: no cover - trivial
            return "simple"

    captured = {}

    class Result:
        fbest = 0.0

    class Optimizer:
        result = Result()

    def fake_fmin2(objective, initial_guess, sigma, options):
        captured["loss"] = objective(initial_guess)
        captured["sigma"] = sigma
        captured["options"] = options
        return numpy.array([1.0, 0.0]), Optimizer()

    monkeypatch.setattr(cma, "fmin2", fake_fmin2)

    btp = SimpleBTP(
        Path(),
        save=False,
        config={
            "method": "cmaes_hamming",
            "normalize_input": False,
            "cma_sigma": 0.25,
            "cma_maxiter": 4,
            "cma_popsize": 6,
            "cma_seed": 7,
        },
    )
    result = btp._invert_atomic(  # noqa: SLF001
        protected_template=numpy.array([0.0, 0.0]),
        key=0,
        initial_guess=numpy.array([0.0, 1.0]),
    )

    assert numpy.array_equal(result, numpy.array([1.0, 0.0]))
    assert captured == {
        "loss": pytest.approx(0.5),
        "sigma": 0.25,
        "options": {
            "ftarget": 0.0,
            "maxiter": 4,
            "popsize": 6,
            "seed": 7,
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
        },
    }
    assert btp.get_inversion_config()["method"] == "cmaes_hamming"


def test_baselinebtp_cmaes_defaults():
    class SimpleBTP(BaselineBTP):
        def get_secret(self, key: int) -> dict:
            return {}

        def _protect(self, feat_vec, key):
            return feat_vec

        def compare(self, ref, probe):  # pragma: no cover - not used here
            return 0.0

        def get_alg_name(self):  # pragma: no cover - trivial
            return "simple"

    btp = SimpleBTP(
        Path(),
        save=False,
        config={"method": "cmaes_hamming"},
    )

    assert btp.get_inversion_config() == {
        "method": "cmaes_hamming",
        "num_guesses": 1,
        "precision": 3,
        "cma_sigma": 0.1,
        "cma_maxiter": 100,
        "cma_popsize": None,
        "cma_seed": 42,
    }
    assert btp._get_cmaes_options() == {  # noqa: SLF001
        "ftarget": 0.0,
        "maxiter": 100,
        "seed": 42,
        "verbose": -9,
        "verb_disp": 0,
        "verb_log": 0,
    }
