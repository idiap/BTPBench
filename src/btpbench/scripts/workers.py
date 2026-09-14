# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import logging

from typing import Any

import numpy

from btpbench.baselines import Template
from btpbench.btps import ProtectedTemplate
from btpbench.dataloader import Sample
from btpbench.exceptions import InvalidInputSampleError
from btpbench.utils import derive_seed


def _init_logging():
    # re-init logging to avoid deadlock in worker
    logging.shutdown()
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


logger = logging.getLogger(__name__)


class BaseWorker:
    """Base worker class."""

    _instance = None

    def __init__(
        self,
        alg_cls,
        alg_args,
        compliant: bool,
        raw_elements: list[Sample] | list[Template],
        references: list[Template] | list[ProtectedTemplate],
        probes: list[Template] | list[ProtectedTemplate],
        extra_args: dict[str, Any] | None = None,
    ):
        _init_logging()
        self.alg = alg_cls(*alg_args)
        self.compliant = compliant
        self.raw_elements = raw_elements
        self.probes = probes
        self.references = references

    @classmethod
    def init(
        cls,
        alg_cls,
        alg_args,
        compliant: bool,
        raw_elements: list[Sample] | list[Template],
        references: list[Template] | list[ProtectedTemplate],
        probes: list[Template] | list[ProtectedTemplate],
        extra_args: dict[str, Any] | None = None,
    ):
        """Initialize object, called once per worker process."""
        cls._instance = cls(
            alg_cls, alg_args, compliant, raw_elements, references, probes, extra_args
        )

    @classmethod
    def compare_verification(
        cls, pair: tuple[int, int]
    ) -> tuple[float, Template, Template]:
        """Perform a within-set verification comparison.

        The difference with the compare method is that here the two idx come from the same list.
        """
        ref_idx, probe_idx = pair
        if cls._instance.references is None:
            raise RuntimeError("No templates")

        ref_template = cls._instance.references[ref_idx]
        probe_template = cls._instance.references[probe_idx]

        return cls._instance.compare_template(ref_template, probe_template)

    @classmethod
    def compare(cls, pair: tuple[int, int]) -> tuple[float, Template, Template]:
        """Perform worker comparison."""
        ref_idx, probe_idx = pair
        if cls._instance.references is None or cls._instance.probes is None:
            raise RuntimeError("No templates")

        ref_template = cls._instance.references[ref_idx]
        probe_template = cls._instance.probes[probe_idx]

        return cls._instance.compare_template(ref_template, probe_template)

    @classmethod
    def compare_template(
        cls, ref_template: Template, probe_template: Template
    ) -> tuple[float, Template, Template]:
        score = numpy.nan

        if (
            ref_template.get_template() is not None
            and probe_template.get_template() is not None
        ):
            score = cls._instance.alg.compare(ref_template, probe_template)

        return score, ref_template, probe_template


class FRWorker(BaseWorker):
    """
    Worker for the FR part of the evaluation.

    Per-process singleton that holds:

    * one FR algorithm instance.
    * A list of templates or samples
    """

    def __init__(
        self,
        fr_alg_cls,
        fr_alg_args,
        compliant: bool,
        samples: list[Sample] | list[Template],
        ref_templates: list[Template] | list[ProtectedTemplate],
        probe_templates: list[Template] | list[ProtectedTemplate],
        extra_args: dict[str, Any] | None = None,
    ):
        super().__init__(
            fr_alg_cls,
            fr_alg_args,
            compliant,
            samples,
            ref_templates,
            probe_templates,
            extra_args,
        )

    @classmethod
    def init(
        cls,
        fr_alg_cls,
        fr_alg_args,
        compliant: bool,
        samples: list[Sample] | list[Template],
        ref_templates: list[Template] | list[ProtectedTemplate],
        probe_templates: list[Template] | list[ProtectedTemplate],
        extra_args: dict[str, Any] | None = None,
    ):
        """Initialize object, called once per worker process."""
        super().init(
            fr_alg_cls, fr_alg_args, compliant, samples, ref_templates, probe_templates
        )

    @classmethod
    def feature_extraction(cls, idx: int) -> Template:
        """Perform feature extraction for samples."""
        if cls._instance.raw_elements is None:
            raise RuntimeError("No samples")

        sample = cls._instance.raw_elements[idx]
        try:
            template = cls._instance.alg.feature_extraction(
                cls._instance.alg.preprocessor(sample)
            )
        except InvalidInputSampleError as e:
            template = Template(
                sample.subject_id, sample.template_id, None, sample.metadata
            )
            if cls._instance.compliant:
                if type(sample) is Sample:
                    logger.warning(f"Invalid sample {sample.path}")
            else:
                raise e
        return template


class BTPWorker(BaseWorker):
    """
    Worker for BTP part of the evaluation.

    Per-process singleton that holds:

    * one BTP-algorithm instance
    * the compliance flag
    * the lists of unprotected probes & protected references templates
    """

    def __init__(
        self,
        btp_alg_cls,
        btp_alg_args,
        compliant: bool,
        templates: list[Sample] | list[Template],
        protected_ref_templates: list[Template] | list[ProtectedTemplate],
        protected_probe_templates: list[Template] | list[ProtectedTemplate],
        extra_args: dict[str, Any] | None = None,
    ):
        extra_args = {} if extra_args is None else extra_args
        super().__init__(
            btp_alg_cls,
            btp_alg_args,
            compliant,
            templates,
            protected_ref_templates,
            protected_probe_templates,
            extra_args,
        )

        self.ref_dist = extra_args.get("ref_distribution", None)
        self.key_dictionary = extra_args.get("key_dictionary", None)
        self.thresh = extra_args.get("thresh", 0.0)
        self.compare_f = extra_args.get("compare_f", None)
        self.n_elements = extra_args.get("n_elements", 60)
        self.attack_seed = extra_args.get("attack_seed")
        self.attack_trial = extra_args.get("attack_trial", 0)
        self.key_sampling_seed = extra_args.get("key_sampling_seed")

    @classmethod
    def init(
        cls,
        btp_alg_cls,
        btp_alg_args,
        compliant: bool,
        templates: list[Sample] | list[Template],
        protected_ref_templates: list[Template] | list[ProtectedTemplate],
        protected_probe_templates: list[Template] | list[ProtectedTemplate],
        extra_args: dict[str, Any] | None = None,
    ):
        """Initialize object, called once per worker process."""
        super().init(
            btp_alg_cls,
            btp_alg_args,
            compliant,
            templates,
            protected_ref_templates,
            protected_probe_templates,
            extra_args,
        )

    @classmethod
    def key_selection_usr(cls, idx: int) -> tuple[ProtectedTemplate, Template, float]:
        """Invert a protected template."""
        inst = cls._instance
        assert isinstance(inst, BTPWorker)
        tpl = inst.raw_elements[idx]
        seed = None
        if inst.key_sampling_seed is not None:
            seed = derive_seed(
                inst.key_sampling_seed,
                "key-selection",
                tpl.subject_id,
                tpl.template_id,
            )

        # mypy doesn't understand that inst is of type BTPWorker
        # then the following ignores...
        return inst.alg.key_selection_usr(
            tpl,
            inst.ref_dist,  # type: ignore
            inst.thresh,  # type: ignore
            inst.compare_f,  # type: ignore
            seed,
        )

    @classmethod
    def invert(cls, idx: int) -> tuple[Template, int]:
        """Invert a protected template."""
        inst = cls._instance
        tpl = inst.probes[idx]

        # mypy doesn't understand that inst is of type BTPWorker
        # then the following ignores...
        return (
            inst.alg.invert(
                tpl,
                inst.ref_dist,  # type: ignore
                False,  # type: ignore
            ),
            idx,
        )

    @classmethod
    def invert_no_mem(cls, idx: int) -> tuple[Template, int]:
        """Invert a protected template."""
        inst = cls._instance
        assert isinstance(inst, BTPWorker)
        tpl = inst.probes[idx]
        assert isinstance(tpl, ProtectedTemplate)
        seed = None
        if inst.attack_seed is not None:
            seed = derive_seed(
                inst.attack_seed,
                "inversion",
                inst.attack_trial,
                tpl.subject_id,
                tpl.template_id,
                tpl.get_keys(),
            )

        # mypy doesn't understand that inst is of type BTPWorker
        # then the following ignores...
        return (
            inst.alg.invert(
                tpl,
                inst.ref_dist,  # type: ignore
                True,  # type: ignore
                seed,
            ),
            idx,
        )

    @classmethod
    def protect(cls, idx: int) -> ProtectedTemplate:
        """Protect one template by index."""
        inst = cls._instance
        assert isinstance(inst, BTPWorker)
        tpl = inst.raw_elements[idx]
        try:
            key = None
            # If possible retrieve key from dictionary
            if inst.key_dictionary is not None:
                if tpl.subject_id not in inst.key_dictionary:
                    raise ValueError(
                        f"Subject id {tpl.subject_id} not in key dictionary."
                    )
                key = inst.key_dictionary[tpl.subject_id]

            prot = inst.alg.protect(tpl, key)
        except InvalidInputSampleError:
            if not inst.compliant:
                raise
            # produce a “dummy” ProtectedTemplate on error
            prot = ProtectedTemplate(tpl.subject_id, tpl.template_id, None, None)
            prot.metadata = tpl.metadata
        return prot
