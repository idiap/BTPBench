# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

"""Single CLI entry point for BTPBench.

Usage examples::

    btpbench identification pipeline -s sys.yaml -e exp.yaml
    btpbench identification metrics -f scores.csv -o out.csv
    btpbench identification plots -f scores.csv -o out.png

    btpbench verification pipeline -s sys.yaml -e exp.yaml
    btpbench verification metrics -s scores.csv -o out.csv
    btpbench verification plots -f scores.csv -o out.png

    btpbench irreversibility pipeline -s sys.yaml -e exp.yaml
    btpbench irreversibility metrics -i irr.csv -s id.csv -o out.csv
    btpbench irreversibility plots -i irr.csv -s id.csv -o out.png

    btpbench unlinkability pipeline -s sys.yaml -e exp.yaml -k keys.json -b bucket

    btpbench keyselection pipeline_user -s sys.yaml -e exp.yaml -v scores.csv

    btpbench diversity pipeline -s sys.yaml -e exp.yaml -k keys.json -b bucket
    btpbench diversity summary -i experiment_dir -o diversity-summary.csv
    btpbench diversity unlinkability-summary -i experiment_dir -o summary.csv
    btpbench diversity plots -f diversity-result.csv -o violin.png

    btpbench plots distribution -s sys.yaml -e exp.yaml -o out_dir
    btpbench plots polyprotect_coeff -i key_explorer.json -o out.png
    btpbench plots histogram -u unprotected.csv -o out.png
"""

import click


@click.group()
def cli():
    """BTPBench — biometric template protection evaluation."""


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------
@cli.group()
def identification():
    """Identification experiment commands."""


def _register_identification():
    from btpbench.scripts.identification.metrics import metrics
    from btpbench.scripts.identification.pipeline import pipeline
    from btpbench.scripts.identification.plots import main as plots

    identification.add_command(pipeline, "pipeline")
    identification.add_command(metrics, "metrics")
    identification.add_command(plots, "plots")


_register_identification()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
@cli.group()
def verification():
    """Verification experiment commands."""


def _register_verification():
    from btpbench.scripts.verification.metrics import metrics
    from btpbench.scripts.verification.pipeline import pipeline
    from btpbench.scripts.verification.plots import main as plots

    verification.add_command(pipeline, "pipeline")
    verification.add_command(metrics, "metrics")
    verification.add_command(plots, "plots")


_register_verification()


# ---------------------------------------------------------------------------
# Irreversibility
# ---------------------------------------------------------------------------
@cli.group()
def irreversibility():
    """Irreversibility experiment commands."""


def _register_irreversibility():
    from btpbench.scripts.irreversibility.metrics import main as metrics
    from btpbench.scripts.irreversibility.pipeline import pipeline
    from btpbench.scripts.irreversibility.plots import main as plots

    irreversibility.add_command(pipeline, "pipeline")
    irreversibility.add_command(metrics, "metrics")
    irreversibility.add_command(plots, "plots")


_register_irreversibility()


# ---------------------------------------------------------------------------
# Key Selection
# ---------------------------------------------------------------------------
@cli.group()
def keyselection():
    """Key selection commands."""


def _register_keyselection_usr():
    from btpbench.scripts.keyselection.pipeline_user import pipeline

    keyselection.add_command(pipeline, "pipeline_user")


def _register_validate_sys():
    from btpbench.scripts.keyselection.validate_sys import pipeline

    keyselection.add_command(pipeline, "validate_sys")


def _register_validate_sys_verification():
    from btpbench.scripts.keyselection.validate_sys_verification import (
        pipeline,
    )

    keyselection.add_command(pipeline, "validate_sys_verification")


def _register_key_explorer():
    from btpbench.scripts.keyselection.key_explorer import key_explorer

    keyselection.add_command(key_explorer, "key_explorer")


_register_keyselection_usr()
_register_validate_sys()
_register_validate_sys_verification()
_register_key_explorer()


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------
@cli.group()
def diversity():
    """Diversity experiment commands."""


def _register_diversity():
    from btpbench.scripts.diversity.pipeline import pipeline
    from btpbench.scripts.diversity.plots import plots
    from btpbench.scripts.diversity.summary import summary
    from btpbench.scripts.diversity.unlinkability_summary import (
        unlinkability_summary,
    )

    diversity.add_command(pipeline, "pipeline")
    diversity.add_command(plots, "plots")
    diversity.add_command(summary, "summary")
    diversity.add_command(unlinkability_summary, "unlinkability-summary")


_register_diversity()


# ---------------------------------------------------------------------------
# Unlinkability
# ---------------------------------------------------------------------------
@cli.group()
def unlinkability():
    """Unlinkability experiment commands."""


def _register_unlinkability():
    from btpbench.scripts.unlinkability.pipeline import pipeline
    from btpbench.scripts.unlinkability.plots import main as plots

    unlinkability.add_command(pipeline, "pipeline")
    unlinkability.add_command(plots, "plots")


_register_unlinkability()


# ---------------------------------------------------------------------------
# Plots (standalone plotting utilities)
# ---------------------------------------------------------------------------
@cli.group()
def plots():
    """Standalone plotting utilities."""


def _register_plots():
    from btpbench.scripts.plots.distribution import pipeline as distribution
    from btpbench.scripts.plots.histogram import main as histogram
    from btpbench.scripts.plots.polyprotect_coeff import (
        main as polyprotect_coeff,
    )

    plots.add_command(distribution, "distribution")
    plots.add_command(polyprotect_coeff, "polyprotect_coeff")
    plots.add_command(histogram, "histogram")


_register_plots()
