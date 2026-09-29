# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

"""Unlinkability scores with subject-specific, per-sample online keys."""

import logging

from pathlib import Path

import click
import numpy

from btpbench.scorewriter import CSVScoreWriter
from btpbench.scripts.online_keyselection import (
    common_options,
    completed_outputs,
    load_templates,
    prepare_run,
    protect_online,
    write_metadata,
)
from btpbench.scripts.unlinkability.pipeline import (
    _flatten_grouped_templates,
    _mated_pairs,
    _non_mated_pairs,
    _sample_non_mated_templates,
    _write_protected_scores,
)

logger = logging.getLogger(__name__)


@click.command()
@common_options
@click.option(
    "--non-mated-samples-per-subject",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Protected templates per subject used for non-mated comparisons.",
)
@click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=42,
    show_default=True,
    help="Seed used only to sample templates for non-mated comparisons.",
)
def online_pipeline(
    system_config_file: Path,
    exp_config_file: Path,
    keys_file: Path | None,
    verification_file: Path | None,
    ks_fmr: float,
    samples_per_subject: int,
    n_subjects: int,
    random_keys: bool,
    output_dir: Path | None,
    override: bool,
    non_mated_samples_per_subject: int,
    seed: int,
) -> None:
    """Evaluate unlinkability with a distinct online key for every sample."""
    if non_mated_samples_per_subject > samples_per_subject:
        raise click.BadParameter(
            "Cannot exceed --samples-per-subject.",
            param_hint="--non-mated-samples-per-subject",
        )
    if n_subjects == 1:
        raise click.BadParameter(
            "Unlinkability needs at least two subjects.",
            param_hint="--n-subjects",
        )

    run = prepare_run(
        system_config_file,
        exp_config_file,
        keys_file,
        verification_file,
        ks_fmr,
        samples_per_subject,
        n_subjects,
        output_dir,
        random_keys,
    )
    stem = (
        f"online-unlinkability-{run.stem(samples_per_subject, n_subjects)}"
        f"-nm{non_mated_samples_per_subject}-seed{seed}"
    )
    mated_file = run.output_dir / f"{stem}-mated.csv"
    non_mated_file = run.output_dir / f"{stem}-non-mated.csv"
    metadata_file = run.output_dir / f"{stem}.json"
    expected_metadata = {
        **run.metadata(samples_per_subject, n_subjects),
        "kind": "online_unlinkability",
        "non_mated_seed": seed,
        "non_mated_samples_per_subject": non_mated_samples_per_subject,
    }
    if completed_outputs(
        [mated_file, non_mated_file, metadata_file],
        override,
        expected_metadata,
    ):
        return

    by_subject, distribution, compare = load_templates(
        run,
        samples_per_subject,
        n_subjects,
    )
    if len(by_subject) < 2:
        raise click.ClickException("Unlinkability needs at least two subjects.")
    protected, audit = protect_online(run, by_subject, distribution, compare)
    non_mated = _sample_non_mated_templates(
        protected,
        non_mated_samples_per_subject,
        numpy.random.default_rng(seed),
    )
    metadata_names = run.dataset.metadata_names(verification=False)

    staged_mated_file = mated_file.with_name(mated_file.name + ".replacement")
    staged_non_mated_file = non_mated_file.with_name(
        non_mated_file.name + ".replacement"
    )
    staged_outputs = (staged_mated_file, staged_non_mated_file)
    try:
        for grouped, pairs_function, path in (
            (protected, _mated_pairs, staged_mated_file),
            (non_mated, _non_mated_pairs, staged_non_mated_file),
        ):
            templates, indices = _flatten_grouped_templates(grouped)
            writer = CSVScoreWriter(path, metadata_names)
            completed = False
            try:
                _write_protected_scores(
                    templates,
                    pairs_function(indices),
                    run.alg_cls,
                    run.alg_args,
                    run.compliant,
                    run.n_workers,
                    100,
                    writer,
                )
                completed = True
            finally:
                writer.close()
                if not completed:
                    path.unlink(missing_ok=True)
    except BaseException:
        for path in staged_outputs:
            path.unlink(missing_ok=True)
        raise

    # Invalidate the previous completion marker before publishing either score
    # file. A crash during the replacements is therefore detected as incomplete.
    metadata_file.unlink(missing_ok=True)
    staged_mated_file.replace(mated_file)
    staged_non_mated_file.replace(non_mated_file)
    logger.info("Saved %s", mated_file)
    logger.info("Saved %s", non_mated_file)

    write_metadata(
        metadata_file,
        {
            **expected_metadata,
            "n_subjects": len(protected),
            "keys_file": None if run.keys_file is None else str(run.keys_file),
            "verification_file": (
                None if run.verification_file is None else str(run.verification_file)
            ),
            "mated_file": mated_file.name,
            "non_mated_file": non_mated_file.name,
        },
        audit,
    )


if __name__ == "__main__":
    online_pipeline()
