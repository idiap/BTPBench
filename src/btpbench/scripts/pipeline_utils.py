# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import json
import logging
import random

from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import combinations, islice, product
from pathlib import Path
from typing import Any

import yaml

from btpbench.baselines import Template
from btpbench.btps import ProtectedTemplate
from btpbench.dataloader import Dataset, cv_loader, cv_loader_video
from btpbench.sample import Sample
from btpbench.scorewriter import CSVScoreWriter
from btpbench.scripts.workers import BTPWorker, FRWorker
from btpbench.utils import Distribution

logger = logging.getLogger("pipeline_utils")

INVERSION_METADATA_NAMES = (
    "attack_trial",
    "attack_solved",
    "key",
    "key_system",
    "key_source",
    "key_scope",
)


def inversion_metadata_names(metadata_names: list[str]) -> list[str]:
    """Add the common inversion-evaluation fields without duplicates."""
    return list(dict.fromkeys([*metadata_names, *INVERSION_METADATA_NAMES]))


def inversion_evaluation_parameters(
    exp_config: dict[str, Any],
) -> tuple[int, int | None]:
    """Return validated attack-trial count and reproducibility seed."""
    n_attack_trials = int(
        exp_config.get("n_attack_trials", exp_config.get("n_validations", 10))
    )
    if n_attack_trials < 1:
        raise ValueError("n_attack_trials must be at least 1")
    attack_seed = exp_config.get("attack_seed", 42)
    if attack_seed is not None:
        attack_seed = int(attack_seed)
    return n_attack_trials, attack_seed


def key_sampling_seed(exp_config: dict[str, Any]) -> int | None:
    """Return the normalized key-sampling seed."""
    seed = exp_config.get("key_sampling_seed", 42)
    return None if seed is None else int(seed)


def get_optimal_chunksize(
    total_items: int,
    n_workers: int,
    min_chunks_per_worker: int = 4,
) -> int:
    """Return a chunksize that provides enough work units for each worker."""
    if total_items <= n_workers:
        return 1

    ideal_total_chunks = n_workers * min_chunks_per_worker
    return max(1, total_items // ideal_total_chunks)


def get_n_keys_per_system(protected_baseline_config: dict[str, Any]) -> int:
    """Return how many key values are needed for one protected system."""
    if protected_baseline_config["type"] == "combined":
        return int(protected_baseline_config["nb_algs"])
    return 1


def sample_key_systems(
    key_pool: list[Any] | range,
    n_keys: int,
    n_keys_per_system: int,
    rng: random.Random | None = None,
) -> list[Any]:
    """Sample scalar or multi-key systems from a shared key pool."""
    if n_keys < 1:
        raise ValueError("n_keys must be at least 1")
    if n_keys_per_system < 1:
        raise ValueError("n_keys_per_system must be at least 1")
    sample = random.sample if rng is None else rng.sample
    if n_keys_per_system == 1:
        return sample(key_pool, n_keys)
    return [sample(key_pool, k=n_keys_per_system) for _ in range(n_keys)]


def key_bucket_tag(keys_bucket: str) -> str:
    """Normalize a key-explorer bucket name for use in output filenames."""
    return keys_bucket.replace(".", "d").replace("-", "m")


def load_key_pool(
    exp_config: dict[str, Any],
    default_mode: str = "keys",
) -> tuple[list[Any] | range, str, str]:
    """Load a key pool and return it with its source label and filename tag."""
    sampling_mode = exp_config.get("sampling_mode", default_mode)
    if sampling_mode == "keys":
        keys_file = Path(exp_config["keys_file"])
        keys_bucket = str(exp_config["keys_bucket"])
        keys_dict = json.loads(keys_file.read_text())
        try:
            key_pool = list(keys_dict["keys"][keys_bucket].values())
        except KeyError as exc:
            raise ValueError(
                f"Key bucket {keys_bucket!r} does not exist in {keys_file}"
            ) from exc
        bucket_tag = key_bucket_tag(keys_bucket)
        return key_pool, f"bucket:{keys_bucket}", f"keys{bucket_tag}"
    if sampling_mode == "distribution":
        return range(1000000), "random-distribution", "distribution"
    raise ValueError(f"Invalid sampling mode: {sampling_mode}")


def sample_user_key_dictionaries(
    subject_ids: list[Any],
    key_pool: list[Any] | range,
    n_key_systems: int,
    n_keys_per_subject: int,
    rng: random.Random,
) -> list[dict[Any, Any]]:
    """Sample one user-specific key assignment per requested key system."""
    assignments = []
    for _ in range(n_key_systems):
        keys = sample_key_systems(
            key_pool,
            len(subject_ids),
            n_keys_per_subject,
            rng,
        )
        assignments.append(dict(zip(subject_ids, keys)))
    return assignments


def iter_protocol_splits(
    dataset: Dataset,
    protocols: str | list[str],
    splits: str | list[str],
    database_name: str,
) -> Iterator[tuple[str, str]]:
    """Yield valid protocol/split selections without mutating the inputs."""
    available_protocols = dataset.protocols()
    selected_protocols = (
        available_protocols
        if protocols == "all"
        else [protocols]
        if isinstance(protocols, str)
        else protocols
    )

    for protocol in selected_protocols:
        if protocol not in available_protocols:
            logger.warning(f"No protocol `{protocol}` for `{database_name}`")
            continue

        available_splits = dataset.protocol_splits(protocol)
        selected_splits = (
            available_splits
            if splits == "all"
            else [splits]
            if isinstance(splits, str)
            else splits
        )
        for split in selected_splits:
            if split not in available_splits:
                logger.warning(
                    f"No split `{split}` for protocol `{protocol}` "
                    f"for dataset `{database_name}`"
                )
                continue
            yield protocol, split


def load_config(system_conf_path: Path, exp_conf_path: Path) -> tuple[dict, dict]:
    """Load system and experiment YAML configs."""
    with system_conf_path.open() as f:
        system_conf = yaml.safe_load(f)
    with exp_conf_path.open() as f:
        exp_conf = yaml.safe_load(f)
    return system_conf, exp_conf


def resolve_output_dir(exp_config: dict, output_dir: Path | None) -> Path:
    """Resolve an output override or configured path and ensure it exists."""
    resolved = output_dir or Path(exp_config["output_dir"])
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def setup_logger(level: int = logging.INFO) -> None:
    """Configure root logger once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def setup_logger_from_verbosity(verbosity: int) -> None:
    """Configure logging from the CLI verbosity scale from zero to four."""
    levels = (
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG,
    )
    level = levels[verbosity] if 0 <= verbosity < len(levels) else logging.INFO
    setup_logger(level)


def validate_detector(detector: str) -> None:
    """Ensure detector is one of the supported backends."""
    if detector not in ("mtcnn", "mediapipe"):
        raise ValueError(f"Unknown detector: {detector!r}")


def check_dir(output_dir: Path, exp_config: dict):
    """Check if the output directory is correct."""

    # If we run a new experiment with the same output folder
    if (output_dir / "exp_info.yaml").exists():
        logger.warning("You are about to write in an existing experiment folder!")
        with (output_dir / "exp_info.yaml").open("r") as f:
            try:
                old_exp_config = yaml.safe_load(f)
            except yaml.scanner.ScannerError:
                logger.warning("Failed to parse existing exp_info.yaml, overwriting.")
                old_exp_config = None

            # Check that database is the same
            if (
                old_exp_config is not None
                and old_exp_config["database"] != exp_config["database"]
            ):
                raise RuntimeError(
                    f"Experiment folder is for database `{old_exp_config['database']}`"
                    f"but your requested database is `{exp_config['database']}`"
                )

    with (output_dir / "exp_info.yaml").open("w") as f:
        yaml.dump(exp_config, f, default_flow_style=False)


def load_common_parameters(system_config: dict, exp_config: dict):
    """Load common parameters from config files."""

    baseline = exp_config["bio_alg"]
    if baseline not in system_config["baselines"]:
        raise RuntimeError(f"Unknown baseline: {baseline}")

    has_protected_part = "btps" in exp_config
    save = exp_config["save"]
    compliant = exp_config["compliant"]
    detector = exp_config["detector"]
    n_worker = exp_config["num_processes"]

    return baseline, has_protected_part, save, compliant, detector, n_worker


def create_dataset(system_conf: dict, exp_conf: dict) -> tuple[str, Dataset]:
    """Instantiate the Dataset based on the 'database' entry in the configs."""

    db_name = exp_conf["database"]
    if db_name not in system_conf["databases"]:
        raise ValueError(f"Unknown database: {db_name!r}")
    db = system_conf["databases"][db_name]
    proto_dir = Path(db["proto_dir"])
    dataset_dir = Path(db["dataset_dir"])
    verification_samples_file = Path(db["verification_samples"])
    unlink_samples_file = Path(db["unlink_samples"])
    ext = db.get("extension") or ""
    load_f = cv_loader_video if db["video"] else cv_loader
    return db_name, Dataset(
        proto_dir,
        dataset_dir,
        verification_samples_file,
        unlink_samples_file,
        ext,
        exp_conf["compliant"],
        load_f,
    )


def chunked_combinations(arr: list[Any], chunk_size: int):
    """Yield successive lists of combinations(arr,2), each of size ≤ chunk_size."""

    it = combinations(arr, 2)
    while True:
        batch = list(islice(it, chunk_size))
        if not batch:
            return
        yield batch


def threaded_feature_extraction(
    samples: list[Sample],
    bio_alg_cls: type,
    bio_alg_args: tuple,
    compliant: bool,
    n_worker: int,
    batch_size: int,
) -> list[Template]:
    """Extract features in parallel using threads.

    As most of our FR models are PyTorch based. We need
    to stay away from the ProcessPoolExecutor. It's causing
    a deadlock preventing the worker to work correctly...
    """
    samples_indexes = list(range(len(samples)))
    with ThreadPoolExecutor(
        max_workers=n_worker,
        initializer=FRWorker.init,
        initargs=(
            bio_alg_cls,
            bio_alg_args,
            compliant,
            samples,
            None,
            None,
        ),
    ) as exe:
        return list(
            exe.map(FRWorker.feature_extraction, samples_indexes, chunksize=batch_size)
        )


def threaded_unprotected_id_matching(
    ref_templates: list[Template],
    probes_templates: list[Template],
    metadata_names: list[str],
    bio_alg_cls: type,
    bio_alg_args: tuple,
    compliant: bool,
    n_worker: int,
    batch_size: int,
    unprotected_scores_file_path: Path,
    override: bool,
) -> None:
    """Perform identification matching in parallel using threads for unprotected templates."""

    comparison_pairs = list(
        product(range(len(ref_templates)), range(len(probes_templates)))
    )

    if not unprotected_scores_file_path.exists() or override:
        # Unprotected score csv file
        unprotected_csv = CSVScoreWriter(unprotected_scores_file_path, metadata_names)

        with ThreadPoolExecutor(
            max_workers=n_worker,
            initializer=FRWorker.init,
            initargs=(
                bio_alg_cls,
                bio_alg_args,
                compliant,
                None,
                ref_templates,
                probes_templates,
            ),
        ) as exe:
            for score, t1, t2 in exe.map(
                FRWorker.compare, comparison_pairs, chunksize=batch_size
            ):
                unprotected_csv.write_score(score, t1, t2)

        unprotected_csv.close()
    else:
        logger.warning(
            f"Score file `{unprotected_scores_file_path}` exists! Skipping comparison!"
        )


def threaded_unprotected_verification_matching(
    unprotected_templates: list[Template],
    bio_alg_cls: type,
    bio_alg_args: tuple,
    compliant: bool,
    n_worker: int,
    batch_size: int,
    unprotected_csv: CSVScoreWriter,
) -> None:
    """Match unprotected templates for verification using worker threads."""

    with ThreadPoolExecutor(
        max_workers=n_worker,
        initializer=FRWorker.init,
        initargs=(
            bio_alg_cls,
            bio_alg_args,
            compliant,
            None,
            unprotected_templates,
            None,
        ),
    ) as exe:
        # stream through chunks of pair-batches
        for pair_batch in chunked_combinations(
            list(range(len(unprotected_templates))), batch_size * n_worker
        ):
            # map over just this batch
            for score, t1, t2 in exe.map(
                FRWorker.compare_verification, pair_batch, chunksize=len(pair_batch)
            ):
                unprotected_csv.write_score(score, t1, t2)


def processed_protection(
    templates: list[Template],
    btp_alg_cls: type,
    btp_alg_args: tuple,
    compliant: bool,
    n_worker: int,
    batch_size: int,
    extra_args: dict | None = None,
) -> list[ProtectedTemplate]:
    """Protect templates in parallel using processes."""
    extra_args = {} if extra_args is None else extra_args

    with ProcessPoolExecutor(
        max_workers=n_worker,
        initializer=BTPWorker.init,
        initargs=(
            btp_alg_cls,
            btp_alg_args,
            compliant,
            templates,
            None,
            None,
            extra_args,
        ),
    ) as exe:
        return list(
            exe.map(
                BTPWorker.protect, list(range(len(templates))), chunksize=batch_size
            )
        )


def processed_protected_id_matching(
    prot_ref_templates: list[ProtectedTemplate],
    prot_probe_templates: list[ProtectedTemplate],
    btp_alg_cls: type,
    btp_alg_args: tuple,
    compliant: bool,
    n_worker: int,
    batch_size: int,
    protected_csv: CSVScoreWriter,
) -> None:
    """Perform identification matching on protected templates in parallel using processes."""

    comparison_pairs = list(
        product(range(len(prot_ref_templates)), range(len(prot_probe_templates)))
    )
    with ProcessPoolExecutor(
        max_workers=n_worker,
        initializer=BTPWorker.init,
        initargs=(
            btp_alg_cls,
            btp_alg_args,
            compliant,
            None,
            prot_ref_templates,
            prot_probe_templates,
        ),
    ) as exe:
        for score, t1, t2 in exe.map(
            BTPWorker.compare, comparison_pairs, chunksize=batch_size
        ):
            protected_csv.write_score(score, t1, t2)


def processed_protected_verification_matching(
    prot_templates: list[ProtectedTemplate],
    btp_alg_cls: type,
    btp_alg_args: tuple,
    compliant: bool,
    n_worker: int,
    batch_size: int,
    protected_csv: CSVScoreWriter,
) -> None:
    """Match protected templates for verification using worker processes."""
    with ProcessPoolExecutor(
        max_workers=n_worker,
        initializer=BTPWorker.init,
        initargs=(
            btp_alg_cls,
            btp_alg_args,
            compliant,
            None,
            prot_templates,
            None,
        ),
    ) as exe:
        for pair_batch in chunked_combinations(
            list(range(len(prot_templates))), batch_size * n_worker
        ):
            chunksize = get_optimal_chunksize(len(pair_batch), n_worker)
            # map over just this small batch
            for score, t1, t2 in exe.map(
                BTPWorker.compare_verification,
                pair_batch,
                chunksize=chunksize,
            ):
                protected_csv.write_score(score, t1, t2)


def build_btp_alg(
    protected_baseline_config: dict,
    protected_baseline_dict: dict,
    output_dir: Path,
    baseline_label: str,
    save: bool,
) -> tuple[Any, type, tuple, Path]:
    """Instantiate the BTP algorithm based on the protected baseline config."""

    btp_alg_extra_args: list[Any] = list()

    protected_baseline = protected_baseline_config["type"]
    if protected_baseline not in protected_baseline_dict:
        logger.error(f"No protected baseline: {protected_baseline}")
        return None, None, None, None
    btp_alg_cls = protected_baseline_dict[protected_baseline]

    # If the alg is combined, need to add an extra argument
    if protected_baseline == "combined":
        btp_alg_extra_args.append(protected_baseline_dict)

    btp_alg_args = (
        output_dir / baseline_label / protected_baseline,
        save,
        protected_baseline_config,
    ) + tuple(btp_alg_extra_args)

    btp_alg = btp_alg_cls(*btp_alg_args)

    score_dir = output_dir
    # If a key dictionary is used, create a subfolder with its name
    if btp_alg.has_key_dictionary():
        score_dir = output_dir / btp_alg.get_key_dictionary_name()
        score_dir.mkdir(parents=True, exist_ok=True)

    return btp_alg, btp_alg_cls, btp_alg_args, score_dir


def processed_inversion_no_mem(
    prot_templates: list[ProtectedTemplate],
    btp_alg_cls: type,
    btp_alg_args: tuple,
    compliant: bool,
    n_worker: int,
    ref_distribution: Distribution,
    protect_chunksize: int,
    attack_seed: int | None = None,
    attack_trial: int = 0,
) -> list[tuple[Template, int]]:
    """Invert protected templates in worker processes without saving results."""
    with ProcessPoolExecutor(
        max_workers=n_worker,
        initializer=BTPWorker.init,
        initargs=(
            btp_alg_cls,
            btp_alg_args,
            compliant,
            None,
            None,
            prot_templates,
            {
                "ref_distribution": ref_distribution,
                "attack_seed": attack_seed,
                "attack_trial": attack_trial,
            },
        ),
    ) as exe:
        return list(
            exe.map(
                BTPWorker.invert_no_mem,
                list(range(len(prot_templates))),
                chunksize=protect_chunksize,
            )
        )


def evaluate_inversion_trials(
    original_templates: list[Template],
    protected_templates: list[ProtectedTemplate],
    btp_alg_cls: type,
    btp_alg_args: tuple,
    compliant: bool,
    n_worker: int,
    ref_distribution: Distribution,
    chunksize: int,
    n_attack_trials: int,
    attack_seed: int | None,
    compare: Callable[[Template, Template], float],
    score_writer: CSVScoreWriter,
    key_source: str,
    key_scope: str,
    key_system: int = 0,
) -> None:
    """Run reproducible one-to-one inversion attacks and write their scores."""
    if n_attack_trials < 1:
        raise ValueError("n_attack_trials must be at least 1")
    if len(original_templates) != len(protected_templates):
        raise ValueError("Original and protected template counts must match")

    for attack_trial in range(n_attack_trials):
        logger.info(
            "Running inversion attack trial %d/%d",
            attack_trial + 1,
            n_attack_trials,
        )
        inverted_templates = processed_inversion_no_mem(
            protected_templates,
            btp_alg_cls,
            btp_alg_args,
            compliant,
            n_worker,
            ref_distribution,
            chunksize,
            attack_seed,
            attack_trial,
        )

        for inverted_template, index in inverted_templates:
            original_template = original_templates[index]
            if original_template.get_template() is None:
                continue

            attack_metadata = {
                "attack_trial": attack_trial,
                "attack_solved": inverted_template.get_template() is not None,
                "key": protected_templates[index].get_keys(),
                "key_system": key_system,
                "key_source": key_source,
                "key_scope": key_scope,
            }
            scored_original = Template(
                original_template.subject_id,
                original_template.template_id,
                original_template.get_template(),
                original_template.metadata | attack_metadata,
            )
            inverted_template.metadata = inverted_template.metadata | attack_metadata

            score = float("nan")
            if inverted_template.get_template() is not None:
                score = float(compare(scored_original, inverted_template))
            score_writer.write_score(score, scored_original, inverted_template)
