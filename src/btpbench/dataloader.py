# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import logging
import random

from collections.abc import Generator
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy
import pandas

from btpbench.sample import Sample

logger = logging.getLogger("experiment")

_REQUIRED_COLUMNS = ("path", "subject_id", "template_id")


def _metadata_names(dataframe: pandas.DataFrame) -> list[str]:
    """Validate a protocol frame and return its metadata columns in order."""
    missing_columns = [
        column for column in _REQUIRED_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "Protocol CSV is missing required column(s): " + ", ".join(missing_columns)
        )
    return [
        str(column) for column in dataframe.columns if column not in _REQUIRED_COLUMNS
    ]


class LoaderFunction(Protocol):
    """Callable used to lazily load a sample."""

    def __call__(
        self,
        file: Path,
        flip: bool = False,
        *,
        frame_idx: int = 0,
    ) -> numpy.ndarray: ...


def cv_loader(file: Path, flip: bool = False, **kwargs: object) -> numpy.ndarray:
    """Image loader using opencv, returned array is h,w,c with c rgb."""
    img = cv2.imread(str(file))

    if flip:
        return cv2.rotate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.ROTATE_180)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cv_loader_video(
    file: Path, flip: bool = False, random_frame: bool = False, frame_idx: int = 0
) -> numpy.ndarray:
    """Video loader using opencv, returned array is h,w,c with c rgb."""
    video = cv2.VideoCapture(str(file), cv2.CAP_FFMPEG)
    video.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    nb_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_idx > 0:
        frame_idx = int(frame_idx / 100.0 * (nb_frames - 1))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    if random_frame:
        frame_idx = random.randint(0, nb_frames - 1)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, img = video.read()

    if not ret:
        raise RuntimeError(f"Did not manage to extract frame from: `{str(file)}`")

    if flip:
        return cv2.rotate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.ROTATE_180)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _sample_metadata(row) -> dict[str, Any]:
    metadata = {str(key): value for key, value in row.to_dict().items()}
    del metadata["subject_id"]
    del metadata["template_id"]
    del metadata["path"]
    return metadata


class Dataloader:
    """Load the files specified by evaluation protocol."""

    def __init__(
        self,
        protocol_files: tuple[Path, Path],
        dataset_dir: Path,
        extension: str,
        compliant=True,
        load_f: LoaderFunction = cv_loader,
    ):
        def parse_csv(df: pandas.DataFrame) -> Generator[Sample, None, None]:
            for _, row in df.iterrows():
                s_path = dataset_dir / (row["path"] + extension)

                should_flip = False
                if "flip" in row:
                    should_flip = row["flip"]

                frame_idx = 0
                if "frame_idx" in row:
                    frame_idx = row["frame_idx"]

                if not s_path.exists():
                    if compliant:
                        logger.warning(f"file `{str(s_path)}` does not exist")
                        continue
                    raise FileNotFoundError(f"file `{str(s_path)}` does not exist")

                data_f = partial(load_f, s_path, should_flip, frame_idx=frame_idx)

                metadata = _sample_metadata(row)

                yield Sample(
                    subject_id=row["subject_id"],
                    template_id=row["template_id"],
                    path=s_path,
                    metadata=metadata,
                    data=data_f,
                )

        self._dataset_dir = dataset_dir
        self._compliant = compliant
        self._load_f = load_f

        enroll_csv_file = protocol_files[0]
        probe_csv_file = protocol_files[1]

        if not enroll_csv_file.exists():
            raise FileNotFoundError(
                f"Enroll csv file `{str(enroll_csv_file)}` does not exist!"
            )

        if not probe_csv_file.exists():
            raise FileNotFoundError(
                f"Probe csv file `{str(probe_csv_file)}` does not exist!"
            )

        self._enroll_df = pandas.read_csv(
            enroll_csv_file,
        )

        # Currently, these lines are commented out to keep all samples in the enrollment set.
        # We will have to decide later if we want to keep all samples or just one per subject.
        # Keep only one sample per subject for enrollment
        # self._enroll_df = self._enroll_df.drop_duplicates(
        #    subset="subject_id", keep="last"
        # )

        self._probe_df = pandas.read_csv(
            probe_csv_file,
        )

        enroll_metadata = _metadata_names(self._enroll_df)
        probe_metadata = _metadata_names(self._probe_df)
        if set(enroll_metadata) != set(probe_metadata):
            raise ValueError(
                "Enrollment and probe CSVs must define the same metadata columns."
            )
        self._metadata_names = enroll_metadata

        self._references = parse_csv(self._enroll_df)
        self._probes = parse_csv(self._probe_df)

    def metadata_names(self) -> list[str]:
        """Return metadata columns shared by enrollment and probe samples."""
        return list(self._metadata_names)

    def references(self) -> Generator[Sample, None, None]:
        return self._references

    def probes(self) -> Generator[Sample, None, None]:
        return self._probes


class Dataset:
    """Entry point to a dataset."""

    def __init__(
        self,
        protocols_dir: Path,
        dataset_dir: Path,
        verification_samples_file: Path,
        unlinkability_samples_file: Path,
        extension: str,
        compliant=True,
        load_f: LoaderFunction = cv_loader,
    ):
        self._dataset_dir = dataset_dir
        self._extension = extension
        self._compliant = compliant
        self._load_f = load_f
        self._verification_samples_df = pandas.read_csv(verification_samples_file)
        self._unlink_samples_df = pandas.read_csv(unlinkability_samples_file)
        self._verification_metadata_names = _metadata_names(
            self._verification_samples_df
        )
        self._unlink_metadata_names = _metadata_names(self._unlink_samples_df)

        self._protocols: dict[str, dict[str, tuple[Path, Path]]] = dict()

        for enroll_file in protocols_dir.rglob("for_enrolling.csv"):
            split_name = enroll_file.parent.stem
            proto_name = enroll_file.parent.parent.stem

            probe_file = enroll_file.parent / "for_probing.csv"

            if not probe_file.exists():
                logger.warning(
                    f"Incomplete protocol `{proto_name}` for split `{split_name}`"
                )
                continue

            if proto_name not in self._protocols:
                self._protocols[proto_name] = dict()

            self._protocols[proto_name][split_name] = (enroll_file, probe_file)

        irr_dist_file = protocols_dir / "irreversibility" / "for_distribution.csv"
        irr_invert_file = protocols_dir / "irreversibility" / "for_irreversibility.csv"
        self._irr_files: tuple[Path, Path] | None = (irr_dist_file, irr_invert_file)

        if not irr_dist_file.exists() or not irr_invert_file.exists():
            logger.warning("Dataset has no irreversibility files")
            self._irr_files = None

    def metadata_names(self, verification: bool = True) -> list[str]:
        """Return the name of the metadata."""
        source_names = (
            self._verification_metadata_names
            if verification
            else self._unlink_metadata_names
        )
        return list(source_names)

    def samples(self, n_subjects: int = -1, verification: bool = True) -> list[Sample]:
        """Return all samples as a list."""

        def __load_sample(row) -> Sample | None:
            s_path = self._dataset_dir / (row["path"] + self._extension)

            should_flip = False
            if "flip" in row:
                should_flip = row["flip"]

            frame_idx = 0
            if "frame_idx" in row:
                frame_idx = row["frame_idx"]

            if not s_path.exists():
                if self._compliant:
                    logger.warning(f"file `{str(s_path)}` does not exist")
                    return None

                raise FileNotFoundError(f"file `{str(s_path)}` does not exist")

            data_f = partial(self._load_f, s_path, should_flip, frame_idx=frame_idx)

            metadata = _sample_metadata(row)

            return Sample(
                subject_id=row["subject_id"],
                template_id=row["template_id"],
                path=s_path,
                metadata=metadata,
                data=data_f,
            )

        source_df = (
            self._verification_samples_df if verification else self._unlink_samples_df
        )

        # Select n subjects if n_subjects is specified
        if n_subjects > 0:
            subjects = source_df["subject_id"].unique()
            if n_subjects > len(subjects):
                raise ValueError(
                    f"Requested {n_subjects} subjects, but only {len(subjects)} are available."
                )
            selected_subjects = random.sample(list(subjects), n_subjects)
            samples_df = source_df[source_df["subject_id"].isin(selected_subjects)]
        else:
            samples_df = source_df

        samples: list[Sample] = []
        for _, row in samples_df.iterrows():
            sample = __load_sample(row)
            if sample is not None:
                samples.append(sample)

        return samples

    def protocols(self) -> list[str]:
        """Return available protocols."""
        return list(self._protocols.keys())

    def protocol_splits(self, proto_name: str) -> list[str]:
        """Return the available splits for a given protocols."""
        return list(self._protocols.get(proto_name, dict()).keys())

    def load_irreversibility(self) -> Dataloader:
        """Load irreversibility protocols."""
        if self._irr_files is None:
            raise RuntimeError(f"No irreversibility files for {self._dataset_dir}!")

        return Dataloader(
            self._irr_files,
            self._dataset_dir,
            self._extension,
            self._compliant,
            self._load_f,
        )

    def load_protocol(self, proto_name: str, split_name: str = "eval") -> Dataloader:
        """Load a specific protocol."""

        if proto_name not in self._protocols:
            raise RuntimeError(f"Unknown protocol `{proto_name}`")

        proto = self._protocols[proto_name]

        if split_name not in proto:
            raise RuntimeError(
                f"Unknown split `{split_name}` for protocol `{proto_name}`"
            )

        return Dataloader(
            proto[split_name],
            self._dataset_dir,
            self._extension,
            self._compliant,
            self._load_f,
        )
