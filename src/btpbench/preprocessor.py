# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import importlib.resources
import logging
import math

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any

import cv2
import mediapipe as mp
import numpy
import torch

from facenet_pytorch import MTCNN
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from btpbench.exceptions import InvalidInputSampleError
from btpbench.sample import Sample

logger = logging.getLogger()


class Detector:
    """Base class for face detectors."""

    @abstractmethod
    def __call__(
        self,
        sample: Sample,
    ) -> Sample:
        pass


class MTCNNDetector(Detector):
    """Class to detect and align face using MTCNN."""

    def __init__(
        self,
        image_size: tuple[int, int] = (112, 112),
        desired_eyes_location: list[tuple[int, int]] | None = None,
    ):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._image_size = image_size
        self._desired_eyes_location = desired_eyes_location or [(55, 72), (55, 40)]
        self._detector = MTCNN(
            self._image_size[0], device=self._device, post_process=False
        )

    def __call__(
        self,
        sample: Sample,
    ) -> Sample:
        """Crop and align face using MTCNN face detector.

        Args:

            sample: The sample to transform
            detector: The MTCNN detector
        """

        def _crop_align_image_automatic(
            load_f: Callable[[], numpy.ndarray],
            detector: Any,
            image_size: tuple[int, int],
            desired_eyes_location: list[tuple[int, int]],
        ) -> numpy.ndarray:
            img = load_f()

            _, _, points = detector.detect(img, landmarks=True)

            if points is None:
                raise InvalidInputSampleError("No face in image")

            eye1 = points[0][1]
            eye2 = points[0][0]

            return _align(
                desired_eyes_location,
                [eye1[::-1], eye2[::-1]],
                image_size,
                img,
            )

        load_f = sample.data
        func = partial(
            _crop_align_image_automatic,
            load_f,
            self._detector,
            self._image_size,
            self._desired_eyes_location,
        )

        sample.data = func
        return sample


class MediaPipeDetector(Detector):
    """Class to detect and align face using MediaPipe."""

    def __init__(
        self,
        image_size: tuple[int, int] = (112, 112),
        desired_eyes_location: list[tuple[int, int]] | None = None,
    ):
        self._image_size = image_size
        self._desired_eyes_location = desired_eyes_location or [(55, 72), (55, 40)]

        weight_file = str(
            importlib.resources.files("btpbench.baselines.weights").joinpath(
                "detector.tflite"
            )
        )

        self._mp_base_options = python.BaseOptions(model_asset_path=weight_file)
        self._mp_options = vision.FaceDetectorOptions(
            base_options=self._mp_base_options
        )
        self._mp_detector = vision.FaceDetector.create_from_options(self._mp_options)

    def __call__(
        self,
        sample: Sample,
    ) -> Sample:
        """Crop and align face using Mediapipe face detector.

        Args:

            sample: The sample to transform
            image_size: desired final image size
            desired_eyes_location: Eye location in resized image, array is [leye,reye]
        """

        def _crop_align_image_automatic(
            load_f: Callable[[], numpy.ndarray],
            detector: Any,
            image_size: tuple[int, int],
            desired_eyes_location: list[tuple[int, int]],
        ) -> numpy.ndarray:
            def _normalized_to_pixel_coordinates(
                normalized_x: float,
                normalized_y: float,
                image_width: int,
                image_height: int,
            ) -> None | tuple[int, int]:
                """Convert normalized value pair to pixel coordinates."""

                # Checks if the float value is between 0 and 1.
                def is_valid_normalized_value(value: float) -> bool:
                    return (value > 0 or math.isclose(0, value)) and (
                        value < 1 or math.isclose(1, value)
                    )

                if not (
                    is_valid_normalized_value(normalized_x)
                    and is_valid_normalized_value(normalized_y)
                ):
                    raise InvalidInputSampleError("Eye location outside of image")

                x_px = min(math.floor(normalized_x * image_width), image_width - 1)
                y_px = min(math.floor(normalized_y * image_height), image_height - 1)
                return x_px, y_px

            data = load_f()
            mp_image = mp.Image(mp.ImageFormat.SRGB, data=data)
            results = detector.detect(mp_image)

            if not results.detections:
                raise InvalidInputSampleError("No face in image")

            res = results.detections[0]
            left_eye = _normalized_to_pixel_coordinates(
                res.keypoints[1].x, res.keypoints[1].y, data.shape[1], data.shape[0]
            )
            right_eye = _normalized_to_pixel_coordinates(
                res.keypoints[0].x, res.keypoints[0].y, data.shape[1], data.shape[0]
            )

            # Issue with mediapipe where from time to time a bad detection is given.
            if left_eye[0] == right_eye[0] and left_eye[1] == right_eye[1]:
                raise InvalidInputSampleError("No face in image")

            return _align(
                desired_eyes_location,
                [left_eye[::-1], right_eye[::-1]],
                image_size,
                data,
            )

        load_f = sample.data
        func = partial(
            _crop_align_image_automatic,
            load_f,
            self._mp_detector,
            self._image_size,
            self._desired_eyes_location,
        )
        sample.data = func
        return sample


def create_detector(
    detector: str,
    image_size: tuple[int, int] = (112, 112),
    desired_eyes_location: list[tuple[int, int]] | None = None,
) -> Detector:
    """Construct one of the supported face detectors."""
    if detector == "mediapipe":
        return MediaPipeDetector(image_size, desired_eyes_location)
    if detector == "mtcnn":
        return MTCNNDetector(image_size, desired_eyes_location)
    raise RuntimeError(f"Unknown detector: {detector}")


def reshape(sample: Sample, height: int, width: int) -> Sample:
    """Reshape an input image."""

    def _reshape(
        load_f: Callable[[], numpy.ndarray], height: int, width: int
    ) -> numpy.ndarray:
        data = load_f()
        return cv2.resize(data, (width, height), interpolation=cv2.INTER_LINEAR)

    # Modify data to call _reshape
    load_f = sample.data
    reshape_f = partial(_reshape, load_f, height, width)
    sample.data = reshape_f
    return sample


def normalize(sample: Sample) -> Sample:
    """Normalize input image from -1 to 1."""

    def _norm(load_f: Callable[[], numpy.ndarray]) -> numpy.ndarray:
        data = load_f()
        data = data.astype(numpy.float32, copy=False)
        return (data - 127.5) / 128

    # Modify data to call _norm
    load_f = sample.data
    func = partial(_norm, load_f)
    sample.data = func
    return sample


def chanel_first(sample: Sample) -> Sample:
    """Put color chanel first.

    Always call this preprocessor last!
    """

    def _chanel_first(load_f: Callable[[], numpy.ndarray]) -> numpy.ndarray:
        data = load_f()
        data = numpy.moveaxis(data, -1, 0)
        return data[numpy.newaxis, :]

    # Modify data to call _chanel_first
    load_f = sample.data
    func = partial(_chanel_first, load_f)
    sample.data = func
    return sample


def _align(
    desired_eye_locations: list[tuple[int, int]],  # [leye,reye]
    actual_eye_locations: list[tuple[int, int]],
    final_image_size: tuple[int, int],
    image: numpy.ndarray,
):
    """Align eyes in image.

    Args:
        desired_eyes_locations: Eye location in resized image, array is [leye,reye].
        actual_eye_locations: Unaligned eyes location in actual image.
        final_image_size: Desired final image size.
        image: The image to align.
    """

    def _get_anthropometric_measurements(positions: list[tuple[int, int]]):
        """Compute anthropometric measurements.

        Given the eyes coordinates, it computes the
         - The angle between the eyes coordinates
         - The distance between the eyes coordinates
         - The center of the eyes coordinates

        """

        coordinate_a = numpy.array(positions[0])  # Left eye
        coordinate_b = numpy.array(positions[1])  # Right eye

        delta = coordinate_a - coordinate_b
        eyes_angle = numpy.arctan2(delta[0], delta[1]) * 180 / numpy.pi  # to degrees

        eyes_distance = numpy.linalg.norm(delta)

        eyes_center = 1 / 2 * (coordinate_a + coordinate_b)

        return eyes_distance, eyes_center, eyes_angle

    image = numpy.moveaxis(image, -1, -3)

    (target_dist, target_center, target_angle) = _get_anthropometric_measurements(
        desired_eye_locations
    )
    (dist, center, angle) = _get_anthropometric_measurements(actual_eye_locations)

    rotational_angle = angle - target_angle
    ratio = target_dist / dist

    image = numpy.moveaxis(image, -3, -1)
    original_height = image.shape[0]
    original_width = image.shape[1]

    # Rotate image
    rot_mat = cv2.getRotationMatrix2D(center[::-1], rotational_angle, 1.0)
    image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    # Crop

    target_eyes_center_rescaled = numpy.floor(target_center / ratio).astype("int")

    top = int(center[0] - target_eyes_center_rescaled[0])
    left = int(center[1] - target_eyes_center_rescaled[1])

    bottom = max(0, top) + (int(final_image_size[0] / ratio))
    right = max(0, left) + (int(final_image_size[1] / ratio))

    cropped_image = image[max(0, top) : bottom, max(0, left) : right, ...]

    # Checking if we need to pad the cropped image
    # This happens when the cropped image extrapolate the original image dimensions
    expanded_image = cropped_image

    if original_height < bottom or original_width < right:
        pad_height = (
            cropped_image.shape[0] + (bottom - original_height)
            if original_height < bottom
            else cropped_image.shape[0]
        )

        pad_width = (
            cropped_image.shape[1] + (right - original_width)
            if original_width < right
            else cropped_image.shape[1]
        )

        expanded_image = (
            numpy.zeros(
                (pad_height, pad_width, 3),
                dtype=cropped_image.dtype,
            )
            if cropped_image.ndim > 2
            else numpy.zeros((pad_height, pad_width), dtype=cropped_image.dtype)
        )

        expanded_image[0 : cropped_image.shape[0], 0 : cropped_image.shape[1], ...] = (
            cropped_image
        )

    # Checking if we need to translate the image.
    # This happens when the top, left coordinates on the source images is negative
    if top < 0 or left < 0:
        t_mat = numpy.float32([[1, 0, -1 * min(0, left)], [0, 1, -1 * min(0, top)]])
        expanded_image = cv2.warpAffine(
            expanded_image, t_mat, expanded_image.shape[1::-1], flags=cv2.INTER_LINEAR
        )

    # Scaling
    return cv2.resize(
        expanded_image,
        final_image_size[::-1],
        interpolation=cv2.INTER_LINEAR,
    )


def crop_align_image_annotated(
    sample: Sample,
    image_size: tuple[int, int] = (112, 112),
    desired_eyes_location: list[tuple[int, int]] = [(55, 72), (55, 40)],
) -> Sample:
    """Crop an image based on annotations (leye and reye)."""

    def _crop(
        load_f: Callable[[], numpy.ndarray],
        image_size: tuple[int, int],
        desired_eyes_location: list[tuple[int, int]],
        leye: tuple[int, int],
        reye: tuple[int, int],
    ) -> numpy.ndarray:
        data = load_f()
        return _align(desired_eyes_location, [leye[::-1], reye[::-1]], image_size, data)

    left_eye_keys = ["leye_x", "leye_y"]
    right_eye_keys = ["reye_x", "reye_y"]

    for k in left_eye_keys + right_eye_keys:
        if k not in sample.metadata:
            raise RuntimeError(f"No metadata '{k}' in protocol files.")

    left_eye_loc: tuple[int, int] = (
        sample.metadata[left_eye_keys[0]],
        sample.metadata[left_eye_keys[1]],
    )

    right_eye_loc: tuple[int, int] = (
        sample.metadata[right_eye_keys[0]],
        sample.metadata[right_eye_keys[1]],
    )

    # Modify data to call _crop
    load_f = sample.data
    crop_f = partial(
        _crop, load_f, image_size, desired_eyes_location, left_eye_loc, right_eye_loc
    )
    sample.data = crop_f
    return sample
