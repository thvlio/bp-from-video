from enum import Enum, auto

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.face_detector import FaceDetector, FaceDetectorOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.image_segmenter import ImageSegmenter, ImageSegmenterOptions

from custom_profiler import profiler

type Detections = list[tuple[tuple[int, int, int, int], np.ndarray[int]]]
type Masks = tuple[np.ndarray[int], list[np.ndarray[float]]]

type Location = tuple[int | float, int | float, int | float, int | float, int | float, int | float]


class ModelType(Enum):
    FACE_DETECTOR = auto()
    FACE_LANDMARKER = auto()
    HAND_LANDMARKER = auto()
    PERSON_SEGMENTER = auto()


FACE_DETECTION_NOSE_INDEX = 2
FACE_LANDMARKS_NOSE_INDEX = 4
FACE_LANDMARKS_FOREHEAD_INDEX = 151 # 10
FACE_LANDMARKS_CHEEK_INDEX = 330 # 101
FACE_LANDMARKS_EYEBROW_INDEX = 337 # 108
HAND_LANDMARKS_WRIST_INDEX = 0
HAND_LANDMARKS_MIDDLE_INDEX = 9

FACE_CHEEK_CONFIG = (ModelType.FACE_LANDMARKER, [FACE_LANDMARKS_CHEEK_INDEX], (-0.05, -0.05, 0.15, 0.05))
FACE_EYEBROW_CONFIG = (ModelType.FACE_LANDMARKER, [FACE_LANDMARKS_EYEBROW_INDEX], (-0.10, -0.15, 0.25, 0.00))
FACE_FOREHEAD_CONFIG = (ModelType.FACE_LANDMARKER, [FACE_LANDMARKS_FOREHEAD_INDEX], (-0.00, -0.10, 0.20, 0.05))
HAND_WRIST_CONFIG = (ModelType.HAND_LANDMARKER, [HAND_LANDMARKS_WRIST_INDEX], (-0.10, -0.10, 0.10, 0.10))
HAND_PALM_CONFIG = (ModelType.HAND_LANDMARKER, [HAND_LANDMARKS_WRIST_INDEX, HAND_LANDMARKS_MIDDLE_INDEX], (-0.10, -0.10, 0.10, 0.10))


class InferenceRunner:

    @profiler.timeit
    def __init__(self,
                 face_detection_enabled: bool = False,
                 face_landmarks_enabled: bool = True,
                 hand_landmarks_enabled: bool = True,
                 person_segmentation_enabled: bool = False,
                 face_detector_path: str | None = 'models/blaze_face_short_range.tflite',
                 face_landmarker_path: str | None = 'models/face_landmarker.task',
                 hand_landmarker_path: str | None = 'models/hand_landmarker.task',
                 person_segmenter_path: str | None = 'models/selfie_multiclass.tflite',
                 roi_configs: list[tuple[ModelType, list[int], tuple[float, float, float, float]]] = (FACE_FOREHEAD_CONFIG, HAND_PALM_CONFIG),
                 running_mode: VisionTaskRunningMode = VisionTaskRunningMode.VIDEO) -> None:
        self.face_detection_enabled = face_detection_enabled
        self.face_landmarks_enabled = face_landmarks_enabled
        self.hand_landmarks_enabled = hand_landmarks_enabled
        self.person_segmentation_enabled = person_segmentation_enabled
        self.roi_configs = roi_configs
        self.running_mode = running_mode
        if self.face_detection_enabled and face_detector_path is not None:
            face_detector_options = FaceDetectorOptions(BaseOptions(face_detector_path), self.running_mode)
            self.face_detector = FaceDetector.create_from_options(face_detector_options)
        if self.face_landmarks_enabled and face_landmarker_path is not None:
            face_landmarker_options = FaceLandmarkerOptions(BaseOptions(face_landmarker_path), self.running_mode)
            self.face_landmarker = FaceLandmarker.create_from_options(face_landmarker_options)
        if self.hand_landmarks_enabled and hand_landmarker_path is not None:
            hand_landmarker_options = HandLandmarkerOptions(BaseOptions(hand_landmarker_path), self.running_mode)
            self.hand_landmarker = HandLandmarker.create_from_options(hand_landmarker_options)
        if self.person_segmentation_enabled and person_segmenter_path is not None:
            person_segmenter_options = ImageSegmenterOptions(BaseOptions(person_segmenter_path), self.running_mode, True, True)
            self.person_segmenter = ImageSegmenter.create_from_options(person_segmenter_options)

    @profiler.timeit
    def _run_face_detector(self, frame_mp: Image, timestamp_ms: int | None = None, sort_largest: bool = True) -> tuple[ModelType, Detections]:
        face_detections = []
        if self.face_detection_enabled:
            if self.running_mode == VisionTaskRunningMode.IMAGE:
                face_detection_results = self.face_detector.detect(frame_mp)
            elif self.running_mode == VisionTaskRunningMode.VIDEO:
                face_detection_results = self.face_detector.detect_for_video(frame_mp, timestamp_ms)
            else:
                raise NotImplementedError
            for detection in face_detection_results.detections:
                bbox = (detection.bounding_box.origin_x,
                        detection.bounding_box.origin_y,
                        detection.bounding_box.origin_x + detection.bounding_box.width,
                        detection.bounding_box.origin_y + detection.bounding_box.height)
                x = np.clip(np.array([kpt.x for kpt in detection.keypoints]) * frame_mp.width, 0, frame_mp.width - 1).astype(int)
                y = np.clip(np.array([kpt.y for kpt in detection.keypoints]) * frame_mp.height, 0, frame_mp.height - 1).astype(int)
                points = np.vstack((x, y)).T
                face_detections.append((bbox, points))
            if sort_largest:
                face_detections = sorted(face_detections, key=lambda e: (e[0][2] - e[0][0]) * (e[0][3] - e[0][1]), reverse=True)
        return ModelType.FACE_DETECTOR, face_detections

    @profiler.timeit
    def _run_face_landmarker(self, frame_mp: Image, timestamp_ms: int | None = None, sort_largest: bool = True) -> tuple[ModelType, Detections]:
        face_landmarks = []
        if self.face_landmarks_enabled:
            if self.running_mode == VisionTaskRunningMode.IMAGE:
                face_landmarker_results = self.face_landmarker.detect(frame_mp)
            elif self.running_mode == VisionTaskRunningMode.VIDEO:
                face_landmarker_results = self.face_landmarker.detect_for_video(frame_mp, timestamp_ms)
            else:
                raise NotImplementedError
            for landmarks in face_landmarker_results.face_landmarks:
                x = np.clip(np.array([lm.x for lm in landmarks]) * frame_mp.width, 0, frame_mp.width - 1).astype(int)
                y = np.clip(np.array([lm.y for lm in landmarks]) * frame_mp.height, 0, frame_mp.height - 1).astype(int)
                bbox = [min(x), min(y), max(x), max(y)]
                points = np.vstack((x, y)).T
                face_landmarks.append((bbox, points))
            if sort_largest:
                face_landmarks = sorted(face_landmarks, key=lambda e: (e[0][2] - e[0][0]) * (e[0][3] - e[0][1]), reverse=True)
        return ModelType.FACE_LANDMARKER, face_landmarks

    @profiler.timeit
    def _run_hand_landmarker(self, frame_mp: Image, timestamp_ms: int | None = None, sort_largest: bool = True) -> tuple[ModelType, Detections]:
        hand_landmarks = []
        if self.hand_landmarks_enabled:
            if self.running_mode == VisionTaskRunningMode.IMAGE:
                hand_landmarker_results = self.hand_landmarker.detect(frame_mp)
            elif self.running_mode == VisionTaskRunningMode.VIDEO:
                hand_landmarker_results = self.hand_landmarker.detect_for_video(frame_mp, timestamp_ms)
            else:
                raise NotImplementedError
            for landmarks in hand_landmarker_results.hand_landmarks:
                x = np.clip(np.array([lm.x for lm in landmarks]) * frame_mp.width, 0, frame_mp.width - 1).astype(int)
                y = np.clip(np.array([lm.y for lm in landmarks]) * frame_mp.height, 0, frame_mp.height - 1).astype(int)
                points = np.vstack((x, y)).T
                bbox = [min(x), min(y), max(x), max(y)]
                hand_landmarks.append((bbox, points))
            if sort_largest:
                hand_landmarks = sorted(hand_landmarks, key=lambda e: (e[0][2] - e[0][0]) * (e[0][3] - e[0][1]), reverse=True)
        return ModelType.HAND_LANDMARKER, hand_landmarks

    @profiler.timeit
    def _run_person_segmenter(self, frame_mp: Image, timestamp_ms: int | None = None) -> tuple[ModelType, Masks]:
        class_mask = np.array([])
        conf_masks = []
        if self.person_segmentation_enabled:
            if self.running_mode == VisionTaskRunningMode.IMAGE:
                person_segmenter_results = self.person_segmenter.segment(frame_mp)
            elif self.running_mode == VisionTaskRunningMode.VIDEO:
                person_segmenter_results = self.person_segmenter.segment_for_video(frame_mp, timestamp_ms)
            else:
                raise NotImplementedError
            class_mask = person_segmenter_results.category_mask.numpy_view()
            conf_masks = [conf_mask.numpy_view() for conf_mask in person_segmenter_results.confidence_masks]
        return ModelType.PERSON_SEGMENTER, (class_mask, conf_masks)

    @profiler.timeit
    def run_pipe(self, frame: cv2.typing.MatLike, timestamp_ms: int | None = None,
                 sort_largest: bool = True) -> list[tuple[ModelType, Detections | Masks]]:
        frame_mp = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_detector_results = self._run_face_detector(frame_mp, timestamp_ms, sort_largest)
        face_landmarker_results = self._run_face_landmarker(frame_mp, timestamp_ms, sort_largest)
        hand_landmarker_results = self._run_hand_landmarker(frame_mp, timestamp_ms, sort_largest)
        person_segmenter_results = self._run_person_segmenter(frame_mp, timestamp_ms)
        return face_detector_results, face_landmarker_results, hand_landmarker_results, person_segmenter_results

    @profiler.timeit
    def calc_rois(self, inference_results: list[tuple[ModelType, Detections | Masks]],
                  roi_configs: tuple[tuple] = (FACE_FOREHEAD_CONFIG, HAND_PALM_CONFIG)) -> list[Location]:
        _, (_, face_landmarks), (_, hand_landmarks), _ = inference_results
        rois = []
        for model_type, landmark_indices, (left_m, top_m, right_m, bottom_m) in roi_configs:
            if model_type is ModelType.FACE_LANDMARKER:
                landmarks = face_landmarks
            elif model_type is ModelType.HAND_LANDMARKER:
                landmarks = hand_landmarks
            else:
                raise NotImplementedError
            if len(landmarks) > 0:
                bbox, points = landmarks[0]
                pp = np.squeeze(np.mean([points[i] for i in landmark_indices], axis=0))
                x, y = pp.round().astype(int) if not np.isnan(pp).any() else pp
                x_0 = int(round(x + left_m * (bbox[2] - bbox[0])))
                y_0 = int(round(y + top_m * (bbox[3] - bbox[1])))
                x_1 = int(round(x + right_m * (bbox[2] - bbox[0])))
                y_1 = int(round(y + bottom_m * (bbox[3] - bbox[1])))
                roi = (x, y, x_0, y_0, x_1, y_1)
            else:
                roi = (np.nan,) * 6
            rois.append(roi)
        return rois
