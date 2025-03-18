import collections.abc
import dataclasses
import typing

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.face_detector import FaceDetector, FaceDetectorOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.image_segmenter import ImageSegmenter, ImageSegmenterOptions

import model
import profiler

if typing.TYPE_CHECKING:
    import video_reader

type ModelProcessor = FaceDetector | FaceLandmarker | HandLandmarker | ImageSegmenter


@dataclasses.dataclass
class ModelOutput:

    model_type: model.ModelType
    detections: list[tuple[tuple[int, int, int, int], np.ndarray[int]]] = dataclasses.field(default_factory=list)
    masks: tuple[np.ndarray[int], list[np.ndarray[float]]] = dataclasses.field(default_factory=tuple)


@dataclasses.dataclass
class InferenceResults:

    face_detector: ModelOutput
    face_landmarker: ModelOutput
    hand_landmarker: ModelOutput
    person_segmenter: ModelOutput

    def __iter__(self) -> collections.abc.Iterable[ModelOutput]:
        return (v for v in self.__dict__.values())


MODEL_ENABLED = {
    model.ModelType.FACE_DETECTOR: False,
    model.ModelType.FACE_LANDMARKER: True,
    model.ModelType.HAND_LANDMARKER: True,
    model.ModelType.PERSON_SEGMENTER: False
}

INFERENCE_RUNNING_MODE = VisionTaskRunningMode.VIDEO


class InferenceRunner:

    def __init__(self,
                 model_enabled: dict[model.ModelType: bool] | None = None,
                 *,
                 face_detector_path: str | None = 'models/blaze_face_short_range.tflite',
                 face_landmarker_path: str | None = 'models/face_landmarker.task',
                 hand_landmarker_path: str | None = 'models/hand_landmarker.task',
                 person_segmenter_path: str | None = 'models/selfie_multiclass.tflite',
                 running_mode: str | VisionTaskRunningMode = INFERENCE_RUNNING_MODE) -> None:
        self.model_enabled = model_enabled if model_enabled is not None else MODEL_ENABLED
        self.face_detector_path = face_detector_path
        self.face_landmarker_path = face_landmarker_path
        self.hand_landmarker_path = hand_landmarker_path
        self.person_segmenter_path = person_segmenter_path
        self.running_mode = running_mode
        self.face_detector: FaceDetector | None = None
        self.face_landmarker: FaceLandmarker | None = None
        self.hand_landmarker: HandLandmarker | None = None
        self.person_segmenter: ImageSegmenter | None = None
        self.create_models()

    @profiler.timeit
    def create_models(self):
        if self.model_enabled[model.ModelType.FACE_DETECTOR] and self.face_detector_path is not None:
            face_detector_options = FaceDetectorOptions(BaseOptions(self.face_detector_path), self.running_mode)
            self.face_detector = FaceDetector.create_from_options(face_detector_options)
        if self.model_enabled[model.ModelType.FACE_LANDMARKER] and self.face_landmarker_path is not None:
            face_landmarker_options = FaceLandmarkerOptions(BaseOptions(self.face_landmarker_path), self.running_mode)
            self.face_landmarker = FaceLandmarker.create_from_options(face_landmarker_options)
        if self.model_enabled[model.ModelType.HAND_LANDMARKER] and self.hand_landmarker_path is not None:
            hand_landmarker_options = HandLandmarkerOptions(BaseOptions(self.hand_landmarker_path), self.running_mode)
            self.hand_landmarker = HandLandmarker.create_from_options(hand_landmarker_options)
        if self.model_enabled[model.ModelType.PERSON_SEGMENTER] and self.person_segmenter_path is not None:
            person_segmenter_options = ImageSegmenterOptions(BaseOptions(self.person_segmenter_path), self.running_mode, True, True)
            self.person_segmenter = ImageSegmenter.create_from_options(person_segmenter_options)

    @profiler.timeit
    def run_face_detector(self, frame_mp: Image, timestamp_ms: int) -> ModelOutput:
        face_detections = []
        if self.model_enabled[model.ModelType.FACE_DETECTOR]:
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
            face_detections.sort(key=lambda e: (e[0][2] - e[0][0]) * (e[0][3] - e[0][1]), reverse=True)
        return ModelOutput(model.ModelType.FACE_DETECTOR, detections=face_detections)

    @profiler.timeit
    def run_face_landmarker(self, frame_mp: Image, timestamp_ms: int) -> ModelOutput:
        face_landmarks = []
        if self.model_enabled[model.ModelType.FACE_LANDMARKER]:
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
            face_landmarks.sort(key=lambda e: (e[0][2] - e[0][0]) * (e[0][3] - e[0][1]), reverse=True)
        return ModelOutput(model.ModelType.FACE_LANDMARKER, detections=face_landmarks)

    @profiler.timeit
    def run_hand_landmarker(self, frame_mp: Image, timestamp_ms: int) -> ModelOutput:
        hand_landmarks = []
        if self.model_enabled[model.ModelType.HAND_LANDMARKER]:
            if self.running_mode == VisionTaskRunningMode.IMAGE:
                hand_landmarker_results = self.hand_landmarker.detect(frame_mp)
            elif self.running_mode == VisionTaskRunningMode.VIDEO:
                hand_landmarker_results = self.hand_landmarker.detect_for_video(frame_mp, timestamp_ms)
            else:
                raise NotImplementedError
            for landmarks in hand_landmarker_results.hand_landmarks:
                x = np.clip(np.array([lm.x for lm in landmarks]) * frame_mp.width, 0, frame_mp.width - 1).astype(int)
                y = np.clip(np.array([lm.y for lm in landmarks]) * frame_mp.height, 0, frame_mp.height - 1).astype(int)
                bbox = [min(x), min(y), max(x), max(y)]
                points = np.vstack((x, y)).T
                hand_landmarks.append((bbox, points))
            hand_landmarks.sort(key=lambda e: (e[0][2] - e[0][0]) * (e[0][3] - e[0][1]), reverse=True)
        return ModelOutput(model.ModelType.HAND_LANDMARKER, detections=hand_landmarks)

    @profiler.timeit
    def run_person_segmenter(self, frame_mp: Image, timestamp_ms: int) -> ModelOutput:
        class_mask = np.array([])
        conf_masks = []
        if self.model_enabled[model.ModelType.PERSON_SEGMENTER]:
            if self.running_mode == VisionTaskRunningMode.IMAGE:
                person_segmenter_results = self.person_segmenter.segment(frame_mp)
            elif self.running_mode == VisionTaskRunningMode.VIDEO:
                person_segmenter_results = self.person_segmenter.segment_for_video(frame_mp, timestamp_ms)
            else:
                raise NotImplementedError
            class_mask = person_segmenter_results.category_mask.numpy_view()
            conf_masks = [conf_mask.numpy_view() for conf_mask in person_segmenter_results.confidence_masks]
        return ModelOutput(model.ModelType.PERSON_SEGMENTER, masks=(class_mask, conf_masks))

    @profiler.timeit
    def predict(self, frame_data: 'video_reader.FrameData') -> InferenceResults:
        timestamp_ms = int(frame_data.timestamp * 1000)
        frame_mp = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB))
        face_detector_results = self.run_face_detector(frame_mp, timestamp_ms)
        face_landmarker_results = self.run_face_landmarker(frame_mp, timestamp_ms)
        hand_landmarker_results = self.run_hand_landmarker(frame_mp, timestamp_ms)
        person_segmenter_results = self.run_person_segmenter(frame_mp, timestamp_ms)
        return InferenceResults(face_detector_results, face_landmarker_results, hand_landmarker_results, person_segmenter_results)

    run = predict

    def cleanup(self):
        if self.face_detector is not None:
            self.face_detector.close()
        if self.face_landmarker is not None:
            self.face_landmarker.close()
        if self.hand_landmarker is not None:
            self.hand_landmarker.close()
        if self.person_segmenter is not None:
            self.person_segmenter.close()
