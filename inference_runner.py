import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.face_detector import FaceDetector, FaceDetectorOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.image_segmenter import ImageSegmenter, ImageSegmenterOptions

import config as c
from custom_profiler import timeit


class InferenceRunner:

    def __init__(
                self,
                face_detector_path: str | None = 'models/blaze_face_short_range.tflite',
                face_landmarker_path: str | None = 'models/face_landmarker.task',
                hand_landmarker_path: str | None = 'models/hand_landmarker.task',
                person_segmenter_path: str | None = 'models/selfie_multiclass.tflite',
                running_mode: VisionTaskRunningMode = c.RUNNING_MODE
            ) -> None:
        self.running_mode = running_mode
        if c.FACE_DETECTION_ENABLED and face_detector_path is not None:
            face_detector_options = FaceDetectorOptions(BaseOptions(face_detector_path), self.running_mode)
            self.face_detector = FaceDetector.create_from_options(face_detector_options)
        if c.FACE_LANDMARKS_ENABLED and face_landmarker_path is not None:
            face_landmarker_options = FaceLandmarkerOptions(BaseOptions(face_landmarker_path), self.running_mode)
            self.face_landmarker = FaceLandmarker.create_from_options(face_landmarker_options)
        if c.HAND_LANDMARKS_ENABLED and hand_landmarker_path is not None:
            hand_landmarker_options = HandLandmarkerOptions(BaseOptions(hand_landmarker_path), self.running_mode)
            self.hand_landmarker = HandLandmarker.create_from_options(hand_landmarker_options)
        if c.PERSON_SEGMENTATION_ENABLED and person_segmenter_path is not None:
            person_segmenter_options = ImageSegmenterOptions(BaseOptions(person_segmenter_path), self.running_mode, True, True)
            self.person_segmenter = ImageSegmenter.create_from_options(person_segmenter_options)

    @timeit
    def run_face_detector(
                self,
                frame_mp: Image,
                timestamp_ms: int | None = None
            ) -> tuple[c.ModelType, c.Detections]:
        face_detections = []
        if c.FACE_DETECTION_ENABLED:
            if self.running_mode == VisionTaskRunningMode.IMAGE:
                face_detection_results = self.face_detector.detect(frame_mp)
            elif self.running_mode == VisionTaskRunningMode.VIDEO:
                face_detection_results = self.face_detector.detect_for_video(frame_mp, timestamp_ms)
            else:
                raise NotImplementedError
            if face_detection_results != []:
                for detection in face_detection_results.detections:
                    bbox = [detection.bounding_box.origin_x,
                            detection.bounding_box.origin_y,
                            detection.bounding_box.origin_x + detection.bounding_box.width,
                            detection.bounding_box.origin_y + detection.bounding_box.height]
                    x = np.clip(np.array([kpt.x for kpt in detection.keypoints]) * frame_mp.width, 0, frame_mp.width - 1).astype(int)
                    y = np.clip(np.array([kpt.y for kpt in detection.keypoints]) * frame_mp.height, 0, frame_mp.height - 1).astype(int)
                    points = np.vstack((x, y)).T
                    shape = (detection.bounding_box.height, detection.bounding_box.width)
                    face_detections.append((bbox, points, shape))
        return c.ModelType.FACE_DETECTOR, face_detections

    @timeit
    def run_face_landmarker(
                self,
                frame_mp: Image,
                timestamp_ms: int | None = None
            ) -> tuple[c.ModelType, c.Detections]:
        face_landmarks = []
        if c.FACE_LANDMARKS_ENABLED:
            if self.running_mode == VisionTaskRunningMode.IMAGE:
                face_landmarker_results = self.face_landmarker.detect(frame_mp)
            elif self.running_mode == VisionTaskRunningMode.VIDEO:
                face_landmarker_results = self.face_landmarker.detect_for_video(frame_mp, timestamp_ms)
            else:
                raise NotImplementedError
            if face_landmarker_results != []:
                for landmarks in face_landmarker_results.face_landmarks:
                    x = np.clip(np.array([lm.x for lm in landmarks]) * frame_mp.width, 0, frame_mp.width - 1).astype(int)
                    y = np.clip(np.array([lm.y for lm in landmarks]) * frame_mp.height, 0, frame_mp.height - 1).astype(int)
                    points = np.vstack((x, y)).T
                    bbox = [min(x), min(y), max(x), max(y)]
                    shape = (bbox[3]-bbox[1], bbox[2]-bbox[0])
                    face_landmarks.append((bbox, points, shape))
        return c.ModelType.FACE_LANDMARKER, face_landmarks

    @timeit
    def run_hand_landmarker(
                self,
                frame_mp: Image,
                timestamp_ms: int | None = None
            ) -> tuple[c.ModelType, c.Detections]:
        hand_landmarks = []
        if c.HAND_LANDMARKS_ENABLED:
            if self.running_mode == VisionTaskRunningMode.IMAGE:
                hand_landmarker_results = self.hand_landmarker.detect(frame_mp)
            elif self.running_mode == VisionTaskRunningMode.VIDEO:
                hand_landmarker_results = self.hand_landmarker.detect_for_video(frame_mp, timestamp_ms)
            else:
                raise NotImplementedError
            if hand_landmarker_results != []:
                for landmarks in hand_landmarker_results.hand_landmarks:
                    x = np.clip(np.array([lm.x for lm in landmarks]) * frame_mp.width, 0, frame_mp.width - 1).astype(int)
                    y = np.clip(np.array([lm.y for lm in landmarks]) * frame_mp.height, 0, frame_mp.height - 1).astype(int)
                    points = np.vstack((x, y)).T
                    bbox = [min(x), min(y), max(x), max(y)]
                    shape = (bbox[3]-bbox[1], bbox[2]-bbox[0])
                    hand_landmarks.append((bbox, points, shape))
        return c.ModelType.HAND_LANDMARKER, hand_landmarks

    @timeit
    def run_person_segmenter(
                self,
                frame_mp: Image,
                timestamp_ms: int | None = None
            ) -> tuple[c.ModelType, c.Masks]:
        class_mask = np.array([])
        conf_masks = []
        if c.PERSON_SEGMENTATION_ENABLED:
            if self.running_mode == VisionTaskRunningMode.IMAGE:
                person_segmenter_results = self.person_segmenter.segment(frame_mp)
            elif self.running_mode == VisionTaskRunningMode.VIDEO:
                person_segmenter_results = self.person_segmenter.segment_for_video(frame_mp, timestamp_ms)
            else:
                raise NotImplementedError
            if person_segmenter_results != []:
                class_mask = person_segmenter_results.category_mask.numpy_view()
                conf_masks = [conf_mask.numpy_view() for conf_mask in person_segmenter_results.confidence_masks]
        return c.ModelType.PERSON_SEGMENTER, (class_mask, conf_masks)

    @timeit
    def run_pipe(
                self,
                frame: cv2.typing.MatLike,
                timestamp_ms: int | None = None
            # ) -> tuple[tuple[c.ModelType, c.Detections | c.Masks]]:
            ) -> tuple[tuple[c.ModelType, c.Detections], tuple[c.ModelType, c.Detections], tuple[c.ModelType, c.Detections], tuple[c.ModelType, c.Masks]]:
        frame_mp = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_detector_results = self.run_face_detector(frame_mp, timestamp_ms)
        face_landmarker_results = self.run_face_landmarker(frame_mp, timestamp_ms)
        hand_landmarker_results = self.run_hand_landmarker(frame_mp, timestamp_ms)
        person_segmenter_results = self.run_person_segmenter(frame_mp, timestamp_ms)
        return face_detector_results, face_landmarker_results, hand_landmarker_results, person_segmenter_results
