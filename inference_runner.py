import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.face_detector import FaceDetector, FaceDetectorOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.image_segmenter import ImageSegmenter, ImageSegmenterOptions

from config import Config as c
from custom_profiler import profiler

# TODO: add a InferenceRunner results class
#       try to use two words
#       like InferenceData or InferenceResults (preferred?)

# TODO: make sure run pipeline uses the video reader results class


class InferenceRunner:

    def __init__(self,
                 face_detection_enabled: bool = c.FACE_DETECTION_ENABLED,
                 face_landmarks_enabled: bool = c.FACE_LANDMARKS_ENABLED,
                 hand_landmarks_enabled: bool = c.HAND_LANDMARKS_ENABLED,
                 person_segmentation_enabled: bool = c.PERSON_SEGMENTATION_ENABLED,
                 *,
                 face_detector_path: str | None = 'models/blaze_face_short_range.tflite',
                 face_landmarker_path: str | None = 'models/face_landmarker.task',
                 hand_landmarker_path: str | None = 'models/hand_landmarker.task',
                 person_segmenter_path: str | None = 'models/selfie_multiclass.tflite',
                 running_mode: str | VisionTaskRunningMode = c.INFERENCE_RUNNING_MODE,
                 roi_configs: c.ROIConfig | None = None) -> None:
        self.face_detection_enabled = face_detection_enabled
        self.face_landmarks_enabled = face_landmarks_enabled
        self.hand_landmarks_enabled = hand_landmarks_enabled
        self.person_segmentation_enabled = person_segmentation_enabled
        self.face_detector_path = face_detector_path
        self.face_landmarker_path = face_landmarker_path
        self.hand_landmarker_path = hand_landmarker_path
        self.person_segmenter_path = person_segmenter_path
        self.running_mode = running_mode
        self.roi_configs = roi_configs if roi_configs is not None else c.SELECTED_ROI_CONFIGS
        self.create_models()

    @profiler.timeit
    def create_models(self):
        self.face_detector = None
        if self.face_detection_enabled and self.face_detector_path is not None:
            face_detector_options = FaceDetectorOptions(BaseOptions(self.face_detector_path), self.running_mode)
            self.face_detector = FaceDetector.create_from_options(face_detector_options)
        self.face_landmarker = None
        if self.face_landmarks_enabled and self.face_landmarker_path is not None:
            face_landmarker_options = FaceLandmarkerOptions(BaseOptions(self.face_landmarker_path), self.running_mode)
            self.face_landmarker = FaceLandmarker.create_from_options(face_landmarker_options)
        self.hand_landmarker = None
        if self.hand_landmarks_enabled and self.hand_landmarker_path is not None:
            hand_landmarker_options = HandLandmarkerOptions(BaseOptions(self.hand_landmarker_path), self.running_mode)
            self.hand_landmarker = HandLandmarker.create_from_options(hand_landmarker_options)
        self.person_segmenter = None
        if self.person_segmentation_enabled and self.person_segmenter_path is not None:
            person_segmenter_options = ImageSegmenterOptions(BaseOptions(self.person_segmenter_path), self.running_mode, True, True)
            self.person_segmenter = ImageSegmenter.create_from_options(person_segmenter_options)

    @profiler.timeit
    def run_face_detector(self, frame_mp: Image, timestamp_ms: int | None = None, sort_largest: bool = True) -> c.Detections:
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
        return (c.ModelType.FACE_DETECTOR, face_detections)

    @profiler.timeit
    def run_face_landmarker(self, frame_mp: Image, timestamp_ms: int | None = None, sort_largest: bool = True) -> c.Detections:
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
        return c.ModelType.FACE_LANDMARKER, face_landmarks

    @profiler.timeit
    def run_hand_landmarker(self, frame_mp: Image, timestamp_ms: int | None = None, sort_largest: bool = True) -> c.Detections:
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
        return c.ModelType.HAND_LANDMARKER, hand_landmarks

    @profiler.timeit
    def run_person_segmenter(self, frame_mp: Image, timestamp_ms: int | None = None) -> c.Masks:
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
        return c.ModelType.PERSON_SEGMENTER, (class_mask, conf_masks)

    @staticmethod
    @profiler.timeit
    def calc_rois(model_results: list[c.Detections | c.Masks], roi_configs: c.ROIConfig | None = None) -> list[c.Location]:
        _, (_, face_landmarks), (_, hand_landmarks), _ = model_results
        roi_configs = roi_configs if roi_configs is not None else c.SELECTED_ROI_CONFIGS
        rois = []
        for model_type, landmark_indices, (left_m, top_m, right_m, bottom_m) in roi_configs:
            if model_type is c.ModelType.FACE_LANDMARKER:
                landmarks = face_landmarks
            elif model_type is c.ModelType.HAND_LANDMARKER:
                landmarks = hand_landmarks
            else:
                raise NotImplementedError
            if len(landmarks) > 0:
                bbox, points = landmarks[0]
                pp = np.squeeze(np.mean([points[i] for i in landmark_indices], axis=0))
                x, y = pp.round().astype(int)
                x_0 = int(round(x + left_m * (bbox[2] - bbox[0])))
                y_0 = int(round(y + top_m * (bbox[3] - bbox[1])))
                x_1 = int(round(x + right_m * (bbox[2] - bbox[0])))
                y_1 = int(round(y + bottom_m * (bbox[3] - bbox[1])))
                roi = (x, y, x_0, y_0, x_1, y_1)
            else:
                roi = (np.nan,) * 6
            rois.append(roi)
        return rois

    @profiler.timeit
    def run_pipeline(self, frame: cv2.typing.MatLike, timestamp_ms: int | None = None, sort_largest: bool = True
                     ) -> tuple[list[c.Detections | c.Masks], list[c.Location]]:
        frame_mp = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_detector_results = self.run_face_detector(frame_mp, timestamp_ms, sort_largest)
        face_landmarker_results = self.run_face_landmarker(frame_mp, timestamp_ms, sort_largest)
        hand_landmarker_results = self.run_hand_landmarker(frame_mp, timestamp_ms, sort_largest)
        person_segmenter_results = self.run_person_segmenter(frame_mp, timestamp_ms)
        model_results = (face_detector_results, face_landmarker_results, hand_landmarker_results, person_segmenter_results)
        rois = self.calc_rois(model_results)
        return model_results, rois

    def cleanup(self):
        if self.face_detector is not None:
            self.face_detector.close()
        if self.face_landmarker is not None:
            self.face_landmarker.close()
        if self.hand_landmarker is not None:
            self.hand_landmarker.close()
        if self.person_segmenter is not None:
            self.person_segmenter.close()
