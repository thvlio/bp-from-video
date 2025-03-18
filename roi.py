import numpy as np

import model

type Location = tuple[int, int, int, int, int, int] | tuple[float, float, float, float, float, float] | np.ndarray[float]


class ROIConfig:

    def __init__(self, model_type: model.ModelType, landmark_indices: list[int], relative_bbox: tuple[float, float, float, float]) -> None:
        self.model_type = model_type
        self.landmark_indices = landmark_indices
        self.relative_bbox = relative_bbox


FACE_DETECTION_NOSE_INDEX = 2
FACE_LANDMARKS_NOSE_INDEX = 4
FACE_LANDMARKS_FOREHEAD_INDEX = 151 # 10
FACE_LANDMARKS_CHEEK_INDEX = 330 # 101
FACE_LANDMARKS_EYEBROW_INDEX = 337 # 108
HAND_LANDMARKS_WRIST_INDEX = 0
HAND_LANDMARKS_MIDDLE_INDEX = 9

FACE_CHEEK_CONFIG = ROIConfig(model.ModelType.FACE_LANDMARKER, [FACE_LANDMARKS_CHEEK_INDEX], (-0.05, -0.05, 0.15, 0.05))
FACE_EYEBROW_CONFIG = ROIConfig(model.ModelType.FACE_LANDMARKER, [FACE_LANDMARKS_EYEBROW_INDEX], (-0.10, -0.15, 0.25, 0.00))
FACE_FOREHEAD_CONFIG = ROIConfig(model.ModelType.FACE_LANDMARKER, [FACE_LANDMARKS_FOREHEAD_INDEX], (-0.00, -0.10, 0.20, 0.05))
HAND_WRIST_CONFIG = ROIConfig(model.ModelType.HAND_LANDMARKER, [HAND_LANDMARKS_WRIST_INDEX], (-0.10, -0.10, 0.10, 0.10))
HAND_PALM_CONFIG = ROIConfig(model.ModelType.HAND_LANDMARKER, [HAND_LANDMARKS_WRIST_INDEX, HAND_LANDMARKS_MIDDLE_INDEX], (-0.10, -0.10, 0.10, 0.10))

SELECTED_ROI_CONFIGS = [FACE_FOREHEAD_CONFIG, HAND_PALM_CONFIG]
