from enum import Enum, auto

import cv2
import numpy as np
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


type Detections = list[tuple[list[int, int, int, int], np.ndarray[int], tuple[int, int]]]
type Masks = tuple[np.ndarray[int], list[np.ndarray[float]]]


class ModelType(Enum):
    FACE_DETECTOR = auto()
    FACE_LANDMARKER = auto()
    HAND_LANDMARKER = auto()
    PERSON_SEGMENTER = auto()


class SpectrumTransform(Enum):
    DFT = auto()
    LS_PGRAM = auto()


class SignalProcessingMethod(Enum):
    RAW = auto()
    CONSTANT = auto()
    LINEAR = auto()
    BUTTER = auto()


class PPGPixelValue(Enum):
    G = auto()
    CG = auto()


class CorrelationPair(Enum):
    FACE_FACE = auto()
    FACE_HAND = auto()


#

FACE_DETECTION_ENABLED = False
FACE_LANDMARKS_ENABLED = True
HAND_LANDMARKS_ENABLED = True
PERSON_SEGMENTATION_ENABLED = False


MODEL_COLORS = [(0, 255, 128),
                (128, 255, 0),
                (255, 128, 0),
                (255, 255, 255)]

LANDMARK_COLOR = (255, 0, 255)
ROI_COLOR = (0, 0, 255)

LINE_THICKNESS = 1
LINE_TYPE = cv2.LINE_AA
POINT_RADIUS = 1
TEXT_THICKNESS = 2

PLOT_SIZE = (640, 720) # (640, 480)
PLOT_MARGINS = (40, 40)

#

FACE_DETECTION_NOSE_INDEX = 2
FACE_LANDMARKS_NOSE_INDEX = 4
FACE_LANDMARKS_FOREHEAD_INDEX = 151 # 10
FACE_LANDMARKS_CHEEK_INDEX = 330 # 101
FACE_LANDMARKS_EYEBROW_INDEX = 337 # 108
HAND_LANDMARKS_WRIST_INDEX = 0
HAND_LANDMARKS_MIDDLE_INDEX = 9

ROI_LANDMARK_INDICES = [
    # [FACE_LANDMARKS_CHEEK_INDEX],
    # [FACE_LANDMARKS_EYEBROW_INDEX],
    [FACE_LANDMARKS_FOREHEAD_INDEX],
    [HAND_LANDMARKS_WRIST_INDEX, HAND_LANDMARKS_MIDDLE_INDEX]
    # [HAND_LANDMARKS_WRIST_INDEX]
]

ROI_LANDMARK_CONFIGS = [
    # (-0.05, -0.05, 0.15, 0.05), # (-0.15, -0.05, 0.05, 0.05),
    # (-0.10, -0.15, 0.25, 0.00), # (-0.15, -0.10, 0.10, 0.00),
    (-0.00, -0.10, 0.20, 0.05), # (-0.20, -0.10, 0.20, 0.05),
    (-0.10, -0.10, 0.10, 0.10),
    # (-0.10, -0.10, 0.10, 0.10)
]

CORRELATION_PAIR = CorrelationPair.FACE_HAND

PERSON_SEGMENTER_CLASSES = {
    0: 'background',
    1: 'hair',
    2: 'body-skin',
    3: 'face-skin',
    4: 'clothes',
    5: 'others (accessories)'
}

#

HEATMAP_PADDING = 10
HEATMAP_POINTS = 478

CALC_HEATMAP = False

#

SPECTRUM_TRANSFORM = SpectrumTransform.LS_PGRAM
SIGNAL_PROCESSING_METHOD = SignalProcessingMethod.BUTTER
PPG_PIXEL_VALUE = PPGPixelValue.G

SIGNAL_MAX_SAMPLES = 100
ROI_POS_MAX_SAMPLES = 1

SIGNAL_MIN_FREQUENCY = 0.5
SIGNAL_MAX_FREQUENCY = 3.0

CORR_MIN_LAG = -0.5
CORR_MAX_LAG = 0.5

BUTTER_ORDER = 4
BUTTER_FS = 15

CALC_CORRELATION = True

#

EXPOSURE_ADJUSTMENT_TIME = 5

OPTIMAL_FOCUS = 65.0 # 50.0

RUNNING_MODE = VisionTaskRunningMode.VIDEO
