from enum import Enum, auto

import cv2
import numpy as np
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


class Config:

    type Detections = list[tuple[tuple[int, int, int, int], np.ndarray[int]]]
    type Masks = tuple[np.ndarray[int], list[np.ndarray[float]]]

    type SignalData = tuple[np.ndarray[float], np.ndarray[float]]

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
        CONST = auto()
        LINEAR = auto()
        BUTTER = auto()
        FIR = auto()

    class SignalColorChannel(Enum):
        G = auto()
        CG = auto()

    class LocationPair(Enum):
        FACE_FACE = auto()
        FACE_HAND = auto()

    # ============= #
    # video reading #
    # ============= #

    EXPOSURE_ADJUSTMENT_TIME = 5

    OPTIMAL_FOCUS = 65.0 # 50.0

    # =============== #
    # model inference #
    # =============== #

    RUNNING_MODE = VisionTaskRunningMode.VIDEO

    FACE_DETECTION_ENABLED = False
    FACE_LANDMARKS_ENABLED = True
    HAND_LANDMARKS_ENABLED = True
    PERSON_SEGMENTATION_ENABLED = False

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

    SIGNAL_LOCATION_CONFIGS = (
        FACE_FOREHEAD_CONFIG,
        # FACE_CHEEK_CONFIG,
        HAND_PALM_CONFIG,
        # HAND_WRIST_CONFIG,
    )

    # ================= #
    # signal processing #
    # ================= #

    SPECTRUM_TRANSFORM = SpectrumTransform.LS_PGRAM
    SIGNAL_PROCESSING_METHOD = SignalProcessingMethod.FIR
    SIGNAL_COLOR_CHANNEL = SignalColorChannel.G

    SIGNAL_MAX_SAMPLES = 100
    ROI_POS_MAX_SAMPLES = 1

    SIGNAL_MIN_FREQUENCY = 0.5
    SIGNAL_MAX_FREQUENCY = 3.0

    CORR_MIN_LAG = -0.5
    CORR_MAX_LAG = 0.5

    BUTTER_ORDER = 2
    BUTTER_FS = 15

    FIR_TAPS = 127
    FIR_DF = 0.1
    FIR_FS = 15

    CALC_CORRELATION = True

    # ==================== #
    # drawing and plotting #
    # ==================== #

    class Colors(Enum):
        BLACK = (0, 0, 0)
        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        BLUE = (255, 0, 0)
        WHITE = (255, 255, 255)
        CYAN = (0, 255, 255)
        MAGENTA = (255, 0, 255)
        YELLOW = (255, 255, 0)
        BLUE_AZURE = (255, 128, 0)
        GREEN_SPRING = (128, 255, 0)
        GREEN_PARIS = (0, 255, 128)

    MODEL_COLORS = {
        ModelType.FACE_DETECTOR: Colors.BLUE_AZURE,
        ModelType.FACE_LANDMARKER: Colors.GREEN_SPRING,
        ModelType.HAND_LANDMARKER: Colors.GREEN_PARIS,
        ModelType.PERSON_SEGMENTER: Colors.WHITE
    }

    LANDMARK_COLOR = Colors.MAGENTA
    POI_COLOR = Colors.RED
    ROI_COLOR = Colors.RED

    LINE_THICKNESS = 1
    LINE_TYPE = cv2.LINE_AA
    POINT_RADIUS = 1
    TEXT_THICKNESS = 2

    PLOT_SIZE = (640, 240 * (2 + CALC_CORRELATION)) # (640, 720) # (640, 480)
    PLOT_MARGINS = (40, 40)

    # === #
    # etc #
    # === #

    PROFILE_EXEC_TIMES = True
