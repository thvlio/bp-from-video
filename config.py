import enum

import cv2
import matplotlib.colors
import numpy as np
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


class Config:

    # TODO: split into types and constants (think of filenames)

    # TODO: stop using class

    # =============== #
    # custom_profiler #
    # =============== #

    PROFILER_ENABLED = True

    # ============ #
    # video_reader #
    # ============ #

    class CaptureError(RuntimeError):
        pass

    CAP_AUTO_ADJUST_TIME = 5

    CAP_OPTIMAL_FOCUS = 65.0

    CAP_ADJUSTABLE_PROPS = [
        (cv2.CAP_PROP_FOCUS, 5, 'cv2.CAP_PROP_FOCUS'), # 50 [0, 250]
        (cv2.CAP_PROP_WB_TEMPERATURE, 100, 'cv2.CAP_PROP_WB_TEMPERATURE'), # 4783 [2000, 6500]
        (cv2.CAP_PROP_BRIGHTNESS, 4, 'cv2.CAP_PROP_BRIGHTNESS'), # 128 [0, 255]
        (cv2.CAP_PROP_CONTRAST, 4, 'cv2.CAP_PROP_CONTRAST'), # 128 [0, 255]
        (cv2.CAP_PROP_SATURATION, 4, 'cv2.CAP_PROP_SATURATION'), # 128 [0, 255]
        (cv2.CAP_PROP_EXPOSURE, 32, 'cv2.CAP_PROP_EXPOSURE'), # 128 [0, 255]
        (cv2.CAP_PROP_GAIN, 4, 'cv2.CAP_PROP_GAIN'), # 31 []
    ]

    # ================ #
    # inference_runner #
    # ================ #

    class ModelType(enum.Enum):
        FACE_DETECTOR = enum.auto()
        FACE_LANDMARKER = enum.auto()
        HAND_LANDMARKER = enum.auto()
        PERSON_SEGMENTER = enum.auto()

    type Detections = tuple[ModelType, list[tuple[tuple[int, int, int, int], np.ndarray[int]]]]
    type Masks = tuple[ModelType, tuple[np.ndarray[int], list[np.ndarray[float]]]]

    type Location = tuple[int, int, int, int, int, int] | tuple[float, float, float, float, float, float] | np.ndarray[float]

    type ROIConfig = list[tuple[ModelType, list[int], tuple[float, float, float, float]]]

    FACE_DETECTION_ENABLED = False
    FACE_LANDMARKS_ENABLED = True
    HAND_LANDMARKS_ENABLED = True
    PERSON_SEGMENTATION_ENABLED = False

    INFERENCE_RUNNING_MODE = VisionTaskRunningMode.VIDEO

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

    SELECTED_ROI_CONFIGS = [FACE_FOREHEAD_CONFIG, HAND_PALM_CONFIG]

    # =========== #
    # signal_data #
    # =========== #

    type SignalXs = int | float
    type SignalYs = int | float | Location

    # ================= #
    # signal_processing #
    # ================= #

    class SignalColorChannel(enum.Enum):
        GREEN = enum.auto()
        CHROM_GREEN = enum.auto()

    class SignalProcessingMethod(enum.Enum):
        DIFF_1 = enum.auto()
        DIFF_2 = enum.auto()
        INTERP_LINEAR = enum.auto()
        INTERP_CUBIC = enum.auto()
        DETREND_CONST = enum.auto()
        DETREND_LINEAR = enum.auto()
        FILTER_BUTTER = enum.auto()
        FILTER_FIR = enum.auto()

    class SignalSpectrumTransform(enum.Enum):
        DFT_RFFT = enum.auto()
        PGRAM_WELCH = enum.auto()
        PGRAM_LS = enum.auto()

    SIGNAL_COLOR_CHANNEL = SignalColorChannel.GREEN
    SIGNAL_PROCESSING_METHODS = [
        # SignalProcessingMethod.DIFF_1,
        # SignalProcessingMethod.INTERP_CUBIC,
        SignalProcessingMethod.FILTER_BUTTER,
    ]
    SIGNAL_SPECTRUM_TRANSFORM = SignalSpectrumTransform.PGRAM_LS

    FILTER_BUTTER_ORDER = 16
    FILTER_BUTTER_MIN_BW = 0.1
    FILTER_FIR_TAPS = 127
    FILTER_FIR_DF = 0.3

    FILTER_MIN_FREQ = 0.8
    FILTER_MAX_FREQ = 4.0
    SPECTRUM_MIN_MAG = 0.0
    SPECTRUM_MAX_MAG = 1.0

    SIGNALS_MIN_LAG = -0.5
    SIGNALS_MAX_LAG = 0.5
    SIGNALS_MIN_CORR = -1.0
    SIGNALS_MAX_CORR = 1.0

    # ====== #
    # drawer #
    # ====== #

    class Colors:
        BLACK = (0, 0, 0)
        GRAY = (128, 128, 128)
        LIGHT_GRAY = (224, 224, 224)
        WHITE = (255, 255, 255)
        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        BLUE = (255, 0, 0)
        CYAN = (0, 255, 255)
        MAGENTA = (255, 0, 255)
        YELLOW = (255, 255, 0)
        BLUE_AZURE = (255, 128, 0)
        GREEN_SPRING = (128, 255, 0)
        GREEN_PARIS = (0, 255, 128)

    DRAW_MODEL_COLORMAP = {
        ModelType.FACE_DETECTOR: Colors.BLUE_AZURE,
        ModelType.FACE_LANDMARKER: Colors.GREEN_SPRING,
        ModelType.HAND_LANDMARKER: Colors.GREEN_PARIS,
        ModelType.PERSON_SEGMENTER: Colors.WHITE
    }

    MATPLOTLIB_DEFAULT_COLORS = matplotlib.colors.to_rgba_array([f'C{i}' for i in range(len(SELECTED_ROI_CONFIGS))])
    GRAPH_SIGNAL_COLORMAP = dict(enumerate((MATPLOTLIB_DEFAULT_COLORS[:, :-1] * 255).round().astype(np.uint8).tolist()))

    DRAW_LINE_THICKNESS = 1
    DRAW_LINE_TYPE = cv2.LINE_AA
    DRAW_POINT_RADIUS = 1

    NUM_PLOTS = 3
    WINDOW_SIZE = (640, 240 * NUM_PLOTS) # (640, 720) # (640, 480)
    WINDOW_MARGINS = (40, 40)

    GRAPH_DEFAULT_RANGE = (-1.0, 1.0)

    # == #
    # bp #
    # == #

    ROI_MAX_SAMPLES = 1
    SIGNAL_MAX_SAMPLES = 200
    PEAK_MAX_SAMPLES = 50
