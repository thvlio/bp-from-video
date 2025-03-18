import dataclasses
import time

import cv2
import numpy as np

import exceptions
import profiler


@dataclasses.dataclass
class FrameData:

    read: bool
    frame: cv2.typing.MatLike
    timestamp: float
    sampling_freq: float
    calibrating: bool


CAP_CALIBRATION_TIME = 5

CAP_ADJUSTABLE_PROPS = [
    (cv2.CAP_PROP_FOCUS, 5, 'cv2.CAP_PROP_FOCUS'), # 50 [0, 250]
    (cv2.CAP_PROP_WB_TEMPERATURE, 100, 'cv2.CAP_PROP_WB_TEMPERATURE'), # 4783 [2000, 6500]
    (cv2.CAP_PROP_BRIGHTNESS, 4, 'cv2.CAP_PROP_BRIGHTNESS'), # 128 [0, 255]
    (cv2.CAP_PROP_CONTRAST, 4, 'cv2.CAP_PROP_CONTRAST'), # 128 [0, 255]
    (cv2.CAP_PROP_SATURATION, 4, 'cv2.CAP_PROP_SATURATION'), # 128 [0, 255]
    (cv2.CAP_PROP_EXPOSURE, 32, 'cv2.CAP_PROP_EXPOSURE'), # 128 [0, 255]
    (cv2.CAP_PROP_GAIN, 4, 'cv2.CAP_PROP_GAIN'), # 31 []
]


class VideoReader:

    def __init__(self,
                 path: int | str = 0,
                 target_res: tuple[int, int] | None = None,
                 *,
                 crop_portrait: bool = None,
                 flip_horizontally: bool = None,
                 calibration_time: int | float = CAP_CALIBRATION_TIME,
                 adjustable_props: list[tuple[int, int, str]] | None = None) -> None:
        self.path = path
        self.target_res = target_res
        self.crop_portrait = crop_portrait if crop_portrait is not None else False
        self.flip_horizontally = flip_horizontally if crop_portrait is not None else isinstance(self.path, int)
        self.calibration_time = calibration_time
        self.adjustable_props = adjustable_props if adjustable_props is not None else CAP_ADJUSTABLE_PROPS
        self.prop_idx = 0
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise exceptions.CaptureError
        read, _ = self.cap.read()
        if not read:
            raise exceptions.CaptureError
        if isinstance(self.path, int):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
            if self.target_res is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_res[0])
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_res[1])
            self.set_prop_calibration(True)
            self.calibrating = True
        else:
            self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
            self.calibrating = False
        self.timestamp_ref = time.time()
        self.timestamp_prev = np.nan

    def set_prop_calibration(self, enable: bool) -> None:
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, int(enable))
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 2 * int(enable) + 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 2 * int(enable) + 1)

    def prop_control(self, key: int) -> None:
        prop_id, inc_value, prop_name = self.adjustable_props[self.prop_idx]
        if ord('0') <= key <= ord('9'):
            if key == ord('8'):
                self.cap.set(prop_id, self.cap.get(prop_id) + inc_value)
            elif key == ord('2'):
                self.cap.set(prop_id, self.cap.get(prop_id) - inc_value)
            elif key == ord('4'):
                self.prop_idx = (self.prop_idx - 1) % len(self.adjustable_props)
            elif key == ord('6'):
                self.prop_idx = (self.prop_idx + 1) % len(self.adjustable_props)
            prop_id, inc_value, prop_name = self.adjustable_props[self.prop_idx]
            print(f'{prop_name}: {self.cap.get(prop_id)}')

    @profiler.timeit
    def read_frame(self) -> FrameData:
        if isinstance(self.path, int):
            timestamp = time.time() - self.timestamp_ref
        else:
            timestamp = self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.cap.get(cv2.CAP_PROP_FPS)
        read, frame = self.cap.read()
        if read:
            if isinstance(self.path, str) and self.target_res is not None:
                frame = cv2.resize(frame, self.target_res[::-1])
            if self.crop_portrait and frame.shape[0] < frame.shape[1]:
                new_width = int(np.round(frame.shape[0] / np.sqrt(2)))
                left = frame.shape[1] // 2 - new_width // 2
                right = frame.shape[1] // 2 + new_width // 2
                frame = frame[:, left:right, :]
            if self.flip_horizontally:
                frame = cv2.flip(frame, 1)
        else:
            raise exceptions.CaptureError
        if timestamp >= self.calibration_time and self.calibrating:
            self.set_prop_calibration(False)
            self.calibrating = False
        sampling_freq = 1 / (timestamp - self.timestamp_prev)
        self.timestamp_prev = timestamp
        return FrameData(read, frame, timestamp, sampling_freq, self.calibrating)

    run = read_frame

    def cleanup(self) -> None:
        self.set_prop_calibration(True)
        self.cap.release()
