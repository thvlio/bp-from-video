import time

import cv2
import numpy as np

from custom_profiler import profiler


class VideoReader:

    def __init__(self, path: int | str, target_res: tuple[int, int] | None = None) -> None:
        self.path = path
        self.target_res = target_res
        self.auto_adjust_time = 5
        self.optimal_focus = 65.0 # 50.0
        self.props = [
            (cv2.CAP_PROP_FOCUS, 5, 'cv2.CAP_PROP_FOCUS'), # 50 [0, 250]
            (cv2.CAP_PROP_WB_TEMPERATURE, 100, 'cv2.CAP_PROP_WB_TEMPERATURE'), # 4783 [2000, 6500]
            (cv2.CAP_PROP_BRIGHTNESS, 4, 'cv2.CAP_PROP_BRIGHTNESS'), # 128 [0, 255]
            (cv2.CAP_PROP_CONTRAST, 4, 'cv2.CAP_PROP_CONTRAST'), # 128 [0, 255]
            (cv2.CAP_PROP_SATURATION, 4, 'cv2.CAP_PROP_SATURATION'), # 128 [0, 255]
            (cv2.CAP_PROP_EXPOSURE, 32, 'cv2.CAP_PROP_EXPOSURE'), # 128 [0, 255]
            (cv2.CAP_PROP_GAIN, 4, 'cv2.CAP_PROP_GAIN'), # 31 []
        ]
        self.prop_idx = 0
        self.cap = cv2.VideoCapture(path)
        read, _ = self.cap.read()
        if not read:
            raise RuntimeWarning
        if isinstance(self.path, int):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
            if self.target_res is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_res[0])
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_res[1])
            self.auto_adjust_props(True)
            self.auto_adjust = True
        else:
            self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
            self.auto_adjust = False
        self.timestamp_ref = time.time()

    def auto_adjust_props(self, enable: bool) -> None:
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, int(enable))
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 2 * int(enable) + 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 2 * int(enable) + 1)

    @profiler.timeit
    def read_frame(self, crop_portrait: bool = False, flip_horizontally: bool = False) -> tuple[bool, cv2.typing.MatLike, float]:
        if isinstance(self.path, int):
            timestamp = time.time() - self.timestamp_ref
        else:
            timestamp = self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.cap.get(cv2.CAP_PROP_FPS)
        read, frame = self.cap.read()
        if read:
            if isinstance(self.path, str) and self.target_res is not None:
                frame = cv2.resize(frame, self.target_res[::-1])
            if crop_portrait and frame.shape[0] < frame.shape[1]:
                new_width = int(np.round(frame.shape[0] / np.sqrt(2)))
                left = frame.shape[1] // 2 - new_width // 2
                right = frame.shape[1] // 2 + new_width // 2
                frame = frame[:, left:right, :]
            if flip_horizontally:
                frame = cv2.flip(frame, 1)
        if timestamp >= self.auto_adjust_time and self.auto_adjust:
            self.auto_adjust_props(False)
            self.auto_adjust = False
        return read, frame, timestamp

    def prop_control(self, key: int) -> None:
        prop_id, inc_value, prop_name = self.props[self.prop_idx]
        if ord('0') <= key <= ord('9'):
            if key == ord('8'):
                self.cap.set(prop_id, self.cap.get(prop_id) + inc_value)
            elif key == ord('2'):
                self.cap.set(prop_id, self.cap.get(prop_id) - inc_value)
            elif key == ord('4'):
                self.prop_idx = (self.prop_idx - 1) % len(self.props)
            elif key == ord('6'):
                self.prop_idx = (self.prop_idx + 1) % len(self.props)
            elif key == ord('0'):
                self.cap.set(cv2.CAP_PROP_FOCUS, self.optimal_focus)
            prop_id, inc_value, prop_name = self.props[self.prop_idx]
            print(f'{prop_name}: {self.cap.get(prop_id)}')
