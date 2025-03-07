import time

import cv2
import numpy as np

from config import Config as c
from custom_profiler import profiler


class VideoReader:

    def __init__(self,
                 path: int | str,
                 target_res: tuple[int, int] | None = None,
                 *,
                 auto_adjust_time: int | float = c.CAP_AUTO_ADJUST_TIME,
                 optimal_focus: int | float = c.CAP_OPTIMAL_FOCUS,
                 adjustable_props: int | float = c.CAP_ADJUSTABLE_PROPS) -> None:
        self.path = path
        self.target_res = target_res
        self.auto_adjust_time = auto_adjust_time
        self.optimal_focus = optimal_focus
        self.adjustable_props = adjustable_props
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
            elif key == ord('0'):
                self.cap.set(cv2.CAP_PROP_FOCUS, self.optimal_focus)
            prop_id, inc_value, prop_name = self.adjustable_props[self.prop_idx]
            print(f'{prop_name}: {self.cap.get(prop_id)}')
