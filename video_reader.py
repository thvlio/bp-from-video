import time

import cv2
import numpy as np

from config import Config as c
from custom_profiler import timeit


class VideoReader:

    def __init__(
                self,
                path: int | str,
                target_res: tuple[int, int] | None = None
            ) -> None:
        self.path = path
        self.target_res = target_res
        self.cap = cv2.VideoCapture(path)
        self.cap.read()
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

    # TODO: implement get prop and set prop
    #       or rather inc prop and dec prop
    #       to get it out of bp.py

    @timeit
    def auto_adjust_props(
                self,
                enable: bool
            ) -> None:
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, int(enable))
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 2 * int(enable) + 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 2 * int(enable) + 1)

    @timeit
    def read_frame(
                self,
                crop_portrait: bool = False,
                flip_horizontally: bool = False
            ) -> tuple[bool, cv2.typing.MatLike, float]:
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
        if timestamp >= c.EXPOSURE_ADJUSTMENT_TIME and self.auto_adjust:
            self.auto_adjust_props(False)
            self.auto_adjust = False
        return read, frame, timestamp
