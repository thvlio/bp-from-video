import os
import time

import cv2
import numpy as np

import config as c
from drawer import Drawer
from inference_runner import InferenceRunner
from signal_processor import SignalProcessor
from video_reader import VideoReader

from custom_profiler import printit


# TODO: resample signals
#   possibly fill signals with zero
#       or with signal mean
#       or with weighted average (weigth going from 1 to 0 from start to end)
#   resample the signal to get uniform sampling rate

# TODO: evaluate filter design
#   reasonable frequencies are 0.8 hz to 2 hz
#       consider giving the filter a little more room
#   check num taps around
#   check padlen around

# TODO: find best spots in the face
#   check the heatmap implementation
#       filtered signal needs to be checked
#       variations idea should be checked
#       check if hotspots can seen or not
#   search for most vascularized spots on the face

# TODO: segment the skin
#   can be useful if rois are outside the face a litte
#   look for mediapipe image segmentation guide

# TODO: properties
#   CAP_PROP_WB_TEMPE
#   CAP_PROP_TEMPERATURE
#   CAP_PROP_BRIGHTNESS
#   CAP_PROP_CONTRAST
#   CAP_PROP_SATURATION
#   CAP_PROP_HUE
#   CAP_PROP_GAIN

# TODO: filters
#   spatial and temporal

# TODO: check papers
#   see differences
#   implement any signal processing missing

# TODO: what
#   try ppg = (g/ug) / (r/ur) - 1
#   uc is calculated over time interval

# TODO: hmm
#   try and make the image zero mean and 1 std

# TODO: color space
#   g / 2 - b / 4 - r / 4 + 1 / 2


def main():

    # video_reader = VideoReader(0, (720, 1280))
    # video_reader = VideoReader(0, (360, 640))
    # video_reader = VideoReader(0, (240, 320))
    video_reader = VideoReader(0)
    # video_reader = VideoReader('/home/thulio/Downloads/20250213_174156.mp4', (640, 360)) # 320, 180
    # video_reader = VideoReader('/home/thulio/Downloads/20250209_185615.mp4', (640, 360))

    inference_runner = InferenceRunner(running_mode=c.RUNNING_MODE)

    signal_processor = SignalProcessor()

    drawer = Drawer()

    timestamp = np.nan
    timestamp_prev = np.nan

    cv2.namedWindow('frame')

    video_reader.cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    props = [
        (cv2.CAP_PROP_FOCUS, 5, 'cv2.CAP_PROP_FOCUS'), # 50 [0, 250]
        (cv2.CAP_PROP_WB_TEMPERATURE, 100, 'cv2.CAP_PROP_WB_TEMPERATURE'), # 4783 [2000, 6500]
        (cv2.CAP_PROP_BRIGHTNESS, 4, 'cv2.CAP_PROP_BRIGHTNESS'), # 128 [0, 255]
        (cv2.CAP_PROP_CONTRAST, 4, 'cv2.CAP_PROP_CONTRAST'), # 128 [0, 255]
        (cv2.CAP_PROP_SATURATION, 4, 'cv2.CAP_PROP_SATURATION'), # 128 [0, 255]
        (cv2.CAP_PROP_EXPOSURE, 128, 'cv2.CAP_PROP_EXPOSURE'), # 128 [0, 255]
        (cv2.CAP_PROP_GAIN, 4, 'cv2.CAP_PROP_GAIN'), # 31 []
    ]

    prop_idx = 0

    while True:

        # read, frame, timestamp = video_reader.read_frame(crop_portrait=True, flip_horizontally=True)
        read, frame, timestamp = video_reader.read_frame(flip_horizontally=True)
        if not read:
            break

        timestamp_ms = int(timestamp * 1000)
        face_detections, face_landmarks, hand_landmarks, person_masks = inference_runner.run_pipe(frame, timestamp_ms)

        single_face_landmarks = face_landmarks[1][0] if face_landmarks[1] != [] else None
        single_hand_landmarks = hand_landmarks[1][0] if hand_landmarks[1] != [] else None
        # landmarks_collection = [single_face_landmarks, single_face_landmarks]
        landmarks_collection = [single_face_landmarks, single_hand_landmarks]
        # time_signals, freq_signals, peak_freqs_filtered = signal_processor.update_signals(frame, timestamp, landmarks_collection, person_masks)
        time_signals, freq_signals, peak_freqs_filtered, correlations, peak_lags_filtered = signal_processor.update_signals(frame, timestamp, landmarks_collection)

        drawer.draw_signals(time_signals, freq_signals, correlations)

        drawer.set_frame(frame)

        sampling_rate = 1 / (timestamp - timestamp_prev)
        drawer.write_info(video_reader.auto_exposure, sampling_rate, peak_freqs_filtered, peak_lags_filtered)

        results = [face_detections, face_landmarks, hand_landmarks, person_masks]
        rois = [signal_processor.roi_filtered, signal_processor.roi_bboxes]
        drawer.draw_results(results, rois)

        # if single_face_landmarks is not None:
        #     variations = signal_processor.update_heatmap(frame, single_face_landmarks[1])
        #     drawer.draw_heatmap(single_face_landmarks[1], variations)

        cv2.moveWindow('frame', 1080 + 1920 // 2 - frame.shape[1] // 2, 0)
        cv2.imshow('frame', drawer.get_frame())
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        prop_id, inc_value, prop_name = props[prop_idx]
        if key == ord('8'):
            video_reader.cap.set(prop_id, video_reader.cap.get(prop_id) + inc_value)
            print(f'{prop_name}: {video_reader.cap.get(prop_id)}')
        elif key == ord('2'):
            video_reader.cap.set(prop_id, video_reader.cap.get(prop_id) - inc_value)
            print(f'{prop_name}: {video_reader.cap.get(prop_id)}')
        elif key == ord('4'):
            p = (p - 1) % len(props)
            print(f'{prop_name}: {video_reader.cap.get(prop_id)}')
        elif key == ord('5'):
            print(f'{prop_name}: {video_reader.cap.get(prop_id)}')
        elif key == ord('6'):
            p = (p + 1) % len(props)
            print(f'{prop_name}: {video_reader.cap.get(prop_id)}')

        timestamp_prev = timestamp

        # os.system('clear')
        # printit(clear=True)

    cv2.destroyAllWindows()

    printit()


if __name__ == '__main__':
    main()
