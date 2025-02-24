import cv2
import numpy as np

import config as c
from drawer import Drawer
from inference_runner import InferenceRunner
from signal_processor import SignalProcessor
from video_reader import VideoReader

from custom_profiler import printit


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
        inference_results = inference_runner.run_pipe(frame, timestamp_ms)

        # face_detector_results, face_landmarker_results, hand_landmarker_results, person_segmenter_results = inference_results
        _, (_, face_landmarks), (_, hand_landmarks), _ = inference_results

        if c.CORRELATION_PAIR == c.CorrelationPair.FACE_FACE:
            largest_face_landmarks = face_landmarks[np.argmax([h * w for _, _, (h, w) in face_landmarks])] if face_landmarks != [] else None
            landmarks_collection = [largest_face_landmarks, largest_face_landmarks]
        elif c.CORRELATION_PAIR == c.CorrelationPair.FACE_HAND:
            largest_face_landmarks = face_landmarks[np.argmax([h * w for _, _, (h, w) in face_landmarks])] if face_landmarks != [] else None
            largest_hand_landmarks = hand_landmarks[np.argmax([h * w for _, _, (h, w) in hand_landmarks])] if hand_landmarks != [] else None
            landmarks_collection = [largest_face_landmarks, largest_hand_landmarks]
        else:
            raise NotImplementedError

        time_signals, freq_signals, peak_freqs_filtered, mean_roi_positions, roi_bboxes, correlations, peak_lags_filtered = \
            signal_processor.update_signals(frame, timestamp, landmarks_collection)

        drawer.draw_signals(time_signals, freq_signals, correlations)

        drawer.set_frame(frame)

        sampling_rate = 1 / (timestamp - timestamp_prev)
        drawer.write_info(video_reader.auto_adjust, sampling_rate, peak_freqs_filtered, peak_lags_filtered)

        drawer.draw_results(inference_results)
        drawer.draw_rois(mean_roi_positions, roi_bboxes)

        if c.CALC_HEATMAP and largest_face_landmarks is not None:
            variations = signal_processor.update_heatmap(frame, largest_face_landmarks[1]) # pylint: disable=E1136
            drawer.draw_heatmap(largest_face_landmarks[1], variations) # pylint: disable=E1136

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
        elif key == ord('0'):
            video_reader.cap.set(cv2.CAP_PROP_FOCUS, c.OPTIMAL_FOCUS)

        timestamp_prev = timestamp

        # os.system('clear')
        # printit(clear=True)

    cv2.destroyAllWindows()

    printit()


if __name__ == '__main__':
    main()
