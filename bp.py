import cv2
import numpy as np

from config import Config as c
from custom_profiler import printit
from drawer import Drawer
from inference_runner import InferenceRunner
from signal_processor import SignalProcessor
from video_reader import VideoReader


def main():

    # video_reader = VideoReader(0, (720, 1280))
    # video_reader = VideoReader(0, (360, 640))
    # video_reader = VideoReader(0, (240, 320))
    video_reader = VideoReader(0)
    # video_reader = VideoReader('/home/thulio/Downloads/20250213_174156.mp4', (640, 360)) # 320, 180
    # video_reader = VideoReader('/home/thulio/Downloads/20250209_185615.mp4', (640, 360))

    inference_runner = InferenceRunner(running_mode=c.RUNNING_MODE)

    print('-' * 80)

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

    prop_idx = 5

    while True:

        # read, frame, timestamp = video_reader.read_frame(crop_portrait=True, flip_horizontally=True)
        read, frame, timestamp = video_reader.read_frame(flip_horizontally=True)
        if not read:
            break

        timestamp_ms = int(timestamp * 1000)
        inference_results = inference_runner.run_pipe(frame, timestamp_ms)

        mean_pois, mean_rois = signal_processor.update_rois(inference_results)

        time_signals, freq_signals, correlations, mean_peak_freqs, mean_peak_lags = signal_processor.update_signals(frame, timestamp, mean_pois, mean_rois)

        drawer.draw_signals(time_signals, freq_signals, correlations)

        drawer.set_frame(frame)

        sampling_rate = 1 / (timestamp - timestamp_prev)
        drawer.write_info(video_reader.auto_adjust, sampling_rate, mean_peak_freqs, mean_peak_lags)

        drawer.draw_results(inference_results)
        drawer.draw_rois(mean_pois, mean_rois)

        cv2.moveWindow('frame', 1080 + 1920 // 2 - frame.shape[1] // 2, 0)
        cv2.imshow('frame', drawer.get_frame())
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        prop_id, inc_value, prop_name = props[prop_idx]
        if ord('0') <= key <= ord('9'):
            if key == ord('8'):
                video_reader.cap.set(prop_id, video_reader.cap.get(prop_id) + inc_value)
            elif key == ord('2'):
                video_reader.cap.set(prop_id, video_reader.cap.get(prop_id) - inc_value)
            elif key == ord('4'):
                prop_idx = (prop_idx - 1) % len(props)
            elif key == ord('6'):
                prop_idx = (prop_idx + 1) % len(props)
            elif key == ord('0'):
                video_reader.cap.set(cv2.CAP_PROP_FOCUS, c.OPTIMAL_FOCUS)
            prop_id, inc_value, prop_name = props[prop_idx]
            print(f'{prop_name}: {video_reader.cap.get(prop_id)}')

        timestamp_prev = timestamp

        # os.system('clear')
        # printit(clear=True)

    cv2.destroyAllWindows()

    video_reader.auto_adjust_props(True)

    printit()


if __name__ == '__main__':
    main()
