import math

import cv2
import numpy as np

from custom_profiler import printit
from data import SignalCollection
from drawer import Drawer
from inference_runner import InferenceRunner
from signal_processor import SignalProcessor
from video_reader import VideoReader


def main():

    profile_exec_times = True

    # video_reader = VideoReader(0, (720, 1280))
    # video_reader = VideoReader(0, (360, 640))
    # video_reader = VideoReader(0, (240, 320))
    video_reader = VideoReader(0)
    # video_reader = VideoReader('/home/thulio/Downloads/20250213_174156.mp4', (640, 360)) # 320, 180
    # video_reader = VideoReader('/home/thulio/Downloads/20250209_185615.mp4', (640, 360))

    inference_runner = InferenceRunner()

    warmup_steps = 10
    for _ in range(warmup_steps):
        read, frame, timestamp = video_reader.read_frame()
        if not read:
            raise RuntimeWarning
        inference_runner.run_pipe(frame, int(timestamp * 1000))

    signal_processor = SignalProcessor()

    drawer = Drawer()

    roi_max_samples = 1
    signal_max_samples = 200
    peak_max_samples = 50

    num_signals = len(inference_runner.roi_configs)
    signals_roi = SignalCollection(num_signals, fill_value_y=(np.nan,)*6, max_length=roi_max_samples)
    signals_raw = SignalCollection(num_signals, max_length=signal_max_samples)
    signals_bpm = SignalCollection(num_signals, max_length=peak_max_samples)
    signals_ptt = SignalCollection(math.comb(num_signals, 2), max_length=peak_max_samples)

    timestamp = np.nan
    timestamp_prev = np.nan

    cv2.namedWindow('frame')

    print('-' * 80)

    while True:

        # read, frame, timestamp = video_reader.read_frame(crop_portrait=True, flip_horizontally=True)
        read, frame, timestamp = video_reader.read_frame(flip_horizontally=True)
        if not read:
            break

        inference_results = inference_runner.run_pipe(frame, int(timestamp * 1000))

        rois = inference_runner.calc_rois(inference_results)
        signals_roi.add_samples(timestamp, rois)
        rois = signals_roi.get_means(as_int=True)

        samples = signal_processor.sample_signals(frame, rois)
        signals_raw.add_samples(timestamp, samples)

        signals_proc = signal_processor.process_signals(signals_raw)

        signals_spectrum = signal_processor.transform_signals(signals_proc)
        signals_bpm.add_samples(timestamp, [f * 60 for f, _ in signals_spectrum.get_peaks()])

        signals_corr = signal_processor.correlate_signals(signals_proc)
        signals_ptt.add_samples(timestamp, [t * 1000 for t, _ in signals_corr.get_peaks()])

        drawer.set_frame(frame)

        drawer.draw_results(inference_results)
        drawer.draw_rois(rois)

        auto_adjust = video_reader.auto_adjust
        curr_fs = 1 / (timestamp - timestamp_prev)
        mean_fs = signals_proc.signals[0].get_fs()
        mean_bpms = signals_bpm.get_means(as_int=True)
        mean_ptts = signals_ptt.get_means(as_int=True)
        drawer.write_info(auto_adjust, curr_fs, mean_fs, mean_bpms, mean_ptts)

        drawer.draw_signals(signals_proc, signals_spectrum, signals_corr)

        cv2.moveWindow('frame', 1080 + 1920 // 2 - frame.shape[1] // 2, 0)
        cv2.imshow('frame', drawer.get_frame())
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        video_reader.prop_control(key)

        timestamp_prev = timestamp

        # os.system('clear')
        # printit(clear=True)

    cv2.destroyAllWindows()

    video_reader.auto_adjust_props(True)

    printit()


if __name__ == '__main__':
    main()
