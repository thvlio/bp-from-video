from config import Config as c
from custom_profiler import profiler
from drawer import Drawer
from inference_runner import InferenceRunner
from signal_processor import SignalProcessor
from video_reader import VideoReader

# TODO: change import style on files and import entire modules

# TODO: check *, again everywhere

# TODO: adapt bp.py to pbp.py


def main():

    # video_reader = VideoReader('/home/thulio/Downloads/20250213_174156.mp4', (640, 360))
    # video_reader = VideoReader('/home/thulio/Downloads/20250213_174156.mp4', (320, 180))
    # video_reader = VideoReader('/home/thulio/Downloads/20250209_185615.mp4', (640, 360))
    # video_reader = VideoReader(0, (720, 1280))
    # video_reader = VideoReader(0, (360, 640))
    # video_reader = VideoReader(0, (240, 320))
    video_reader = VideoReader(0)
    inference_runner = InferenceRunner()
    signal_processor = SignalProcessor()
    # signals_roi, signals_raw, signals_bpm, signals_ptt = signal_processor.create_signals()
    drawer = Drawer()

    while True:
        try:
            frame, timestamp, timestamp_ms, timestamp_delta, auto_adjust = video_reader.read_frame()
            model_results, rois = inference_runner.run_pipeline(frame, timestamp_ms)
            signal_results = signal_processor.run_pipeline(frame, timestamp, rois)
            signals_roi, signals_raw, signals_proc, signals_spec, signals_corr, signals_bpm, signals_ptt = signal_results
            drawer.draw_results(frame, model_results, signal_results, timestamp_delta, auto_adjust)
            drawer.plot_signals([signals_proc, signals_spec, signals_corr])
            key = drawer.wait_key()
            video_reader.prop_control(key)
            # os.system('clear')
            # printit(clear=True)
        except (c.CaptureError, KeyboardInterrupt):
            break

    video_reader.cleanup()
    inference_runner.cleanup()
    signal_processor.cleanup()
    drawer.cleanup()

    profiler.printit()


if __name__ == '__main__':
    main()
