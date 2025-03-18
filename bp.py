import exceptions
import profiler
from drawer import Drawer
from inference_runner import InferenceRunner
from signal_processor import SignalProcessor
from video_reader import VideoReader


def main():

    # video_reader = VideoReader('/home/thulio/Downloads/20250213_174156.mp4', (640, 360))
    # video_reader = VideoReader('/home/thulio/Downloads/20250213_174156.mp4', (320, 180))
    # video_reader = VideoReader('/home/thulio/Downloads/20250209_185615.mp4', (640, 360))
    # video_reader = VideoReader(0, (720, 1280))
    # video_reader = VideoReader(0, (240, 320))
    video_reader = VideoReader(0)
    inference_runner = InferenceRunner()
    signal_processor = SignalProcessor()
    drawer = Drawer()

    while True:
        try:
            frame_data = video_reader.read_frame()
            model_results = inference_runner.predict(frame_data)
            signal_results = signal_processor.process(frame_data, model_results)
            key = drawer.draw_and_plot(frame_data, model_results, signal_results)
            video_reader.prop_control(key)
            # printit(clear=True)
        except (exceptions.CaptureError, KeyboardInterrupt):
            break

    video_reader.cleanup()
    inference_runner.cleanup()
    signal_processor.cleanup()
    drawer.cleanup()

    profiler.printit()


if __name__ == '__main__':
    main()
