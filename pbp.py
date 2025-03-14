import abc
import dataclasses
import multiprocessing as mp
import queue
import threading

import cv2

from config import Config as c
from custom_profiler import profiler
from drawer import Drawer
from inference_runner import InferenceRunner
from signal_data import SignalCollection
from signal_processor import SignalProcessor
from video_reader import VideoReader

profiler.enabled = False


@dataclasses.dataclass
class VideoReaderResults:
    frame: cv2.typing.MatLike
    timestamp: float
    timestamp_ms: int
    timestamp_delta: float
    auto_adjust: bool


@dataclasses.dataclass
class InferenceRunnerResults:
    model_results: list[c.Detections | c.Masks]
    rois: list[c.Location]


@dataclasses.dataclass
class SignalProcessorResults:
    signals_roi: SignalCollection
    signals_raw: SignalCollection
    signals_proc: SignalCollection
    signals_spec: SignalCollection
    signals_corr: SignalCollection
    signals_bpm: SignalCollection
    signals_ptt: SignalCollection


class Processor:

    type ProcessResults = VideoReaderResults | InferenceRunnerResults | SignalProcessorResults

    @staticmethod
    def get_input_data(queue_in: queue.Queue | None) -> ProcessResults | tuple[ProcessResults, ...]:
        return queue_in.get() if queue_in is not None else None

    @staticmethod
    def put_output_data(queue_out: queue.Queue | None, data_out: ProcessResults | tuple[ProcessResults, ...]) -> None:
        if queue_out is not None:
            try:
                queue_out.get_nowait()
            except queue.Empty:
                pass
            if data_out is not None:
                queue_out.put_nowait(data_out)

    # @staticmethod
    # @abc.abstractmethod
    # def process_data(results_in) -> ProcessResults:
    #     ...

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        self.proc: mp.Process # TODO: try to use a pool for the inference runner (or a process for each enabled model)

    def join(self) -> None:
        self.proc.join()


class VideoReaderMP(Processor):

    def run(self, queue_out: queue.Queue, event_exit: threading.Event, *args, **kwargs) -> None:
        video_reader = VideoReader(*args, **kwargs)
        try:
            while True:
                if event_exit.is_set():
                    break
                results_vr = VideoReaderResults(*video_reader.read_frame())
                self.put_output_data(queue_out, results_vr)
        except (KeyboardInterrupt, BrokenPipeError, c.CaptureError):
            pass
        except Exception as e:
            raise e
        finally:
            event_exit.set()
            video_reader.cleanup()

    def __init__(self, *args, queue_out: queue.Queue, event_exit: threading.Event, **kwargs) -> None:
        extra_args = (queue_out, event_exit)
        self.proc = mp.Process(target=self.run, args=extra_args+args, kwargs=kwargs)
        self.proc.start()


class InferenceRunnerMP(Processor):

    def run(self, queue_in: queue.Queue, queue_out: queue.Queue, event_exit: threading.Event, *args, **kwargs) -> None:
        inference_runner = InferenceRunner(*args, **kwargs)
        try:
            while True:
                if event_exit.is_set():
                    break
                results_vr = self.get_input_data(queue_in)
                results_ir = InferenceRunnerResults(*inference_runner.run_pipeline(results_vr.frame, results_vr.timestamp_ms))
                self.put_output_data(queue_out, (results_vr, results_ir))
        except (KeyboardInterrupt, BrokenPipeError):
            pass
        except Exception as e:
            raise e
        finally:
            event_exit.set()
            inference_runner.cleanup()

    def __init__(self, *args, queue_in: queue.Queue, queue_out: queue.Queue, event_exit: threading.Event, **kwargs) -> None:
        extra_args = (queue_in, queue_out, event_exit)
        self.proc = mp.Process(target=self.run, args=extra_args+args, kwargs=kwargs)
        self.proc.start()


class SignalProcessorMP(Processor):

    def run(self, queue_in: queue.Queue, queue_out: queue.Queue, event_exit: threading.Event, *args, **kwargs) -> None:
        signal_processor = SignalProcessor(*args, **kwargs)
        signals_roi, signals_raw, signals_bpm, signals_ptt = signal_processor.create_signals()
        try:
            while True:
                if event_exit.is_set():
                    break
                results_vr, results_ir = self.get_input_data(queue_in)
                results_sp = SignalProcessorResults(*signal_processor.run_pipeline(results_vr.frame, results_vr.timestamp, results_ir.rois,
                                                                                   signals_roi, signals_raw, signals_bpm, signals_ptt))
                _, _, _, signals_roi, signals_raw, signals_bpm, signals_ptt = dataclasses.astuple(results_sp)
                self.put_output_data(queue_out, (results_vr, results_ir, results_sp))
        except (KeyboardInterrupt, BrokenPipeError):
            pass
        except Exception as e:
            raise e
        finally:
            event_exit.set()
            signal_processor.cleanup()

    def __init__(self, *args, queue_in: queue.Queue, queue_out: queue.Queue, event_exit: threading.Event, **kwargs) -> None:
        extra_args = (queue_in, queue_out, event_exit)
        self.proc = mp.Process(target=self.run, args=extra_args+args, kwargs=kwargs)
        self.proc.start()


class DrawerMP(Processor):

    def run(self, queue_in: queue.Queue, event_exit: threading.Event, *args, **kwargs) -> None:
        drawer = Drawer(*args, **kwargs)
        try:
            while True:
                if event_exit.is_set():
                    break
                results_vr, results_ir, results_sp = self.get_input_data(queue_in)
                drawer.draw_results(results_vr.frame, results_ir.model_results,
                                    signals_roi=results_sp.signals_roi, signals_bpm=results_sp.signals_bpm, signals_ptt=results_sp.signals_ptt,
                                    timestamp_delta=results_vr.timestamp_delta, auto_adjust=results_vr.auto_adjust)
                drawer.plot_signals([results_sp.signals_proc, results_sp.signals_spec, results_sp.signals_corr])
                drawer.wait_key()
        except (KeyboardInterrupt, BrokenPipeError):
            pass
        except Exception as e:
            raise e
        finally:
            event_exit.set()
            drawer.cleanup()

    def __init__(self, *args, queue_in: queue.Queue, event_exit: threading.Event, **kwargs) -> None:
        extra_args = (queue_in, event_exit)
        self.proc = mp.Process(target=self.run, args=extra_args+args, kwargs=kwargs)
        self.proc.start()


def main():

    manager = mp.Manager()
    queue_frame = manager.Queue(1)
    queue_inference = manager.Queue(1)
    queue_signal = manager.Queue(1)
    event_exit = manager.Event()

    # video_reader = VideoReaderMP(0, (720, 1280), queue_out=queue_frame, event_exit=event_exit)
    # video_reader = VideoReaderMP(0, (360, 640), queue_out=queue_frame, event_exit=event_exit)
    # video_reader = VideoReaderMP(0, (240, 320), queue_out=queue_frame, event_exit=event_exit)
    video_reader = VideoReaderMP(0, queue_out=queue_frame, event_exit=event_exit)
    inference_runner = InferenceRunnerMP(queue_in=queue_frame, queue_out=queue_inference, event_exit=event_exit)
    signal_processor = SignalProcessorMP(queue_in=queue_inference, queue_out=queue_signal, event_exit=event_exit)
    drawer = DrawerMP(queue_in=queue_signal, event_exit=event_exit)

    video_reader.join()
    inference_runner.join()
    signal_processor.join()
    drawer.join()
    manager.shutdown()
    manager.join()


if __name__ == '__main__':
    main()
