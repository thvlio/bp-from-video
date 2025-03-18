import multiprocessing as mp
import queue
import threading

import profiler
from drawer import Drawer
from inference_runner import InferenceRunner, InferenceResults
from signal_processor import SignalProcessor, SignalStore
from video_reader import VideoReader, FrameData

profiler.profiler.enabled = False


class Node:

    type NodeType = type[VideoReader | InferenceRunner | SignalProcessor | Drawer]
    type NodeResults = FrameData | InferenceResults | SignalStore

    @staticmethod
    def get_input_data(q_in: queue.Queue | None = None) -> tuple[NodeResults, ...]:
        return q_in.get() if q_in is not None else ()

    @staticmethod
    def put_output_data(data_out: tuple[NodeResults, ...], q_out: queue.Queue | None = None) -> None:
        if data_out is not None and q_out is not None:
            try:
                q_out.get_nowait()
            except queue.Empty:
                pass
            q_out.put_nowait(data_out)

    def __init__(self, *args, node: type[NodeType], q_in: queue.Queue | None, q_out: queue.Queue | None, e_exit: threading.Event, **kwargs) -> None:
        extra_args = (node, q_in, q_out, e_exit)
        self.proc = mp.Process(target=self.run, args=extra_args + args, kwargs=kwargs)

    @classmethod
    def run(cls, node: type[NodeType], q_in: queue.Queue | None, q_out: queue.Queue | None, e_exit: threading.Event, *args, **kwargs) -> None:
        processor = node(*args, **kwargs)
        try:
            while True:
                if e_exit.is_set():
                    break
                input_data = cls.get_input_data(q_in)
                results = processor.run(*input_data)
                output_data = (*input_data, results)
                cls.put_output_data(output_data, q_out)
        except (KeyboardInterrupt, BrokenPipeError):
            pass
        except Exception as e:
            raise e
        finally:
            e_exit.set()
            processor.cleanup()

    def start(self) -> None:
        self.proc.start()

    def join(self) -> None:
        self.proc.join()


def main():

    manager = mp.Manager()
    q_frame = manager.Queue(1)
    q_models = manager.Queue(1)
    q_signals = manager.Queue(1)
    e_exit = manager.Event()

    # video_reader = Node(0, (720, 1280), node=VideoReader, q_in=None, q_out=q_frame, e_exit=e_exit)
    # video_reader = Node(0, (240, 320), node=VideoReader, q_in=None, q_out=q_frame, e_exit=e_exit)
    video_reader = Node(node=VideoReader, q_in=None, q_out=q_frame, e_exit=e_exit)
    inference_runner = Node(node=InferenceRunner, q_in=q_frame, q_out=q_models, e_exit=e_exit)
    signal_processor = Node(node=SignalProcessor, q_in=q_models, q_out=q_signals, e_exit=e_exit)
    drawer = Node(node=Drawer, q_in=q_signals, q_out=None, e_exit=e_exit)

    video_reader.start()
    inference_runner.start()
    signal_processor.start()
    drawer.start()

    video_reader.join()
    inference_runner.join()
    signal_processor.join()
    drawer.join()

    manager.shutdown()
    manager.join()


if __name__ == '__main__':
    main()
