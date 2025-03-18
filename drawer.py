import itertools
import typing

import cv2
import matplotlib.colors
import numpy as np

import model
import profiler
import roi

if typing.TYPE_CHECKING:
    import inference_runner
    import signal_processor
    import video_reader


class Colors:
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    LIGHT_GRAY = (224, 224, 224)
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    YELLOW = (255, 255, 0)
    BLUE_AZURE = (255, 128, 0)
    GREEN_SPRING = (128, 255, 0)
    GREEN_PARIS = (0, 255, 128)


DRAW_MODEL_COLORMAP = {
    model.ModelType.FACE_DETECTOR: Colors.BLUE_AZURE,
    model.ModelType.FACE_LANDMARKER: Colors.GREEN_SPRING,
    model.ModelType.HAND_LANDMARKER: Colors.GREEN_PARIS,
    model.ModelType.PERSON_SEGMENTER: Colors.WHITE
}

MATPLOTLIB_DEFAULT_COLORS = matplotlib.colors.to_rgba_array([f'C{i}' for i in range(len(roi.SELECTED_ROI_CONFIGS))])
GRAPH_SIGNAL_COLORMAP = dict(enumerate((MATPLOTLIB_DEFAULT_COLORS[:, :-1] * 255).round().astype(np.uint8).tolist()))

DRAW_LINE_THICKNESS = 1
DRAW_LINE_TYPE = cv2.LINE_AA
DRAW_POINT_RADIUS = 1

NUM_PLOTS = 3
WINDOW_SIZE = (640, 240 * NUM_PLOTS) # (640, 720) # (640, 480)
WINDOW_MARGINS = (40, 40)

GRAPH_DEFAULT_RANGE = (-1.0, 1.0)


class Drawer:

    def __init__(self,
                 model_colormap: dict[model.ModelType, Colors] | None = None,
                 signal_colormap: dict[int, np.ndarray[np.uint8]] | None = None,
                 *,
                 line_thickness: float = DRAW_LINE_THICKNESS,
                 line_type: int = DRAW_LINE_TYPE,
                 point_radius: float = DRAW_POINT_RADIUS,
                 num_plots: int = NUM_PLOTS,
                 window_size: tuple[int, int] = WINDOW_SIZE,
                 window_margins: tuple[int, int] = WINDOW_MARGINS,
                 graph_default_range: tuple[float, float] = GRAPH_DEFAULT_RANGE) -> None:
        self.model_colormap = model_colormap if model_colormap is not None else DRAW_MODEL_COLORMAP
        self.signal_colormap = signal_colormap if signal_colormap is not None else GRAPH_SIGNAL_COLORMAP
        self.line_thickness = line_thickness
        self.line_type = line_type
        self.point_radius = point_radius
        window_width, window_height = window_size
        window_margin_x, window_margin_y = window_margins
        self.graph_width = window_width - 2 * window_margin_x
        self.graph_height = (window_height - (num_plots + 1) * window_margin_y) // num_plots
        self.graph_origins = [(window_margin_x, i * self.graph_height + (i + 1) * window_margin_y) for i in range(num_plots)]
        self.graph_default_range = graph_default_range
        self.plot = np.full((window_height, window_width, 3), 255, dtype=np.uint8)
        cv2.namedWindow('frame')
        cv2.namedWindow('plot')

    @profiler.timeit
    def draw_inferences(self, frame: cv2.typing.MatLike, model_results: 'inference_runner.InferenceResults') -> cv2.typing.MatLike:
        drawn = frame.copy()
        # result: 'inference_runner.ModelOutput' # TODO: check why its not detecting the type
        for result in model_results:
            if result.model_type in [model.ModelType.FACE_DETECTOR, model.ModelType.FACE_LANDMARKER, model.ModelType.HAND_LANDMARKER]:
                if len(result.detections) > 0:
                    color = self.model_colormap[result.model_type]
                    for bbox, points in result.detections:
                        x_0, y_0, x_1, y_1 = bbox
                        drawn = cv2.rectangle(drawn, (x_0, y_0), (x_1, y_1), color, self.line_thickness, self.line_type)
                        for x, y in points:
                            drawn = cv2.circle(drawn, (x, y), self.point_radius, color, self.line_thickness, self.line_type)
            elif result.model_type in [model.ModelType.PERSON_SEGMENTER]:
                if len(result.masks) > 0:
                    class_mask, conf_masks = result.masks
                    if class_mask.size > 0:
                        drawn = (drawn * np.expand_dims(conf_masks[3], 2)).round().astype(np.uint8)
            else:
                raise NotImplementedError
        return drawn

    @profiler.timeit
    def draw_rois(self, frame: cv2.typing.MatLike, signal_results: 'signal_processor.SignalStore') -> cv2.typing.MatLike:
        drawn = frame.copy()
        rois = signal_results.sg_roi.get_means(as_int=True)
        for s, (x, y, x_0, y_0, x_1, y_1) in enumerate(rois):
            if not np.isnan([x, y, x_0, x_1, y_0, y_1]).any():
                color = self.signal_colormap[s]
                drawn = cv2.rectangle(drawn, (x_0, y_0), (x_1, y_1), color, self.line_thickness, self.line_type)
                drawn = cv2.drawMarker(drawn, (x, y), color, cv2.MARKER_CROSS, self.line_thickness*5, self.line_thickness, self.line_type)
        return drawn

    def write_text(self,
                   frame: cv2.typing.MatLike,
                   text: str,
                   color: tuple[int, int, int] = Colors.GRAY,
                   font_size: float | None = None) -> cv2.typing.MatLike:
        text_x = 15
        text_y = (self.curr_text_index + 1) * 30
        font_size = font_size if font_size is not None else frame.shape[1] / 1024
        written = cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, font_size, color, self.line_thickness, self.line_type)
        self.curr_text_index += 1
        return written

    @profiler.timeit
    def write_info(self,
                   frame: cv2.typing.MatLike,
                   frame_data: 'video_reader.FrameData',
                   signal_results: 'signal_processor.SignalStore') -> cv2.typing.MatLike:
        written = frame.copy()
        mean_fs = signal_results.sg_bpm.signals[0].get_fs()
        mean_bpms = signal_results.sg_bpm.get_means(as_int=True)
        mean_ptts = signal_results.sg_ptt.get_means(as_int=True)
        self.curr_text_index = 0
        written = self.write_text(written, f'curr_fs: {frame_data.sampling_freq:.2f} Hz', Colors.BLUE)
        written = self.write_text(written, f'mean_fs: {mean_fs:.2f} Hz', Colors.BLUE_AZURE)
        self.curr_text_index += 1
        for s, mean_bpm in enumerate(mean_bpms):
            mean_bpm_text = f'mean_bpm_{s}: {mean_bpm} bpm' if not np.isnan(mean_bpm) else 'NaN'
            written = self.write_text(written, mean_bpm_text, Colors.RED)
        self.curr_text_index += 1
        for s, mean_ptt in enumerate(mean_ptts):
            mean_ptt_text = f'mean_ptt_{s}: {mean_ptt} ms' if not np.isnan(mean_ptt) else 'NaN'
            written = self.write_text(written, mean_ptt_text, Colors.GREEN)
        self.curr_text_index += 1
        if frame_data.calibrating:
            written = self.write_text(written, 'calibrating camera', Colors.RED)
        return written

    @profiler.timeit
    def draw_results(self,
                     frame_data: 'video_reader.FrameData',
                     model_results: 'inference_runner.InferenceResults',
                     signal_results: 'signal_processor.SignalStore',
                     alpha: float = 0.75) -> None:
        drawn = frame_data.frame.copy()
        drawn = self.draw_inferences(drawn, model_results)
        drawn = self.draw_rois(drawn, signal_results)
        drawn = self.write_info(drawn, frame_data, signal_results)
        blended = cv2.addWeighted(drawn, alpha, frame_data.frame, 1.0-alpha, 0.0)
        cv2.moveWindow('frame', 1080 + 1920 // 2 - frame_data.frame.shape[1] // 2, 0)
        cv2.imshow('frame', blended)

    @profiler.timeit
    def draw_graph(self,
                   graph_origin: tuple[int, int],
                   graph_range: tuple[tuple[float, float], tuple[float, float]]) -> None:
        (min_x, max_x), (min_y, max_y) = graph_range
        graph_origin_x, graph_origin_y = graph_origin
        graph_tl = (graph_origin_x, graph_origin_y)
        graph_br = (graph_origin_x + self.graph_width, graph_origin_y + self.graph_height)
        self.plot = cv2.rectangle(self.plot, graph_tl, graph_br, Colors.BLACK, lineType=self.line_type)
        if not np.isnan(min_x) and not np.isnan(max_x):
            order_mag = 10 ** np.floor(min(np.log10(max_x - min_x), 1))
            vline_dist = order_mag / 2 if (max_x - min_x) / (order_mag / 2) < 10 else order_mag
            lower_vline_x = np.ceil(min_x / vline_dist) * vline_dist
            upper_vline_x = np.ceil(max_x / vline_dist) * vline_dist
            for x in np.arange(lower_vline_x, upper_vline_x, vline_dist):
                vline_x_n = (x - min_x) / (max_x - min_x)
                vline_x_p = int(vline_x_n * self.graph_width + graph_origin_x)
                vline_p_0 = (vline_x_p, graph_origin_y)
                vline_p_1 = (vline_x_p, graph_origin_y + self.graph_height)
                self.plot = cv2.line(self.plot, vline_p_0, vline_p_1, Colors.LIGHT_GRAY, lineType=self.line_type)
                (tw, th), _ = cv2.getTextSize(f'{x: .2f}', cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)
                tp = (vline_x_p - tw // 2, graph_origin_y + self.graph_height + th + 5)
                self.plot = cv2.putText(self.plot, f'{x: .2f}', tp, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, Colors.LIGHT_GRAY, lineType=self.line_type)
        if min_x <= 0.0 <= max_x:
            axis_x_n = max_x / (max_x - min_x)
            axis_x_p = int(axis_x_n * self.graph_width + graph_origin_x)
            axis_point_0 = (axis_x_p, graph_origin_y)
            axis_point_1 = (axis_x_p, graph_origin_y + self.graph_height)
            self.plot = cv2.line(self.plot, axis_point_0, axis_point_1, Colors.BLACK, lineType=self.line_type)
        if min_y <= 0.0 <= max_y:
            axis_y_n = max_y / (max_y - min_y)
            axis_y_p = int(axis_y_n * self.graph_height + graph_origin_y)
            axis_point_0 = (graph_origin_x, axis_y_p)
            axis_point_1 = (graph_origin_x + self.graph_width, axis_y_p)
            self.plot = cv2.line(self.plot, axis_point_0, axis_point_1, Colors.BLACK, lineType=self.line_type)
        pos_min_x = (graph_origin_x - 5, graph_origin_y + self.graph_height + 15)
        pos_max_x = (graph_origin_x + self.graph_width - 25, graph_origin_y + self.graph_height + 15)
        pos_min_y = (graph_origin_x - 40, graph_origin_y + self.graph_height - 5)
        pos_max_y = (graph_origin_x - 40, graph_origin_y + 15)
        self.plot = cv2.putText(self.plot, f'{min_x: .2f}', pos_min_x, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, Colors.BLACK, lineType=self.line_type)
        self.plot = cv2.putText(self.plot, f'{max_x: .2f}', pos_max_x, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, Colors.BLACK, lineType=self.line_type)
        self.plot = cv2.putText(self.plot, f'{min_y: .2f}', pos_min_y, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, Colors.BLACK, lineType=self.line_type)
        self.plot = cv2.putText(self.plot, f'{max_y: .2f}', pos_max_y, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, Colors.BLACK, lineType=self.line_type)

    @profiler.timeit
    def draw_signal(self,
                    signal: 'signal_processor.Signal',
                    graph_origin: tuple[int, int],
                    graph_range: tuple[tuple[float, float], tuple[float, float]],
                    color: Colors) -> None:
        (min_x, max_x), (min_y, max_y) = graph_range
        data_x, data_y = np.array(signal.x), np.array(signal.y)
        graph_origin_x, graph_origin_y = graph_origin
        graph_cond = (data_x >= min_x) & (data_x <= max_x) & (data_y >= min_y) & (data_y <= max_y)
        data_x_p = (data_x[graph_cond] - min_x) / (max_x - min_x) * self.graph_width + graph_origin_x
        data_y_p = (data_y[graph_cond] - max_y) / (min_y - max_y) * self.graph_height + graph_origin_y
        groups = itertools.groupby(np.vstack((data_x_p, data_y_p)).T, lambda k: np.all(np.isfinite(k)))
        for isfinite, group in groups:
            if isfinite:
                data_g = np.vstack(list(group)).astype(int)
                self.plot = cv2.polylines(self.plot, [data_g], False, color, lineType=self.line_type)

    # TODO: add to config flags to plot each signal

    @profiler.timeit
    def plot_signals(self, signal_results: 'signal_processor.SignalStore') -> None:
        self.plot = np.full_like(self.plot, 255)
        signal_groups = [signal_results.sg_proc, signal_results.sg_spec, signal_results.sg_corr] # TODO: dont hard code it here
        for signal_group, graph_origin in zip(signal_groups, self.graph_origins):
            graph_range_x = signal_group.range_x if np.isfinite(signal_group.range_x).all() else self.graph_default_range
            graph_range_y = signal_group.range_y if np.isfinite(signal_group.range_y).all() else self.graph_default_range
            graph_range = (graph_range_x, graph_range_y)
            self.draw_graph(graph_origin, graph_range)
            for s, signal in enumerate(signal_group):
                self.draw_signal(signal, graph_origin, graph_range, self.signal_colormap[s])
        cv2.moveWindow('plot', 1080, 0)
        cv2.imshow('plot', self.plot)

    def wait_key(self, delay: int = 1) -> int:
        key = cv2.waitKey(delay)
        if key == ord('q'):
            raise KeyboardInterrupt
        return key

    def draw_and_plot(self,
                      frame_data: 'video_reader.FrameData',
                      model_results: 'inference_runner.InferenceResults',
                      signal_results: 'signal_processor.SignalStore',
                      alpha: float = 0.75) -> int:
        self.draw_results(frame_data, model_results, signal_results, alpha)
        self.plot_signals(signal_results)
        return self.wait_key()

    run = draw_and_plot

    def cleanup(self) -> None:
        cv2.destroyAllWindows()
