import itertools

import cv2
import matplotlib
import matplotlib.colors
import numpy as np

from custom_profiler import timeit
from data import Location, SignalCollection
from inference_runner import Detections, Masks, ModelType


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


class Drawer:

    def __init__(self) -> None:

        self.model_colors = {
            ModelType.FACE_DETECTOR: Colors.BLUE_AZURE,
            ModelType.FACE_LANDMARKER: Colors.GREEN_SPRING,
            ModelType.HAND_LANDMARKER: Colors.GREEN_PARIS,
            ModelType.PERSON_SEGMENTER: Colors.WHITE
        }
        self.line_thickness = 1
        self.line_type = cv2.LINE_AA
        self.point_radius = 1

        num_plots = 3
        window_width, window_height = (640, 240 * num_plots) # (640, 720) # (640, 480)
        window_margin_x, window_margin_y = (40, 40)
        self.graph_width = window_width - 2 * window_margin_x
        self.graph_height = (window_height - (num_plots + 1) * window_margin_y) // num_plots
        self.graph_origins = [(window_margin_x, i * self.graph_height + (i + 1) * window_margin_y) for i in range(num_plots)]
        self.graph_vline_dists = [1.0, 0.5, 0.1]
        self.graph_default_range = (-1.0, 1.0)

        default_colors = matplotlib.colors.to_rgba_array([f'C{i}' for i in range(10)])
        self.signal_colors = [clr.tolist() for clr in (default_colors[:, :-1] * 255).round().astype(np.uint8)]

        self.plot = np.full((window_height, window_width, 3), 255, dtype=np.uint8)
        self.original = None
        self.drawn = None
        cv2.namedWindow('plot')

    def set_frame(self, frame: cv2.typing.MatLike) -> None:
        self.original = frame.copy()
        self.drawn = frame.copy()

    def get_frame(self, alpha: float = 0.75) -> cv2.typing.MatLike:
        return cv2.addWeighted(self.drawn, alpha, self.original, 1.0-alpha, 0.0)

    @timeit
    def draw_results(self, results: list[tuple[ModelType, Detections | Masks]]) -> None:
        for model_type, result in results:
            if model_type in [ModelType.FACE_DETECTOR, ModelType.FACE_LANDMARKER, ModelType.HAND_LANDMARKER]:
                if len(result) == 0:
                    continue
                color = self.model_colors[model_type]
                for bbox, points in result:
                    x_0, y_0, x_1, y_1 = bbox
                    self.drawn = cv2.rectangle(self.drawn, (x_0, y_0), (x_1, y_1), color, self.line_thickness, self.line_type)
                    for x, y in points:
                        self.drawn = cv2.circle(self.drawn, (x, y), self.point_radius, color, self.line_thickness, self.line_type)
            elif model_type is ModelType.PERSON_SEGMENTER:
                class_mask, conf_masks = result
                if class_mask.size == 0:
                    continue
                self.drawn = (self.drawn * np.expand_dims(conf_masks[3], 2)).round().astype(np.uint8)
            else:
                raise NotImplementedError

    @timeit
    def draw_rois(self, rois: list[Location]) -> None:
        for (x, y, x_0, y_0, x_1, y_1), color in zip(rois, self.signal_colors):
            if not np.isnan([x, y, x_0, x_1, y_0, y_1]).any():
                self.drawn = cv2.rectangle(self.drawn, (x_0, y_0), (x_1, y_1), color, self.line_thickness, self.line_type)
                self.drawn = cv2.drawMarker(self.drawn, (x, y), color, cv2.MARKER_CROSS, self.line_thickness*5, self.line_thickness, self.line_type)

    def write_text(self, text: str, color: tuple[int, int, int] = Colors.GRAY, font_size: float | None = None) -> None:
        text_x = 15
        text_y = (self.curr_text_index + 1) * 30
        font_size = self.drawn.shape[1] / 1024 if font_size is None else font_size
        self.drawn = cv2.putText(self.drawn, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, font_size, color, self.line_thickness, self.line_type)
        self.curr_text_index += 1

    @timeit
    def write_info(self, auto_adjust: bool, curr_fs: float, mean_fs: float, mean_bpms: list[float], mean_ptts: list[float]) -> None:
        self.curr_text_index = 0
        self.write_text(f'curr_fs: {curr_fs:.2f} Hz', Colors.BLUE)
        self.write_text(f'mean_fs: {mean_fs:.2f} Hz', Colors.BLUE_AZURE)
        self.curr_text_index += 1
        for s, mean_bpm in enumerate(mean_bpms):
            mean_bpm_text = f'mean_bpm_{s}: {mean_bpm} bpm' if not np.isnan(mean_bpm) else 'NaN'
            self.write_text(mean_bpm_text, Colors.RED)
        self.curr_text_index += 1
        for s, mean_ptt in enumerate(mean_ptts):
            mean_ptt_text = f'mean_ptt_{s}: {mean_ptt} ms' if not np.isnan(mean_ptt) else 'NaN'
            self.write_text(mean_ptt_text, Colors.GREEN)
        self.curr_text_index += 1
        if auto_adjust:
            self.write_text('adjusting exposure & focus', Colors.RED)

    @timeit
    def _draw_graph(self,
                    graph_origin: tuple[int, int],
                    graph_range_x: tuple[float, float],
                    graph_range_y: tuple[float, float],
                    vline_dist: int | float) -> None:
        min_x, max_x = graph_range_x
        min_y, max_y = graph_range_y
        graph_origin_x, graph_origin_y = graph_origin
        graph_tl = (graph_origin_x, graph_origin_y)
        graph_br = (graph_origin_x + self.graph_width, graph_origin_y + self.graph_height)
        self.plot = cv2.rectangle(self.plot, graph_tl, graph_br, Colors.BLACK, lineType=self.line_type)
        if not np.isnan(min_x) and not np.isnan(max_x):
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

    @timeit
    def _draw_signals(self,
                      signal_collection: SignalCollection,
                      graph_origin: tuple[int, int],
                      graph_range_x: tuple[float, float],
                      graph_range_y: tuple[float, float]) -> None:
        min_x, max_x = graph_range_x
        min_y, max_y = graph_range_y
        for signal, color in zip(signal_collection, self.signal_colors):
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

    @timeit
    def draw_signals(self, time_signals: SignalCollection, freq_signals: SignalCollection, corr_signals: SignalCollection) -> None:
        self.plot = np.full_like(self.plot, 255)
        signal_collections = [time_signals, freq_signals, corr_signals]
        for signal_collection, graph_origin, vline_dist in zip(signal_collections, self.graph_origins, self.graph_vline_dists):
            signal_collection.set_ranges()
            graph_range_x = signal_collection.range_x if np.isfinite(signal_collection.range_x).all() else self.graph_default_range
            graph_range_y = signal_collection.range_y if np.isfinite(signal_collection.range_y).all() else self.graph_default_range
            self._draw_graph(graph_origin, graph_range_x, graph_range_y, vline_dist)
            self._draw_signals(signal_collection, graph_origin, graph_range_x, graph_range_y)
        cv2.moveWindow('plot', 1080, 0)
        cv2.imshow('plot', self.plot)
