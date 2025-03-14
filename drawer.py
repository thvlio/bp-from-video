import itertools

import cv2
import numpy as np

from config import Config as c
from custom_profiler import profiler
from signal_data import Signal, SignalCollection


class Drawer:

    def __init__(self,
                 model_colormap: dict[c.ModelType, c.Colors] | None = None,
                 signal_colormap: dict[int, np.ndarray[np.uint8]] | None = None,
                 *,
                 line_thickness: float = c.DRAW_LINE_THICKNESS,
                 line_type: int = c.DRAW_LINE_TYPE,
                 point_radius: float = c.DRAW_POINT_RADIUS,
                 num_plots: int = c.NUM_PLOTS,
                 window_size: tuple[int, int] = c.WINDOW_SIZE,
                 window_margins: tuple[int, int] = c.WINDOW_MARGINS,
                 graph_default_range: tuple[float, float] = c.GRAPH_DEFAULT_RANGE) -> None:
        self.model_colormap = model_colormap if model_colormap is not None else c.DRAW_MODEL_COLORMAP
        self.signal_colormap = signal_colormap if signal_colormap is not None else c.GRAPH_SIGNAL_COLORMAP
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

    def wait_key(self, delay: int = 1) -> int:
        key = cv2.waitKey(delay)
        if key == ord('q'):
            raise KeyboardInterrupt
        return key

    @profiler.timeit
    def draw_inferences(self, frame: cv2.typing.MatLike, model_results: list[c.Detections | c.Masks]) -> cv2.typing.MatLike:
        drawn = frame.copy()
        for model_type, model_result in model_results:
            if model_type in [c.ModelType.FACE_DETECTOR, c.ModelType.FACE_LANDMARKER, c.ModelType.HAND_LANDMARKER]:
                if len(model_result) == 0:
                    continue
                color = self.model_colormap[model_type]
                for bbox, points in model_result:
                    x_0, y_0, x_1, y_1 = bbox
                    drawn = cv2.rectangle(drawn, (x_0, y_0), (x_1, y_1), color, self.line_thickness, self.line_type)
                    for x, y in points:
                        drawn = cv2.circle(drawn, (x, y), self.point_radius, color, self.line_thickness, self.line_type)
            elif model_type is c.ModelType.PERSON_SEGMENTER:
                class_mask, conf_masks = model_result
                if class_mask.size == 0:
                    continue
                drawn = (drawn * np.expand_dims(conf_masks[3], 2)).round().astype(np.uint8)
            else:
                raise NotImplementedError
        return drawn

    @profiler.timeit
    def draw_rois(self, frame: cv2.typing.MatLike, rois: list[c.Location]) -> cv2.typing.MatLike:
        drawn = frame.copy()
        for s, (x, y, x_0, y_0, x_1, y_1) in enumerate(rois):
            if not np.isnan([x, y, x_0, x_1, y_0, y_1]).any():
                color = self.signal_colormap[s]
                drawn = cv2.rectangle(drawn, (x_0, y_0), (x_1, y_1), color, self.line_thickness, self.line_type)
                drawn = cv2.drawMarker(drawn, (x, y), color, cv2.MARKER_CROSS, self.line_thickness*5, self.line_thickness, self.line_type)
        return drawn

    def write_text(self,
                   frame: cv2.typing.MatLike,
                   text: str,
                   color: tuple[int, int, int] = c.Colors.GRAY,
                   font_size: float | None = None) -> cv2.typing.MatLike:
        text_x = 15
        text_y = (self.curr_text_index + 1) * 30
        font_size = font_size if font_size is not None else frame.shape[1] / 1024
        written = cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, font_size, color, self.line_thickness, self.line_type)
        self.curr_text_index += 1
        return written

    # TODO: pass the classes to write info too

    @profiler.timeit
    def write_info(self,
                   frame: cv2.typing.MatLike,
                   # *,
                   curr_fs: float = np.nan,
                   mean_fs: float = np.nan,
                   mean_bpms: list[float] | None = None,
                   mean_ptts: list[float] | None = None,
                   auto_adjust: bool | None = None) -> cv2.typing.MatLike:
        written = frame.copy()
        mean_bpms = mean_bpms if mean_bpms is not None else []
        mean_ptts = mean_ptts if mean_ptts is not None else []
        self.curr_text_index = 0
        written = self.write_text(written, f'curr_fs: {curr_fs:.2f} Hz', c.Colors.BLUE)
        written = self.write_text(written, f'mean_fs: {mean_fs:.2f} Hz', c.Colors.BLUE_AZURE)
        self.curr_text_index += 1
        for s, mean_bpm in enumerate(mean_bpms):
            mean_bpm_text = f'mean_bpm_{s}: {mean_bpm} bpm' if not np.isnan(mean_bpm) else 'NaN'
            written = self.write_text(written, mean_bpm_text, c.Colors.RED)
        self.curr_text_index += 1
        for s, mean_ptt in enumerate(mean_ptts):
            mean_ptt_text = f'mean_ptt_{s}: {mean_ptt} ms' if not np.isnan(mean_ptt) else 'NaN'
            written = self.write_text(written, mean_ptt_text, c.Colors.GREEN)
        self.curr_text_index += 1
        if auto_adjust:
            written = self.write_text(written, 'adjusting exposure & focus', c.Colors.RED)
        return written

    # TODO: use all result classes here

    @profiler.timeit
    def draw_results(self,
                     frame: cv2.typing.MatLike,
                     model_results: list[c.Detections | c.Masks],
                     signal_results: tuple[SignalCollection, ...],
                     # *,
                     signals_roi: SignalCollection,
                     signals_bpm: SignalCollection,
                     signals_ptt: SignalCollection,
                     timestamp_delta: int | float = np.nan,
                     auto_adjust: bool | None = None,
                     alpha: float = 0.75) -> None:
        signals_roi, _, _, _, _,
        drawn = frame.copy()
        drawn = self.draw_inferences(drawn, model_results)
        rois = signals_roi.get_means(as_int=True)
        drawn = self.draw_rois(drawn, rois)
        curr_fs = 1 / timestamp_delta
        mean_fs = signals_bpm.signals[0].get_fs()
        mean_bpms = signals_bpm.get_means(as_int=True)
        mean_ptts = signals_ptt.get_means(as_int=True)
        drawn = self.write_info(drawn, curr_fs, mean_fs, mean_bpms, mean_ptts, auto_adjust)
        blended = cv2.addWeighted(drawn, alpha, frame, 1.0-alpha, 0.0)
        cv2.moveWindow('frame', 1080 + 1920 // 2 - frame.shape[1] // 2, 0)
        cv2.imshow('frame', blended)

    @profiler.timeit
    def draw_graph(self,
                   graph_origin: tuple[int, int],
                   graph_range: tuple[tuple[float, float], tuple[float, float]]) -> None:
        (min_x, max_x), (min_y, max_y) = graph_range
        graph_origin_x, graph_origin_y = graph_origin
        graph_tl = (graph_origin_x, graph_origin_y)
        graph_br = (graph_origin_x + self.graph_width, graph_origin_y + self.graph_height)
        self.plot = cv2.rectangle(self.plot, graph_tl, graph_br, c.Colors.BLACK, lineType=self.line_type)
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
                self.plot = cv2.line(self.plot, vline_p_0, vline_p_1, c.Colors.LIGHT_GRAY, lineType=self.line_type)
                (tw, th), _ = cv2.getTextSize(f'{x: .2f}', cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)
                tp = (vline_x_p - tw // 2, graph_origin_y + self.graph_height + th + 5)
                self.plot = cv2.putText(self.plot, f'{x: .2f}', tp, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, c.Colors.LIGHT_GRAY, lineType=self.line_type)
        if min_x <= 0.0 <= max_x:
            axis_x_n = max_x / (max_x - min_x)
            axis_x_p = int(axis_x_n * self.graph_width + graph_origin_x)
            axis_point_0 = (axis_x_p, graph_origin_y)
            axis_point_1 = (axis_x_p, graph_origin_y + self.graph_height)
            self.plot = cv2.line(self.plot, axis_point_0, axis_point_1, c.Colors.BLACK, lineType=self.line_type)
        if min_y <= 0.0 <= max_y:
            axis_y_n = max_y / (max_y - min_y)
            axis_y_p = int(axis_y_n * self.graph_height + graph_origin_y)
            axis_point_0 = (graph_origin_x, axis_y_p)
            axis_point_1 = (graph_origin_x + self.graph_width, axis_y_p)
            self.plot = cv2.line(self.plot, axis_point_0, axis_point_1, c.Colors.BLACK, lineType=self.line_type)
        pos_min_x = (graph_origin_x - 5, graph_origin_y + self.graph_height + 15)
        pos_max_x = (graph_origin_x + self.graph_width - 25, graph_origin_y + self.graph_height + 15)
        pos_min_y = (graph_origin_x - 40, graph_origin_y + self.graph_height - 5)
        pos_max_y = (graph_origin_x - 40, graph_origin_y + 15)
        self.plot = cv2.putText(self.plot, f'{min_x: .2f}', pos_min_x, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, c.Colors.BLACK, lineType=self.line_type)
        self.plot = cv2.putText(self.plot, f'{max_x: .2f}', pos_max_x, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, c.Colors.BLACK, lineType=self.line_type)
        self.plot = cv2.putText(self.plot, f'{min_y: .2f}', pos_min_y, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, c.Colors.BLACK, lineType=self.line_type)
        self.plot = cv2.putText(self.plot, f'{max_y: .2f}', pos_max_y, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, c.Colors.BLACK, lineType=self.line_type)

    @profiler.timeit
    def draw_signal(self,
                    signal: Signal,
                    graph_origin: tuple[int, int],
                    graph_range: tuple[tuple[float, float], tuple[float, float]],
                    color: c.Colors) -> None:
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

    # TODO: use the signal processor results class
    #       add to config flags to plot each signal
    #       or organize it myself when calling plot signals from bp.py and pbp.py

    @profiler.timeit
    def plot_signals(self, signal_collections: list[SignalCollection]) -> None:
        self.plot = np.full_like(self.plot, 255)
        for signal_collection, graph_origin in zip(signal_collections, self.graph_origins):
            graph_range_x = signal_collection.range_x if np.isfinite(signal_collection.range_x).all() else self.graph_default_range
            graph_range_y = signal_collection.range_y if np.isfinite(signal_collection.range_y).all() else self.graph_default_range
            graph_range = (graph_range_x, graph_range_y)
            self.draw_graph(graph_origin, graph_range)
            for s, signal in enumerate(signal_collection):
                self.draw_signal(signal, graph_origin, graph_range, self.signal_colormap[s])
        cv2.moveWindow('plot', 1080, 0)
        cv2.imshow('plot', self.plot)

    def cleanup(self) -> None:
        cv2.destroyAllWindows()
