import itertools
from collections import deque

import cv2
import matplotlib
import matplotlib.colors
import numpy as np
from matplotlib import pyplot as plt

from config import Config as c
from custom_profiler import timeit


class Drawer:

    def __init__(
                self
            ) -> None:
        cv2.namedWindow('plot')
        self.plot_width, self.plot_height = c.PLOT_SIZE
        self.plot_shape = (self.plot_height, self.plot_width, 3)
        self.plot_margin_x, self.plot_margin_y = c.PLOT_MARGINS
        self.plot_graph_width = self.plot_width - 2 * self.plot_margin_x
        num_plots = 3 if c.CALC_CORRELATION else 2
        self.plot_graph_height = (self.plot_height - (num_plots + 1) * self.plot_margin_y) // num_plots
        self.plot_graph_origins = [(self.plot_margin_x, i * self.plot_graph_height + (i + 1) * self.plot_margin_y) for i in range(num_plots)]
        self.plot_graph_div_incs = [1.0, 0.5, 0.1] if c.CALC_CORRELATION else [1.0, 0.5]
        default_colors = matplotlib.colors.to_rgba_array([f'C{i}' for i in range(10)])
        self.plot_graph_colors = [cc.tolist() for cc in (default_colors[:, :-1] * 255).round().astype(np.uint8)]
        self.cmap = plt.colormaps.get_cmap('plasma')
        self.original = None
        self.drawn = None
        self.curr_text_index = 0

    def set_frame(
                self,
                frame: cv2.typing.MatLike
            ) -> None:
        self.original = frame.copy()
        self.drawn = frame.copy()

    def get_frame(
                self,
                alpha: float = 0.75
            ) -> cv2.typing.MatLike:
        return cv2.addWeighted(self.drawn, alpha, self.original, 1.0-alpha, 0.0)

    @timeit
    def draw_results(
                self,
                results: list[tuple[c.ModelType, c.Detections | c.Masks]]
            ) -> None:
        for model_type, result in results:
            if model_type in [c.ModelType.FACE_DETECTOR, c.ModelType.FACE_LANDMARKER, c.ModelType.HAND_LANDMARKER]:
                if len(result) == 0:
                    continue
                color = c.MODEL_COLORS[model_type].value
                for bbox, points in result:
                    x_0, y_0, x_1, y_1 = bbox
                    self.drawn = cv2.rectangle(self.drawn, (x_0, y_0), (x_1, y_1), color, c.LINE_THICKNESS, c.LINE_TYPE)
                    for x_p, y_p in points:
                        self.drawn = cv2.circle(self.drawn, (x_p, y_p), c.POINT_RADIUS, color, c.LINE_THICKNESS, c.LINE_TYPE)
            elif model_type is c.ModelType.PERSON_SEGMENTER:
                class_mask, conf_masks = result
                if class_mask.size == 0:
                    continue
                # face_conf_mask = (conf_masks[3] * 255).round().astype(np.uint8)
                # self.drawn = cv2.cvtColor(face_conf_mask, cv2.COLOR_GRAY2BGR)
                self.drawn = (self.drawn * np.expand_dims(conf_masks[3], 2)).round().astype(np.uint8)
            else:
                raise NotImplementedError

    @timeit
    def draw_rois(
                self,
                roi_references: list[deque],
                roi_bboxes: list[deque]
            ) -> None:
        roi_references = list(roi_references)
        roi_bboxes = list(roi_bboxes)
        for (x_f, y_f), (x_0, y_0, x_1, y_1), color in zip(roi_references, roi_bboxes, self.plot_graph_colors):
            if np.isnan([x_f, y_f, x_0, x_1, y_0, y_1]).any():
                continue
            self.drawn = cv2.rectangle(self.drawn, (x_0, y_0), (x_1, y_1), color, c.LINE_THICKNESS, c.LINE_TYPE)
            self.drawn = cv2.drawMarker(self.drawn, (int(x_f), int(y_f)), color, cv2.MARKER_CROSS, c.LINE_THICKNESS*5, c.LINE_THICKNESS, c.LINE_TYPE)

    def write_text(
                self,
                text: str,
                color: tuple[int, int, int] = (128, 128, 128),
                font_size: float | None = None
            ) -> None:
        text_x = 15
        text_y = (self.curr_text_index + 1) * 30
        font_size = self.drawn.shape[1] / 1024 if font_size is None else font_size
        self.drawn = cv2.putText(self.drawn, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, font_size, color, c.LINE_THICKNESS, c.LINE_TYPE)
        self.curr_text_index += 1

    def write_info(
                self,
                auto_adjust: bool,
                sampling_rate: float,
                peak_freqs: list[float],
                peak_lags: list[float]
            ) -> None:
        self.curr_text_index = 0
        self.write_text(f'sample rate: {sampling_rate:.2f} Hz', (255, 0, 0))
        self.curr_text_index += 1
        for s, peak_freq in enumerate(peak_freqs):
            peak_freq_text = f'peak_freq_{s}: {int(peak_freq * 60)} bpm' if not np.isnan(peak_freq) else 'NaN'
            self.write_text(peak_freq_text, (0, 0, 255))
        self.curr_text_index += 1
        for s, peak_lag in enumerate(peak_lags):
            peak_lag_text = f'peak_lag_{s}: {int(peak_lag * 1000)} ms' if not np.isnan(peak_lag) else 'NaN'
            self.write_text(peak_lag_text, (0, 255, 0))
        if auto_adjust:
            self.write_text('adjusting exposure & focus', (0, 0, 255))

    def _draw_signal(
                self,
                data_x: np.ndarray,
                data_y: np.ndarray,
                range_x: tuple[float, float],
                range_y: tuple[float, float],
                graph_origin_x: int,
                graph_origin_y: int,
                color: tuple[np.uint8],
            ) -> None:
        min_x, max_x = range_x
        # max_y, min_y = range_y
        min_y, max_y = range_y
        graph_cond = (data_x >= min_x) & (data_x <= max_x) & (data_y >= min_y) & (data_y <= max_y)
        data_x_p = (data_x[graph_cond] - min_x) / (max_x - min_x) * self.plot_graph_width + graph_origin_x
        data_y_p = (data_y[graph_cond] - max_y) / (min_y - max_y) * self.plot_graph_height + graph_origin_y
        groups = itertools.groupby(np.vstack((data_x_p, data_y_p)).T, lambda k: np.all(np.isfinite(k)))
        for isfinite, group in groups:
            if not isfinite:
                continue
            data_g = np.vstack(list(group)).astype(int)
            self.plot = cv2.polylines(self.plot, [data_g], False, color, lineType=c.LINE_TYPE)

    @timeit
    def draw_signals(
                self,
                time_signals: list[tuple[np.ndarray]],
                freq_signals: list[tuple[np.ndarray]],
                correlations
            ) -> None:

        range_scale = 1.0
        timestamps, signals = np.array(time_signals).transpose([1, 0, 2])
        timestamps_range = (np.nanmin(timestamps), np.nanmax(timestamps))
        signal_range = (range_scale * np.nanmin(signals), range_scale * np.nanmax(signals))

        # frequencies, magnitudes = np.array(freq_signals).transpose([1, 0, 2])
        # frequency_range = (np.nanmin(frequencies), np.nanmax(frequencies)) if frequencies.size > 0 else (0.0, 1.0)
        # magnitude_range = (0.0, range_scale * np.nanmax(magnitudes)) if magnitudes.size > 0 else (0.0, 1.0)
        frequency_range = (c.SIGNAL_MIN_FREQUENCY, c.SIGNAL_MAX_FREQUENCY)
        magnitude_range = (0.0, 1.0)

        # lags, corrs = np.array(correlations).transpose([1, 0, 2])
        # lags_range = (np.nanmin(lags), np.nanmax(lags)) if corrs.size > 0 else (0.0, 1.0)
        # corrs_range = (range_scale * np.nanmin(corrs), range_scale * np.nanmax(corrs)) if corrs.size > 0 else (0.0, 1.0)
        lags_range = (c.CORR_MIN_LAG, c.CORR_MAX_LAG)
        corrs_range = (-1.0, 1.0)

        if c.CALC_CORRELATION:
            signal_groups = [time_signals, freq_signals, correlations]
            group_ranges = [(timestamps_range, signal_range), (frequency_range, magnitude_range), (lags_range, corrs_range)]
        else:
            signal_groups = [time_signals, freq_signals]
            group_ranges = [(timestamps_range, signal_range), (frequency_range, magnitude_range)]

        self.plot = np.ones(self.plot_shape, dtype=np.uint8) * 255
        gen = zip(signal_groups, group_ranges, self.plot_graph_origins, self.plot_graph_div_incs)
        for signal_group, (group_range_x, group_range_y), (graph_origin_x, graph_origin_y), div_inc in gen:

            corner_0 = (graph_origin_x, graph_origin_y)
            corner_1 = (graph_origin_x + self.plot_graph_width, graph_origin_y + self.plot_graph_height)
            self.plot = cv2.rectangle(self.plot, corner_0, corner_1, (0, 0, 0), lineType=c.LINE_TYPE)

            min_x, max_x = group_range_x
            min_y, max_y = group_range_y

            if not np.isnan(min_x) and not np.isnan(max_x):
                for x in np.arange(np.ceil(min_x / div_inc) * div_inc, np.ceil(max_x / div_inc) * div_inc, div_inc):
                    div_x_n = np.nan_to_num((x - min_x) / (max_x - min_x), nan=0.0, neginf=0.0, posinf=1.0)
                    div_x_p = int(div_x_n * self.plot_graph_width + graph_origin_x)
                    div_point_0 = (div_x_p, graph_origin_y)
                    div_point_1 = (div_x_p, graph_origin_y + self.plot_graph_height)
                    self.plot = cv2.line(self.plot, div_point_0, div_point_1, (224, 224, 224), lineType=c.LINE_TYPE)
                    (tw, th), _ = cv2.getTextSize(f'{x: .2f}', cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)
                    num_point = (div_x_p - tw // 2, graph_origin_y + self.plot_graph_height + th + 5)
                    self.plot = cv2.putText(self.plot, f'{x: .2f}', num_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (160, 160, 160), lineType=c.LINE_TYPE)

            if min_x <= 0.0 <= max_x:
                axis_x_n = np.nan_to_num(max_x / (max_x - min_x), nan=0.0, neginf=0.0, posinf=1.0)
                axis_x_p = int(axis_x_n * self.plot_graph_width + graph_origin_x)
                axis_point_0 = (axis_x_p, graph_origin_y)
                axis_point_1 = (axis_x_p, graph_origin_y + self.plot_graph_height)
                self.plot = cv2.line(self.plot, axis_point_0, axis_point_1, (0, 0, 0), lineType=c.LINE_TYPE)

            if min_y <= 0.0 <= max_y:
                axis_y_n = np.nan_to_num(max_y / (max_y - min_y), nan=0.0, neginf=0.0, posinf=1.0)
                axis_y_p = int(axis_y_n * self.plot_graph_height + graph_origin_y)
                axis_point_0 = (graph_origin_x, axis_y_p)
                axis_point_1 = (graph_origin_x + self.plot_graph_width, axis_y_p)
                self.plot = cv2.line(self.plot, axis_point_0, axis_point_1, (0, 0, 0), lineType=c.LINE_TYPE)

            pos_min_x = (graph_origin_x - 5, graph_origin_y + self.plot_graph_height + 15)
            pos_max_x = (graph_origin_x + self.plot_graph_width - 25, graph_origin_y + self.plot_graph_height + 15)
            pos_min_y = (graph_origin_x - 40, graph_origin_y + self.plot_graph_height - 5)
            pos_max_y = (graph_origin_x - 40, graph_origin_y + 15)
            self.plot = cv2.putText(self.plot, f'{min_x: .2f}', pos_min_x, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), lineType=c.LINE_TYPE)
            self.plot = cv2.putText(self.plot, f'{max_x: .2f}', pos_max_x, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), lineType=c.LINE_TYPE)
            self.plot = cv2.putText(self.plot, f'{min_y: .2f}', pos_min_y, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), lineType=c.LINE_TYPE)
            self.plot = cv2.putText(self.plot, f'{max_y: .2f}', pos_max_y, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), lineType=c.LINE_TYPE)

            for (data_x, data_y), color in zip(signal_group, self.plot_graph_colors):
                self._draw_signal(data_x, data_y, group_range_x, group_range_y, graph_origin_x, graph_origin_y, color)

        cv2.moveWindow('plot', 1080, 0)
        cv2.imshow('plot', self.plot)
