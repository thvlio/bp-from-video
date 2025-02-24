import itertools
import math
from collections import deque
from typing import Any

import cv2
import numpy as np
import scipy.fft
import scipy.interpolate
import scipy.signal
import scipy.stats

import config as c
from custom_profiler import timeit


class SignalProcessor:

    @staticmethod
    def create_deques(
                num_deques: int = 1,
                max_length: int | None = None,
                filled: bool = True,
                fill_value: Any = np.nan
            ) -> list[deque]:
        values = [fill_value] * max_length if filled else []
        return [deque(values, max_length) for _ in range(num_deques)]

    def __init__(
                self,
            ) -> None:
        self.num_signals = len(c.ROI_LANDMARK_INDICES)
        self.timestamps, = self.create_deques(1, c.SIGNAL_MAX_SAMPLES)
        self.signals_raw = self.create_deques(self.num_signals, c.SIGNAL_MAX_SAMPLES)
        self.signals_proc = self.create_deques(self.num_signals, c.SIGNAL_MAX_SAMPLES)
        self.roi_positions = self.create_deques(self.num_signals, c.ROI_POS_MAX_SAMPLES, fill_value=(np.nan, np.nan))
        self.roi_bboxes = self.create_deques(self.num_signals, c.ROI_POS_MAX_SAMPLES, fill_value=(np.nan, np.nan, np.nan, np.nan))
        self.landmark_variations = self.create_deques(c.HEATMAP_POINTS, c.SIGNAL_MAX_SAMPLES)
        self.frequencies = [[]] * self.num_signals
        self.magnitudes = [[]] * self.num_signals
        self.peak_freqs = self.create_deques(self.num_signals, c.SIGNAL_MAX_SAMPLES)
        self.peak_lags = self.create_deques(math.comb(self.num_signals, 2), c.SIGNAL_MAX_SAMPLES)
        self.sos = scipy.signal.butter(c.BUTTER_ORDER, [c.SIGNAL_MIN_FREQUENCY, c.SIGNAL_MAX_FREQUENCY], btype='bandpass', output='sos', fs=c.BUTTER_FS)

    @timeit
    def update_heatmap(
                self,
                frame: cv2.typing.MatLike,
                points: np.ndarray
            ) -> list[float]:
        variations = []
        for k, (x_p, y_p) in enumerate(points):
            padding = 10
            surrounding = frame[y_p-padding:y_p+padding, x_p-padding:x_p+padding, 1]
            value = np.mean(surrounding)
            self.landmark_variations[k].append(value)
            variation = scipy.stats.variation(self.landmark_variations[k])
            variations.append(variation)
        return variations

    @timeit
    def detrend_signal(
                self,
                signal: deque,
                method: str
            ) -> deque:
        signal_notnan = np.array(signal)[~np.isnan(signal)]
        if signal_notnan.size == 0:
            signal_notnan = [0]
        signal_detrended_notnan = scipy.signal.detrend(signal_notnan, type=method)
        signal_detrended = np.array([np.nan] * len(signal))
        signal_detrended[~np.isnan(signal)] = signal_detrended_notnan
        return deque(signal_detrended, maxlen=c.SIGNAL_MAX_SAMPLES)

    @timeit
    def filter_signal(
                self,
                signal: deque
            ) -> deque:
        signal_notnan = np.array(signal)[~np.isnan(signal)]
        if signal_notnan.size == 0:
            signal_notnan = [0]
        default_padlen = 3 * (2 * len(self.sos) + 1 - min((self.sos[:, 2] == 0).sum(), (self.sos[:, 5] == 0).sum()))
        padlen = len(signal_notnan) - 1 if len(signal_notnan) <= default_padlen else default_padlen
        signal_filtered_notnan = scipy.signal.sosfiltfilt(self.sos, signal_notnan, padlen=padlen)
        signal_filtered = np.array([np.nan] * len(signal))
        signal_filtered[~np.isnan(signal)] = signal_filtered_notnan
        return deque(signal_filtered, maxlen=c.SIGNAL_MAX_SAMPLES)

    @timeit
    def get_dft(
                self,
                signal: deque
            ) -> tuple[np.ndarray, np.ndarray, float]:
        mean_period = np.nanmean(np.diff(self.timestamps))
        signal_notnan = np.array(signal)[~np.isnan(signal)]
        if np.isnan(mean_period) or signal_notnan.size == 0:
            mean_period = 1
            signal_notnan = [0]
        timestamps_notnan = np.array(self.timestamps)[~np.isnan(signal)]
        if timestamps_notnan.size == 0:
            timestamps_notnan = [0]
        new_timestamps = np.linspace(timestamps_notnan[0], timestamps_notnan[-1])
        new_signal = np.interp(new_timestamps, timestamps_notnan, signal_notnan)
        period = (self.timestamps[-1] - self.timestamps[0]) / len(self.timestamps)
        freqs = scipy.fft.rfftfreq(c.SIGNAL_MAX_SAMPLES, period)
        spectrum = scipy.fft.rfft(new_signal, n=c.SIGNAL_MAX_SAMPLES, norm='ortho')
        mags = 2 * np.abs(spectrum) / len(new_signal)
        freqs_f = freqs[(freqs >= c.SIGNAL_MIN_FREQUENCY) & (freqs <= c.SIGNAL_MAX_FREQUENCY)]
        mags_f = mags[(freqs >= c.SIGNAL_MIN_FREQUENCY) & (freqs <= c.SIGNAL_MAX_FREQUENCY)]
        peak_freq = freqs_f[np.argmax(mags_f)] if len(freqs_f) > 0 else np.nan
        return freqs, mags, peak_freq

    @timeit
    def get_ls_pgram(
                self,
                signal: deque
            ) -> tuple[np.ndarray, np.ndarray, float]:
        freqs = np.arange(c.SIGNAL_MIN_FREQUENCY, c.SIGNAL_MAX_FREQUENCY, 0.02)
        timestamps_notnan = np.array(self.timestamps)[~np.isnan(signal)]
        signal_notnan = np.array(signal)[~np.isnan(signal)]
        if timestamps_notnan.size == 0 or signal_notnan.size == 0:
            timestamps_notnan = [0]
            signal_notnan = [0]
        pgram = scipy.signal.lombscargle(timestamps_notnan, signal_notnan, freqs=freqs*2*np.pi, floating_mean=True, normalize=True)
        peak_freq = freqs[np.argmax(pgram)] if not np.all(np.isnan(freqs)) else np.nan
        return freqs, pgram, peak_freq

    @timeit
    def get_corr(
                self,
                signal_0: deque,
                signal_1: deque
            ) -> tuple[np.ndarray, np.ndarray, float]:
        notnan = (~np.isnan(signal_0)) & (~np.isnan(signal_1))
        signal_0_notnan = np.array(signal_0)[notnan]
        signal_1_notnan = np.array(signal_1)[notnan]
        if signal_0_notnan.size == 0 or signal_1_notnan.size == 0:
            signal_0_notnan = np.zeros((1,))
            signal_1_notnan = np.zeros((1,))
        corr = scipy.signal.correlate(signal_0_notnan, signal_1_notnan)
        corr /= np.max([np.dot(signal_0_notnan, signal_0_notnan),
                        np.dot(signal_1_notnan, signal_1_notnan),
                        np.dot(signal_0_notnan, signal_1_notnan)])
        lag_indices = scipy.signal.correlation_lags(signal_0_notnan.size, signal_1_notnan.size)
        lags = (self.timestamps[-1] - np.array(self.timestamps)[::-1])[np.abs(lag_indices)] * np.sign(lag_indices)
        lags_f = lags[(lags >= c.CORR_MIN_LAG) & (lags <= c.CORR_MAX_LAG)]
        corr_f = corr[(lags >= c.CORR_MIN_LAG) & (lags <= c.CORR_MAX_LAG)]
        peak_lag = lags_f[np.argmax(corr_f)] if corr_f.size > 1 and lags_f.size > 1 else np.nan
        return lags, corr, peak_lag

    @timeit
    def update_signals(
                self,
                frame: cv2.typing.MatLike,
                timestamp: float,
                landmark_collections: tuple
            ) -> tuple[list]:

        self.timestamps.append(timestamp)

        time_signals = []
        freq_signals = []

        mean_roi_positions = []
        mean_roi_bboxes = []
        mean_peak_freqs = []
        mean_peak_lags = []

        gen = zip(landmark_collections, c.ROI_LANDMARK_INDICES, c.ROI_LANDMARK_CONFIGS)
        for s, (landmarks, landmark_indices, (left_m, top_m, right_m, bottom_m)) in enumerate(gen):

            if landmarks is not None:
                _, points, (bbox_height, bbox_width) = landmarks
                x_roi, y_roi = np.squeeze(np.mean([points[i] for i in landmark_indices], axis=0))
                self.roi_positions[s].append((x_roi, y_roi))
                roi_positions_notnan = np.array(self.roi_positions[s])[~np.isnan(self.roi_positions[s]).any(axis=1)]
                x_f, y_f = np.squeeze(np.mean(roi_positions_notnan, axis=0))
                mean_roi_positions.append((x_f, y_f))
                x_0 = int(x_f + left_m * bbox_width)
                y_0 = int(y_f + top_m * bbox_height)
                x_1 = int(x_f + right_m * bbox_width)
                y_1 = int(y_f + bottom_m * bbox_height)
                mean_roi_bboxes.append((x_0, y_0, x_1, y_1))
                roi_bgr = frame[y_0:y_1, x_0:x_1, :]
                if c.PPG_PIXEL_VALUE == c.PPGPixelValue.G:
                    values = roi_bgr[..., 1]
                elif c.PPG_PIXEL_VALUE == c.PPGPixelValue.CG:
                    values = roi_bgr[..., 1] / 2 - roi_bgr[..., 0] / 4 - roi_bgr[..., 2] / 4 + 0.5
                else:
                    raise NotImplementedError
                value = np.mean(values)
                self.signals_raw[s].append(value)
            else:
                self.roi_positions[s].append((np.nan, np.nan))
                mean_roi_positions.append((np.nan, np.nan))
                mean_roi_bboxes.append((np.nan, np.nan, np.nan, np.nan))
                self.signals_raw[s].append(np.nan)

            if c.SIGNAL_PROCESSING_METHOD == c.SignalProcessingMethod.RAW:
                self.signals_proc[s] = self.signals_raw[s]
            elif c.SIGNAL_PROCESSING_METHOD == c.SignalProcessingMethod.CONSTANT:
                self.signals_proc[s] = self.detrend_signal(self.signals_raw[s], 'constant')
            elif c.SIGNAL_PROCESSING_METHOD == c.SignalProcessingMethod.LINEAR:
                self.signals_proc[s] = self.detrend_signal(self.signals_raw[s], 'linear')
            elif c.SIGNAL_PROCESSING_METHOD == c.SignalProcessingMethod.BUTTER:
                self.signals_proc[s] = self.filter_signal(self.signals_raw[s])
            else:
                raise NotImplementedError

            time_signals.append((np.array(self.timestamps), np.array(self.signals_proc[s])))

            if c.SPECTRUM_TRANSFORM == c.SpectrumTransform.DFT:
                self.frequencies[s], self.magnitudes[s], peak_freq = self.get_dft(self.signals_proc[s])
            elif c.SPECTRUM_TRANSFORM == c.SpectrumTransform.LS_PGRAM:
                self.frequencies[s], self.magnitudes[s], peak_freq = self.get_ls_pgram(self.signals_proc[s])
            else:
                raise NotImplementedError

            self.peak_freqs[s].append(peak_freq)
            mean_peak_freq = np.nanmean(self.peak_freqs[s])
            mean_peak_freqs.append(mean_peak_freq)

            freq_signals.append((self.frequencies[s], self.magnitudes[s]))

        correlations = []
        if c.CALC_CORRELATION:
            for s, ((_, signal_0), (_, signal_1)) in enumerate(itertools.combinations(time_signals, 2)):
                lags, corr, peak_lag = self.get_corr(signal_0, signal_1)
                correlations.append((lags, corr))
                self.peak_lags[s].append(peak_lag)
                mean_peak_lag = np.nanmean(self.peak_lags[s])
                mean_peak_lags.append(mean_peak_lag)

        return time_signals, freq_signals, mean_peak_freqs, mean_roi_positions, mean_roi_bboxes, correlations, mean_peak_lags
