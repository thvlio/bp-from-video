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
        self.roi_filtered = self.create_deques(self.num_signals, c.ROI_POS_MAX_SAMPLES, fill_value=(np.nan, np.nan))
        self.roi_bboxes = self.create_deques(self.num_signals, c.ROI_POS_MAX_SAMPLES, fill_value=(np.nan, np.nan, np.nan, np.nan))
        self.landmark_variations = self.create_deques(c.HEATMAP_POINTS, c.SIGNAL_MAX_SAMPLES)
        self.frequencies = [[]] * self.num_signals
        self.magnitudes = [[]] * self.num_signals
        # self.peak_freqs = [np.nan] * self.num_signals
        self.peak_freqs = self.create_deques(self.num_signals, c.SIGNAL_MAX_SAMPLES)
        self.peak_freqs_filtered = self.create_deques(self.num_signals, c.SIGNAL_MAX_SAMPLES)
        self.peak_lags = self.create_deques(math.comb(self.num_signals, 2), c.SIGNAL_MAX_SAMPLES)
        self.peak_lags_filtered = self.create_deques(math.comb(self.num_signals, 2), c.SIGNAL_MAX_SAMPLES)

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
            ) -> None:
        signal_notnan = np.array(signal)[~np.isnan(signal)]
        if signal_notnan.size == 0:
            signal_notnan = [0]
        signal_detrended_notnan = scipy.signal.detrend(signal_notnan, type=method)
        signal_detrended = np.array([np.nan] * len(signal))
        signal_detrended[~np.isnan(signal)] = signal_detrended_notnan
        return deque(signal_detrended, maxlen=c.SIGNAL_MAX_SAMPLES)

    @timeit
    def get_dft(
                self,
                signal: deque
            ) -> tuple[np.ndarray | float]:

        mean_period = np.nanmean(np.diff(self.timestamps))
        signal_notnan = np.array(signal)[~np.isnan(signal)]
        if np.isnan(mean_period) or signal_notnan.size == 0:
            mean_period = 1
            signal_notnan = [0]

        # freqs = scipy.fft.rfftfreq(c.SIGNAL_MAX_SAMPLES, mean_period)
        # spectrum = scipy.fft.rfft(signal_notnan, n=c.SIGNAL_MAX_SAMPLES)
        # mags = 2 * np.abs(spectrum) / len(signal_notnan)

        # opt_len = scipy.fft.next_fast_len(len(signal_notnan))
        # freqs = scipy.fft.rfftfreq(opt_len, mean_period)
        # spectrum = scipy.fft.rfft(signal_notnan, opt_len)
        # mags = 2 * np.abs(spectrum) / opt_len

        timestamps_notnan = np.array(self.timestamps)[~np.isnan(signal)]
        if timestamps_notnan.size == 0:
            timestamps_notnan = [0]
        new_timestamps = np.linspace(timestamps_notnan[0], timestamps_notnan[-1])
        new_signal = np.interp(new_timestamps, timestamps_notnan, signal_notnan)

        period = (self.timestamps[-1] - self.timestamps[0]) / len(self.timestamps)
        freqs = scipy.fft.rfftfreq(c.SIGNAL_MAX_SAMPLES, period)
        spectrum = scipy.fft.rfft(new_signal, n=c.SIGNAL_MAX_SAMPLES)
        mags = 2 * np.abs(spectrum) / len(new_signal)

        freqs_f = freqs[(freqs >= c.FFT_MIN_FREQUENCY) & (freqs <= c.FFT_MAX_FREQUENCY)]
        mags_f = mags[(freqs >= c.FFT_MIN_FREQUENCY) & (freqs <= c.FFT_MAX_FREQUENCY)]
        peak_freq = freqs_f[np.argmax(mags_f)] if len(freqs_f) > 0 else np.nan

        return freqs, mags, peak_freq

    @timeit
    def get_pgram(
                self,
                signal: deque
            ) -> tuple[np.ndarray | float]:
        # opt_len = scipy.fft.next_fast_len(len(signal_notnan))
        # freqs = np.arange(c.FFT_MIN_FREQUENCY, c.FFT_MAX_FREQUENCY, 0.05) * 2 * np.pi
        freqs = np.arange(c.FFT_MIN_FREQUENCY, c.FFT_MAX_FREQUENCY, 0.02)
        timestamps_notnan = np.array(self.timestamps)[~np.isnan(signal)]
        signal_notnan = np.array(signal)[~np.isnan(signal)]
        if timestamps_notnan.size == 0 or signal_notnan.size == 0:
            timestamps_notnan = [0]
            signal_notnan = [0]
        pgram = scipy.signal.lombscargle(timestamps_notnan, signal_notnan, freqs=freqs*2*np.pi, floating_mean=True, normalize='normalize')
        peak_freq = freqs[np.argmax(pgram)] if not np.all(np.isnan(freqs)) else np.nan
        return freqs, pgram, peak_freq

    @timeit
    def get_corr(
                self,
                signal_0: deque,
                signal_1: deque
            ) -> tuple[np.ndarray | float]:
        notnan = (~np.isnan(signal_0)) & (~np.isnan(signal_1))
        signal_0_notnan = np.array(signal_0)[notnan]
        signal_1_notnan = np.array(signal_1)[notnan]
        if signal_0_notnan.size == 0 or signal_1_notnan.size == 0:
            signal_0_notnan = np.zeros((1,))
            signal_1_notnan = np.zeros((1,))
        corr = scipy.signal.correlate(signal_0_notnan, signal_1_notnan)
        lags = scipy.signal.correlation_lags(signal_0_notnan.size, signal_1_notnan.size)
        lags = (self.timestamps[-1] - np.array(self.timestamps)[::-1])[np.abs(lags)] * np.sign(lags)
        # peak_lag = lags[np.argmax(corr)] if corr.size > 1 and lags.size > 1 else np.nan
        lags_f = lags[(lags >= -1) & (lags <= 1)]
        corr_f = corr[(lags >= -1) & (lags <= 1)]
        peak_lag = lags_f[np.argmax(corr_f)] if corr_f.size > 1 and lags_f.size > 1 else np.nan
        return lags, corr, peak_lag

    @timeit
    def update_signals(
                self,
                frame: cv2.typing.MatLike,
                timestamp: float,
                landmark_collections: tuple,
                # person_masks: list[np.ndarray, list[np.ndarray]]
            ) -> list[list | deque | np.ndarray]:

        self.timestamps.append(timestamp)

        time_signals = []
        freq_signals = []

        gen = zip(landmark_collections, c.ROI_LANDMARK_INDICES, c.ROI_LANDMARK_CONFIGS)
        for s, (landmarks, landmark_indices, (left_m, top_m, right_m, bottom_m)) in enumerate(gen):

            if landmarks is not None:
                _, points, (bbox_height, bbox_width) = landmarks
                x_roi, y_roi = np.squeeze(np.mean([points[i] for i in landmark_indices], axis=0))
                self.roi_positions[s].append((x_roi, y_roi))
                roi_positions_notnan = np.array(self.roi_positions[s])[~np.isnan(self.roi_positions[s]).any(axis=1)]
                x_f, y_f = np.squeeze(np.mean(roi_positions_notnan, axis=0))
                self.roi_filtered[s].append((x_f, y_f))
                x_0 = int(x_f + left_m * bbox_width)
                y_0 = int(y_f + top_m * bbox_height)
                x_1 = int(x_f + right_m * bbox_width)
                y_1 = int(y_f + bottom_m * bbox_height)
                self.roi_bboxes[s].append((x_0, y_0, x_1, y_1))
                roi = frame[y_0:y_1, x_0:x_1, 1]
                value = np.mean(roi)
                self.signals_raw[s].append(value)
            else:
                self.roi_positions[s].append((np.nan, np.nan))
                self.roi_filtered[s].append((np.nan, np.nan))
                self.roi_bboxes[s].append((np.nan, np.nan, np.nan, np.nan))
                self.signals_raw[s].append(np.nan)

            # time_signals.append((np.array(self.timestamps), np.array(self.signals_raw[s])))

            # self.signals_proc[s] = self.signals_raw[s]
            self.signals_proc[s] = self.detrend_signal(self.signals_raw[s], 'linear')
            # self.signals_proc[s] = self.detrend_signal(self.signals_raw[s], 'constant')

            time_signals.append((np.array(self.timestamps), np.array(self.signals_proc[s])))

            self.frequencies[s], self.magnitudes[s], peak_freq = self.get_pgram(self.signals_proc[s])
            # self.frequencies[s], self.magnitudes[s], peak_freq = self.get_dft(self.signals_proc[s])

            self.peak_freqs[s].append(peak_freq)
            # peak_freqs_notnan = np.array(self.peak_freqs[s])[~np.isnan(self.peak_freqs[s])]
            # mean_peak_freq = np.mean(peak_freqs_notnan)
            mean_peak_freq = np.nanmean(self.peak_freqs[s])
            self.peak_freqs_filtered[s].append(mean_peak_freq)

            freq_signals.append((self.frequencies[s], self.magnitudes[s]))

        correlations = []
        for s, ((_, signal_0), (_, signal_1)) in enumerate(itertools.combinations(time_signals, 2)):
            lags, corr, peak_lag = self.get_corr(signal_0, signal_1)
            correlations.append((lags, corr))
            self.peak_lags[s].append(peak_lag)
            mean_peak_lag = np.nanmean(self.peak_lags[s])
            self.peak_lags_filtered[s].append(mean_peak_lag)

        return time_signals, freq_signals, self.peak_freqs_filtered, correlations, self.peak_lags_filtered
