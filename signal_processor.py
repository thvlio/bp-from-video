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

from config import Config as c
from custom_profiler import timeit


class SignalProcessor:

    @staticmethod
    def _create_deques(
                num_deques: int = 1,
                max_length: int | None = None,
                filled: bool = True,
                fill_value: Any = np.nan
            ) -> list[deque[Any]]:
        values = [fill_value] * max_length if filled else []
        return [deque(values, max_length) for _ in range(num_deques)]

    def __init__(
                self,
            ) -> None:
        self.num_signals = len(c.SIGNAL_LOCATION_CONFIGS)
        self.timestamps, = self._create_deques(1, c.SIGNAL_MAX_SAMPLES)
        self.signals_raw = self._create_deques(self.num_signals, c.SIGNAL_MAX_SAMPLES)
        self.signals_proc = self._create_deques(self.num_signals, c.SIGNAL_MAX_SAMPLES)
        self.signal_pois = self._create_deques(self.num_signals, c.ROI_POS_MAX_SAMPLES, fill_value=(np.nan, np.nan))
        self.signal_rois = self._create_deques(self.num_signals, c.ROI_POS_MAX_SAMPLES, fill_value=(np.nan, np.nan, np.nan, np.nan))
        self.frequencies = [[]] * self.num_signals
        self.magnitudes = [[]] * self.num_signals
        self.peak_freqs = self._create_deques(self.num_signals, c.SIGNAL_MAX_SAMPLES)
        self.peak_lags = self._create_deques(math.comb(self.num_signals, 2), c.SIGNAL_MAX_SAMPLES)
        if c.SIGNAL_PROCESSING_METHOD == c.SignalProcessingMethod.BUTTER:
            bands = [c.SIGNAL_MIN_FREQUENCY, c.SIGNAL_MAX_FREQUENCY]
            self.butter = scipy.signal.butter(c.BUTTER_ORDER, bands, btype='bandpass', output='sos', fs=c.BUTTER_FS)
        elif c.SIGNAL_PROCESSING_METHOD == c.SignalProcessingMethod.FIR:
            bands = [0, c.SIGNAL_MIN_FREQUENCY - c.FIR_DF, c.SIGNAL_MIN_FREQUENCY, c.SIGNAL_MAX_FREQUENCY, c.SIGNAL_MAX_FREQUENCY + c.FIR_DF, c.FIR_FS / 2]
            self.fir = scipy.signal.firls(c.FIR_TAPS, bands, [0, 0, 1, 1, 0, 0], fs=c.FIR_FS)

    # TODO: get dft and pgram on the same function

    # TODO: clean update signals

    # TODO: add option to get uniform sample rate signal
    #       use splines to interpolate to periodic timestamps

    # TODO: also process 1st and 2nd derivatives of the signal

    @timeit
    def filter_signal(
                self,
                signal: deque[float]
            ) -> deque[float]:
        signal_notnan = np.array(signal)[~np.isnan(signal)]
        signal_filtered = np.array([np.nan] * len(signal))
        if signal_notnan.size > 0:
            if c.SIGNAL_PROCESSING_METHOD == c.SignalProcessingMethod.RAW:
                signal_filtered = signal
            else:
                if c.SIGNAL_PROCESSING_METHOD == c.SignalProcessingMethod.CONST:
                    signal_filtered_notnan = scipy.signal.detrend(signal_notnan, type='constant')
                elif c.SIGNAL_PROCESSING_METHOD == c.SignalProcessingMethod.LINEAR:
                    signal_filtered_notnan = scipy.signal.detrend(signal_notnan, type='linear')
                elif c.SIGNAL_PROCESSING_METHOD == c.SignalProcessingMethod.BUTTER:
                    default_padlen = 3 * (2 * len(self.butter) + 1 - min((self.butter[:, 2] == 0).sum(), (self.butter[:, 5] == 0).sum()))
                    padlen = signal_notnan.size - 1 if signal_notnan.size <= default_padlen else default_padlen
                    signal_filtered_notnan = scipy.signal.sosfiltfilt(self.butter, signal_notnan, padlen=padlen)
                elif c.SIGNAL_PROCESSING_METHOD == c.SignalProcessingMethod.FIR:
                    default_padlen = 3 * len(self.fir)
                    padlen = signal_notnan.size - 1 if signal_notnan.size <= default_padlen else default_padlen
                    signal_filtered_notnan = scipy.signal.filtfilt(self.fir, 1, signal_notnan, padlen=padlen)
                else:
                    raise NotImplementedError
                signal_filtered[~np.isnan(signal)] = signal_filtered_notnan
        return deque(signal_filtered, maxlen=c.SIGNAL_MAX_SAMPLES)

    @timeit
    def get_dft(
                self,
                signal: deque[float]
            ) -> tuple[np.ndarray[float], np.ndarray[float], float]:
        signal_notnan = np.array(signal)[~np.isnan(signal)]
        timestamps_notnan = np.array(self.timestamps)[~np.isnan(signal)]
        if signal_notnan.size == 0:
            signal_notnan = [0]
        if timestamps_notnan.size == 0:
            timestamps_notnan = [0]
        new_timestamps = np.linspace(timestamps_notnan[0], timestamps_notnan[-1])
        new_signal = np.interp(new_timestamps, timestamps_notnan, signal_notnan)
        mean_period = (self.timestamps[-1] - self.timestamps[0]) / len(self.timestamps)
        freqs = scipy.fft.rfftfreq(c.SIGNAL_MAX_SAMPLES, mean_period)
        spectrum = scipy.fft.rfft(new_signal, n=c.SIGNAL_MAX_SAMPLES, norm='ortho')
        mags = 2 * np.abs(spectrum) / len(new_signal)
        freqs_f = freqs[(freqs >= c.SIGNAL_MIN_FREQUENCY) & (freqs <= c.SIGNAL_MAX_FREQUENCY)]
        mags_f = mags[(freqs >= c.SIGNAL_MIN_FREQUENCY) & (freqs <= c.SIGNAL_MAX_FREQUENCY)]
        peak_freq = freqs_f[np.argmax(mags_f)] if len(freqs_f) > 0 else np.nan
        return freqs, mags, peak_freq

    @timeit
    def get_ls_pgram(
                self,
                signal: deque[float]
            ) -> tuple[np.ndarray[float], np.ndarray[float], float]:
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
                signal_0: deque[float],
                signal_1: deque[float]
            ) -> tuple[np.ndarray[float], np.ndarray[float], float]:
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

    # TODO: update_rois, sample_signal, filter_signal, transform_signal, correlate_signals

    # TODO: figure how to reinclude this inside update signals
    #       or even maybe I should split update signal into its parts to keep it small
    @timeit
    def update_rois(
                self,
                inference_results: list[tuple[c.ModelType, c.Detections | c.Masks]]
            ) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]]]:
        _, (_, face_landmarks), (_, hand_landmarks), _ = inference_results
        mean_pois = []
        mean_rois = []
        # TODO: same as below
        for s in range(self.num_signals):
            model_type, landmark_indices, (left_m, top_m, right_m, bottom_m) = c.SIGNAL_LOCATION_CONFIGS[s]
            if model_type is c.ModelType.FACE_LANDMARKER:
                landmarks = face_landmarks
            elif model_type is c.ModelType.HAND_LANDMARKER:
                landmarks = hand_landmarks
            else:
                raise NotImplementedError
            if len(landmarks) > 0:
                bbox, points = landmarks[0]
                x_p, y_p = np.squeeze(np.mean([points[i] for i in landmark_indices], axis=0).round().astype(int))
                self.signal_pois[s].append((x_p, y_p))
                x_f, y_f = np.squeeze(np.nanmean(self.signal_pois[s], axis=0).round().astype(int))
                mean_pois.append((x_f, y_f))
                x_r_0 = int(round(x_p + left_m * (bbox[2] - bbox[0])))
                y_r_0 = int(round(y_p + top_m * (bbox[3] - bbox[1])))
                x_r_1 = int(round(x_p + right_m * (bbox[2] - bbox[0])))
                y_r_1 = int(round(y_p + bottom_m * (bbox[3] - bbox[1])))
                self.signal_rois[s].append((x_r_0, y_r_0, x_r_1, y_r_1))
                x_f_0, y_f_0, x_f_1, y_f_1 = np.squeeze(np.nanmean(self.signal_rois[s], axis=0).round().astype(int))
                mean_rois.append((x_f_0, y_f_0, x_f_1, y_f_1))
            else:
                self.signal_pois[s].append((np.nan, np.nan))
                mean_pois.append((np.nan, np.nan))
                self.signal_rois[s].append((np.nan, np.nan, np.nan, np.nan))
                mean_rois.append((np.nan, np.nan, np.nan, np.nan))
        return mean_pois, mean_rois

    @timeit
    def update_signals(
                self,
                frame: cv2.typing.MatLike,
                timestamp: float,
                mean_pois: list[tuple[int, int]],
                mean_rois: list[tuple[int, int, int, int]]
            ) -> tuple[list[c.SignalData], list[c.SignalData], list[c.SignalData], list[float], list[float]]:

        self.timestamps.append(timestamp)

        time_signals = []
        freq_signals = []

        mean_peak_freqs = []
        mean_peak_lags = []

        # TODO: try and unpack a zip instead of using an index

        for s in range(self.num_signals):

            # TODO: move this sampling to another function

            if not np.isnan(mean_rois[s]).any():
                x_f_0, y_f_0, x_f_1, y_f_1 = mean_rois[s]
                roi_bgr = frame[y_f_0:y_f_1, x_f_0:x_f_1, :]
                if c.SIGNAL_COLOR_CHANNEL == c.SignalColorChannel.G:
                    values = roi_bgr[..., 1]
                elif c.SIGNAL_COLOR_CHANNEL == c.SignalColorChannel.CG:
                    values = roi_bgr[..., 1] / 2 - roi_bgr[..., 0] / 4 - roi_bgr[..., 2] / 4 + 0.5
                else:
                    raise NotImplementedError
                value = np.mean(values)
                self.signals_raw[s].append(value)
            else:
                self.signals_raw[s].append(np.nan)

            self.signals_proc[s] = self.filter_signal(self.signals_raw[s])

            time_signals.append((np.array(self.timestamps), np.array(self.signals_proc[s])))

            if c.SIGNAL_SPECTRUM_TRANSFORM == c.SignalSpectrumTransform.DFT:
                self.frequencies[s], self.magnitudes[s], peak_freq = self.get_dft(self.signals_proc[s])
            elif c.SIGNAL_SPECTRUM_TRANSFORM == c.SignalSpectrumTransform.LS_PGRAM:
                self.frequencies[s], self.magnitudes[s], peak_freq = self.get_ls_pgram(self.signals_proc[s])
            else:
                raise NotImplementedError

            self.peak_freqs[s].append(peak_freq)
            mean_peak_freqs.append(np.nanmean(self.peak_freqs[s]))

            freq_signals.append((self.frequencies[s], self.magnitudes[s]))

        correlations = []
        if c.CALC_CORRELATION:
            for s, ((_, signal_0), (_, signal_1)) in enumerate(itertools.combinations(time_signals, 2)):
                lags, corr, peak_lag = self.get_corr(signal_0, signal_1)
                correlations.append((lags, corr))
                self.peak_lags[s].append(peak_lag)
                mean_peak_lags.append(np.nanmean(self.peak_lags[s]))

        return time_signals, freq_signals, correlations, mean_peak_freqs, mean_peak_lags
