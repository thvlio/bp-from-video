import itertools
from enum import Enum, auto

import cv2
import numpy as np
import scipy.fft
import scipy.interpolate
import scipy.signal
import scipy.stats

from custom_profiler import timeit
from data import Location, Signal, SignalCollection, SignalYtype


class SignalColorChannel(Enum):
    GREEN = auto()
    CHROM_GREEN = auto()


class SignalProcessingMethod(Enum):
    DIFF_1 = auto()
    DIFF_2 = auto()
    INTERP_LINEAR = auto()
    INTERP_BSPLINE = auto()
    DETREND_CONST = auto()
    DETREND_LINEAR = auto()
    FILTER_BUTTER = auto()
    FILTER_FIR = auto()


class SignalSpectrumTransform(Enum):
    DFT_RFFT = auto()
    PGRAM_WELCH = auto()
    PGRAM_LS = auto()


class SignalProcessor:

    def __init__(self) -> None:

        self.color_channel = SignalColorChannel.GREEN
        self.processing_methods = [
            SignalProcessingMethod.INTERP_LINEAR,
            SignalProcessingMethod.FILTER_BUTTER
        ]
        self.spectrum_transform = SignalSpectrumTransform.PGRAM_LS

        self.min_freq = 0.8
        self.max_freq = 4.0
        self.min_mag = 0.0
        self.max_mag = 1.0

        self.butter_order = 16
        self.fir_taps = 127
        self.fir_df = 0.3

        self.calc_correlation = True
        self.min_lag = -0.5
        self.max_lag = 0.5
        self.min_corr = -1.0
        self.max_corr = 1.0

    @timeit
    def make_filter(self, signal_processing_method, sampling_freq) -> np.ndarray:
        if signal_processing_method == SignalProcessingMethod.FILTER_BUTTER:
            bands = [self.min_freq,
                     self.max_freq]
            filt = scipy.signal.butter(self.butter_order, bands, btype='bandpass', output='sos', fs=sampling_freq)
        elif signal_processing_method == SignalProcessingMethod.FILTER_FIR:
            bands = [0,
                     self.min_freq - self.fir_df,
                     self.min_freq,
                     self.max_freq,
                     self.max_freq + self.fir_df,
                     sampling_freq / 2]
            filt = scipy.signal.firls(self.fir_taps, bands, [0, 0, 1, 1, 0, 0], fs=sampling_freq)
        else:
            raise NotImplementedError
        return filt

    @timeit
    def sample_signal(self, frame: cv2.typing.MatLike, roi: Location) -> SignalYtype:
        if not np.isnan(roi).any():
            _, _, x_0, y_0, x_1, y_1 = roi
            roi_bgr = frame[y_0:y_1, x_0:x_1, :]
            if self.color_channel == SignalColorChannel.GREEN:
                values = roi_bgr[..., 1]
            elif self.color_channel == SignalColorChannel.CHROM_GREEN:
                values = roi_bgr[..., 1] / 2 - roi_bgr[..., 0] / 4 - roi_bgr[..., 2] / 4 + 0.5
            else:
                raise NotImplementedError
            value = np.mean(values)
        else:
            value = np.nan
        return value

    @timeit
    def sample_signals(self, frame: cv2.typing.MatLike, rois: list[Location]) -> list[SignalYtype]:
        return [self.sample_signal(frame, r) for r in rois]

    @timeit
    def process_signal(self, signal_raw: Signal) -> Signal:
        x, y = np.array(signal_raw.x), np.array(signal_raw.y)
        block, valid = signal_raw.v, signal_raw.w
        fs = signal_raw.get_fs()
        if valid.sum() >= 2 and np.isfinite(fs):
            for method in self.processing_methods:
                if method == SignalProcessingMethod.DIFF_1:
                    raise NotImplementedError
                    # TODO: add diff 1
                elif method == SignalProcessingMethod.DIFF_2:
                    raise NotImplementedError
                    # TODO: add diff 2
                elif method == SignalProcessingMethod.INTERP_LINEAR:
                    x_interp_block, ts = np.linspace(x[block][0], x[block][-1], block.sum(), retstep=True)
                    y_interp_block = np.interp(x_interp_block, x[valid], y[valid])
                    x[block], y[block] = x_interp_block, y_interp_block
                    valid = block
                    fs = 1 / ts
                elif method == SignalProcessingMethod.INTERP_BSPLINE:
                    raise NotImplementedError
                    # TODO: add b spline option for interp
                elif method == SignalProcessingMethod.DETREND_CONST:
                    y_detrended_valid = scipy.signal.detrend(y[valid], type='constant')
                    y[valid] = y_detrended_valid
                elif method == SignalProcessingMethod.DETREND_LINEAR:
                    y_detrended_valid = scipy.signal.detrend(y[valid], type='linear')
                    y[valid] = y_detrended_valid
                elif method == SignalProcessingMethod.FILTER_BUTTER:
                    butter = self.make_filter(method, fs)
                    default_padlen = 3 * (2 * len(butter) + 1 - min((butter[:, 2] == 0).sum(), (butter[:, 5] == 0).sum()))
                    padlen = valid.sum() - 1 if valid.sum() <= default_padlen else default_padlen
                    y_filtered_valid = scipy.signal.sosfiltfilt(butter, y[valid], padlen=padlen)
                    y[valid] = y_filtered_valid
                elif method == SignalProcessingMethod.FILTER_FIR:
                    fir = self.make_filter(method, fs)
                    default_padlen = 3 * len(fir)
                    padlen = valid.sum() - 1 if valid.sum() <= default_padlen else default_padlen
                    y_filtered_valid = scipy.signal.filtfilt(fir, 1.0, y[valid], padlen=padlen)
                    y[valid] = y_filtered_valid
                else:
                    raise NotImplementedError
        signal_proc = Signal(x, y, signal_raw.max_length)
        signal_proc.set_range()
        return signal_proc

    @timeit
    def process_signals(self, signals_raw: SignalCollection) -> SignalCollection:
        return SignalCollection(signals=[self.process_signal(s) for s in signals_raw])

    @timeit
    def transform_signal(self, signal_proc: Signal) -> Signal:
        x, y = np.array(signal_proc.x), np.array(signal_proc.y)
        valid = signal_proc.w
        fs = signal_proc.get_fs()
        if valid.sum() >= 2 and np.isfinite(fs):
            if self.spectrum_transform == SignalSpectrumTransform.DFT_RFFT:
                num_samples = len(x[valid])
                sampling_period = 1 / fs
                freqs = scipy.fft.rfftfreq(num_samples, sampling_period)
                spectrum = scipy.fft.rfft(y[valid], n=num_samples) # norm='ortho'
                mags = 2 * np.abs(spectrum) / num_samples
            elif self.spectrum_transform == SignalSpectrumTransform.PGRAM_WELCH:
                freqs, pgram = scipy.signal.welch(y[valid], fs)
                mags = pgram
            elif self.spectrum_transform == SignalSpectrumTransform.PGRAM_LS:
                num_samples = len(x[valid])
                freqs = np.linspace(self.min_freq, self.max_freq, num_samples)
                pgram = scipy.signal.lombscargle(x[valid], y[valid], freqs=freqs*2*np.pi, floating_mean=True, normalize=True)
                mags = pgram
            else:
                raise NotImplementedError
        else:
            freqs, mags = [], []
        signal_spectrum = Signal(freqs, mags, max_length=len(freqs))
        signal_spectrum.set_range((self.min_freq, self.max_freq), (self.min_mag, self.max_mag))
        return signal_spectrum

    @timeit
    def transform_signals(self, signals_proc: SignalCollection) -> SignalCollection:
        return SignalCollection(signals=[self.transform_signal(s) for s in signals_proc])

    @timeit
    def correlate_signal_pair(self, signal_a: Signal, signal_b: Signal) -> Signal:
        x_a, y_a = np.array(signal_a.x), np.array(signal_a.y)
        _, y_b = np.array(signal_b.x), np.array(signal_b.y)
        valid = (signal_a.w) & (signal_b.w)
        if valid.sum() >= 2:
            corr = scipy.signal.correlate(y_a[valid], y_b[valid])
            corr /= np.max([np.dot(y_a[valid], y_a[valid]),
                            np.dot(y_b[valid], y_b[valid]),
                            np.dot(y_a[valid], y_b[valid])])
            lag_indices = scipy.signal.correlation_lags(valid.sum(), valid.sum())
            lags = (x_a[valid][-1] - x_a[valid][::-1])[np.abs(lag_indices)] * np.sign(lag_indices)
        else:
            lags, corr = [], []
        signal_corr = Signal(lags, corr, max_length=len(lags))
        signal_corr.set_range((self.min_lag, self.max_lag), (self.min_corr, self.max_corr))
        return signal_corr

    @timeit
    def correlate_signals(self, signals_proc: SignalCollection) -> SignalCollection:
        return SignalCollection(signals=[self.correlate_signal_pair(s_a, s_b) for s_a, s_b in itertools.combinations(signals_proc, 2)])
