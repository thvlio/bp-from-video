import copy
import itertools
import math

import cv2
import numpy as np
import scipy.fft
import scipy.interpolate
import scipy.signal

from config import Config as c
from custom_profiler import profiler
from signal_data import Signal, SignalCollection


class SignalManager:

    # TODO: call this something else
    #       its not a manager, just a bunch of collections
    #       after renaming SignalCollection to SignalGroup
    #           try SignalGroupCollection
    #           or SignalData (but then rename signal_data.py)
    #           also try SignalStore (preferred?)

    # TODO: return copy of itself as result in run pipeline
    #       after the above change

    # TODO: make sure run pipeline uses inference runner results class

    def __init__(self,
                 num_signals: int = len(c.SELECTED_ROI_CONFIGS),
                 roi_max_samples: int = c.ROI_MAX_SAMPLES,
                 signal_max_samples: int = c.SIGNAL_MAX_SAMPLES,
                 peak_max_samples: int = c.PEAK_MAX_SAMPLES) -> None:
        self.sc_roi = SignalCollection(num_signals, y0=(np.nan,)*6, s_maxlen=roi_max_samples)
        self.sc_raw = SignalCollection(num_signals, s_maxlen=signal_max_samples)
        self.sc_proc = SignalCollection(num_signals)
        self.sc_spec = SignalCollection(num_signals)
        self.sc_corr = SignalCollection(math.comb(num_signals, 2))
        self.sc_bpm = SignalCollection(num_signals, s_maxlen=peak_max_samples)
        self.sc_ptt = SignalCollection(math.comb(num_signals, 2), s_maxlen=peak_max_samples)

    def copy_signal_collections(self) -> tuple[SignalCollection, ...]:
        return copy.deepcopy(vars(self).values())


class SignalProcessor:

    def __init__(self,
                 num_signals: int = len(c.SELECTED_ROI_CONFIGS),
                 roi_max_samples: int = c.ROI_MAX_SAMPLES,
                 signal_max_samples: int = c.SIGNAL_MAX_SAMPLES,
                 peak_max_samples: int = c.PEAK_MAX_SAMPLES,
                 *,
                 color_channel: c.SignalColorChannel = c.SIGNAL_COLOR_CHANNEL,
                 processing_methods: list[c.SignalProcessingMethod] | None = None,
                 spectrum_transform: c.SignalSpectrumTransform = c.SIGNAL_SPECTRUM_TRANSFORM,
                 butter_order: int = c.FILTER_BUTTER_ORDER,
                 butter_min_bw: float = c.FILTER_BUTTER_MIN_BW,
                 fir_taps: int = c.FILTER_FIR_TAPS,
                 fir_df: float = c.FILTER_FIR_DF,
                 min_freq: float = c.FILTER_MIN_FREQ,
                 max_freq: float = c.FILTER_MAX_FREQ,
                 min_mag: float = c.SPECTRUM_MIN_MAG,
                 max_mag: float = c.SPECTRUM_MAX_MAG,
                 min_lag: float = c.SIGNALS_MIN_LAG,
                 max_lag: float = c.SIGNALS_MAX_LAG,
                 min_corr: float = c.SIGNALS_MIN_CORR,
                 max_corr: float = c.SIGNALS_MAX_CORR) -> None:
        self.num_signals = num_signals
        self.roi_max_samples = roi_max_samples
        self.signal_max_samples = signal_max_samples
        self.peak_max_samples = peak_max_samples
        self.sm = SignalManager(self.num_signals, self.roi_max_samples, self.signal_max_samples, self.peak_max_samples)
        self.color_channel = color_channel
        self.processing_methods = processing_methods if processing_methods is not None else c.SIGNAL_PROCESSING_METHODS
        self.spectrum_transform = spectrum_transform
        self.butter_order = butter_order
        self.butter_min_bw = butter_min_bw
        self.fir_taps = fir_taps
        self.fir_df = fir_df
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.min_corr = min_corr
        self.max_corr = max_corr

    @profiler.timeit
    def make_filter(self, signal_processing_method, sampling_freq) -> np.ndarray:
        if signal_processing_method == c.SignalProcessingMethod.FILTER_BUTTER:
            bands = [min(self.min_freq, sampling_freq / 2 - 2 * self.butter_min_bw),
                     min(self.max_freq, sampling_freq / 2 - self.butter_min_bw)]
            filt = scipy.signal.butter(self.butter_order, bands, btype='bandpass', output='sos', fs=sampling_freq)
        elif signal_processing_method == c.SignalProcessingMethod.FILTER_FIR:
            bands = [0,
                     max(self.min_freq - self.fir_df, self.fir_df),
                     self.min_freq,
                     self.max_freq,
                     min(self.max_freq + self.fir_df, sampling_freq / 2 - self.fir_df),
                     sampling_freq / 2]
            filt = scipy.signal.firls(self.fir_taps, bands, [0, 0, 1, 1, 0, 0], fs=sampling_freq)
        else:
            raise NotImplementedError
        return filt

    @profiler.timeit
    def sample_signal(self, frame: cv2.typing.MatLike, roi: c.Location) -> c.SignalYs:
        if not np.isnan(roi).any():
            _, _, x_0, y_0, x_1, y_1 = roi
            roi_bgr = frame[y_0:y_1, x_0:x_1, :]
            if self.color_channel == c.SignalColorChannel.GREEN:
                values = roi_bgr[..., 1]
            elif self.color_channel == c.SignalColorChannel.CHROM_GREEN:
                values = roi_bgr[..., 1] / 2 - roi_bgr[..., 0] / 4 - roi_bgr[..., 2] / 4 + 0.5
            else:
                raise NotImplementedError
            value = np.mean(values)
        else:
            value = np.nan
        return value

    @profiler.timeit
    def sample_signals(self, frame: cv2.typing.MatLike, rois: list[c.Location]) -> list[c.SignalYs]:
        return [self.sample_signal(frame, r) for r in rois]

    @profiler.timeit
    def process_signal(self, signal_raw: Signal) -> Signal:
        x, y = np.array(signal_raw.x), np.array(signal_raw.y)
        block, valid = signal_raw.v, signal_raw.w
        fs = signal_raw.get_fs()
        if valid.sum() >= 2 and np.isfinite(fs):
            for method in self.processing_methods:
                if method == c.SignalProcessingMethod.DIFF_1:
                    y[valid] = np.diff(y[valid], n=1, axis=0, prepend=y[valid][0])
                elif method == c.SignalProcessingMethod.DIFF_2:
                    y[valid] = np.diff(y[valid], n=2, axis=0, prepend=y[valid][:2])
                elif method == c.SignalProcessingMethod.INTERP_LINEAR:
                    x_interp_block, ts = np.linspace(x[block][0], x[block][-1], block.sum(), retstep=True)
                    y_interp_block = np.interp(x_interp_block, x[valid], y[valid])
                    x[block], y[block] = x_interp_block, y_interp_block
                    valid = block
                    fs = 1 / ts
                elif method == c.SignalProcessingMethod.INTERP_CUBIC:
                    cs = scipy.interpolate.CubicSpline(x[valid], y[valid], axis=0)
                    x_interp_block, ts = np.linspace(x[block][0], x[block][-1], block.sum(), retstep=True)
                    y_interp_block = cs(x_interp_block)
                    x[block], y[block] = x_interp_block, y_interp_block
                    valid = block
                    fs = 1 / ts
                elif method == c.SignalProcessingMethod.DETREND_CONST:
                    y_detrended_valid = scipy.signal.detrend(y[valid], type='constant')
                    y[valid] = y_detrended_valid
                elif method == c.SignalProcessingMethod.DETREND_LINEAR:
                    y_detrended_valid = scipy.signal.detrend(y[valid], type='linear')
                    y[valid] = y_detrended_valid
                elif method == c.SignalProcessingMethod.FILTER_BUTTER:
                    butter = self.make_filter(method, fs)
                    default_padlen = 3 * (2 * len(butter) + 1 - min((butter[:, 2] == 0).sum(), (butter[:, 5] == 0).sum()))
                    padlen = valid.sum() - 1 if valid.sum() <= default_padlen else default_padlen
                    y_filtered_valid = scipy.signal.sosfiltfilt(butter, y[valid], padlen=padlen)
                    y[valid] = y_filtered_valid
                elif method == c.SignalProcessingMethod.FILTER_FIR:
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

    @profiler.timeit
    def process_signals(self, signals_raw: SignalCollection) -> SignalCollection:
        return SignalCollection(signals=[self.process_signal(s) for s in signals_raw])

    @profiler.timeit
    def transform_signal(self, signal_proc: Signal) -> Signal:
        x, y = np.array(signal_proc.x), np.array(signal_proc.y)
        valid = signal_proc.w
        fs = signal_proc.get_fs()
        if valid.sum() >= 2 and np.isfinite(fs):
            if self.spectrum_transform == c.SignalSpectrumTransform.DFT_RFFT:
                num_samples = len(x[valid])
                sampling_period = 1 / fs
                freqs = scipy.fft.rfftfreq(num_samples, sampling_period)
                spectrum = scipy.fft.rfft(y[valid], n=num_samples) # norm='ortho'
                mags = 2 * np.abs(spectrum) / num_samples
            elif self.spectrum_transform == c.SignalSpectrumTransform.PGRAM_WELCH:
                freqs, pgram = scipy.signal.welch(y[valid], fs)
                mags = pgram
            elif self.spectrum_transform == c.SignalSpectrumTransform.PGRAM_LS:
                num_samples = len(x[valid])
                freqs = np.linspace(self.min_freq, self.max_freq, num_samples)
                pgram = scipy.signal.lombscargle(x[valid], y[valid], freqs=freqs*2*np.pi, floating_mean=True, normalize=True)
                mags = pgram
            else:
                raise NotImplementedError
        else:
            freqs, mags = [], []
        signal_spectrum = Signal(freqs, mags, s_maxlen=len(freqs))
        signal_spectrum.set_range((self.min_freq, self.max_freq), (self.min_mag, self.max_mag))
        return signal_spectrum

    @profiler.timeit
    def transform_signals(self, signals_proc: SignalCollection) -> SignalCollection:
        return SignalCollection(signals=[self.transform_signal(s) for s in signals_proc])

    @profiler.timeit
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
        signal_corr = Signal(lags, corr, s_maxlen=len(lags))
        signal_corr.set_range((self.min_lag, self.max_lag), (self.min_corr, self.max_corr))
        return signal_corr

    @profiler.timeit
    def correlate_signals(self, signals_proc: SignalCollection) -> SignalCollection:
        return SignalCollection(signals=[self.correlate_signal_pair(s_a, s_b) for s_a, s_b in itertools.combinations(signals_proc, 2)])

    @profiler.timeit
    def run_pipeline(self, frame: cv2.typing.MatLike, timestamp: float, rois: list[c.Location]) -> tuple[SignalCollection, ...]:
        self.sm.sc_roi.add_samples(timestamp, rois)
        rois = self.sm.sc_roi.get_means(as_int=True)
        samples = self.sample_signals(frame, rois)
        self.sm.sc_raw.add_samples(timestamp, samples)
        self.sm.sc_proc = self.process_signals(self.sm.sc_raw)
        self.sm.sc_spec = self.transform_signals(self.sm.sc_proc)
        self.sm.sc_bpm.add_samples(timestamp, [f * 60 for f, _ in self.sm.sc_spec.get_peaks()])
        self.sm.sc_corr = self.correlate_signals(self.sm.sc_proc)
        self.sm.sc_ptt.add_samples(timestamp, [t * 1000 for t, _ in self.sm.sc_corr.get_peaks()])
        return self.sm.copy_signal_collections()

    def cleanup(self):
        pass
