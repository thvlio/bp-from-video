import copy
import enum
import itertools
import math
import typing

import cv2
import numpy as np
import scipy.fft
import scipy.interpolate
import scipy.signal

import model
import profiler
import roi
import signal_data

if typing.TYPE_CHECKING:
    import inference_runner
    import video_reader


class SignalColorChannel(enum.Enum):
    GREEN = enum.auto()
    CHROM_GREEN = enum.auto()


class SignalProcessingMethod(enum.Enum):
    DIFF_1 = enum.auto()
    DIFF_2 = enum.auto()
    INTERP_LINEAR = enum.auto()
    INTERP_CUBIC = enum.auto()
    DETREND_CONST = enum.auto()
    DETREND_LINEAR = enum.auto()
    FILTER_BUTTER = enum.auto()
    FILTER_FIR = enum.auto()


class SignalSpectrumTransform(enum.Enum):
    DFT_RFFT = enum.auto()
    PGRAM_WELCH = enum.auto()
    PGRAM_LS = enum.auto()


SIGNAL_COLOR_CHANNEL = SignalColorChannel.GREEN

ROI_MAX_SAMPLES = 1
SIGNAL_MAX_SAMPLES = 250
PEAK_MAX_SAMPLES = 50

SIGNAL_PROCESSING_METHODS = [
    # SignalProcessingMethod.DIFF_1,
    # SignalProcessingMethod.INTERP_CUBIC,
    SignalProcessingMethod.FILTER_BUTTER,
]

FILTER_BUTTER_ORDER = 16
FILTER_BUTTER_MIN_BW = 0.1
FILTER_FIR_TAPS = 127
FILTER_FIR_DF = 0.3

SIGNAL_SPECTRUM_TRANSFORM = SignalSpectrumTransform.PGRAM_LS

FILTER_MIN_FREQ = 0.8
FILTER_MAX_FREQ = 4.0
SPECTRUM_MIN_MAG = 0.0
SPECTRUM_MAX_MAG = 1.0

SIGNALS_MIN_LAG = -0.5
SIGNALS_MAX_LAG = 0.5
SIGNALS_MIN_CORR = -1.0
SIGNALS_MAX_CORR = 1.0


class SignalStore:

    def __init__(self, num_signals: int, roi_max_samples: int, signal_max_samples: int, peak_max_samples: int) -> None:
        self.sg_roi = signal_data.SignalGroup(num_signals, yi=(np.nan,)*6, s_maxlen=roi_max_samples)
        self.sg_raw = signal_data.SignalGroup(num_signals, s_maxlen=signal_max_samples)
        self.sg_proc = signal_data.SignalGroup(num_signals)
        self.sg_spec = signal_data.SignalGroup(num_signals)
        self.sg_corr = signal_data.SignalGroup(math.comb(num_signals, 2))
        self.sg_bpm = signal_data.SignalGroup(num_signals, s_maxlen=peak_max_samples)
        self.sg_ptt = signal_data.SignalGroup(math.comb(num_signals, 2), s_maxlen=peak_max_samples)


class SignalProcessor:

    def __init__(self,
                 selected_roi_configs: list[roi.ROIConfig] | None = None,
                 roi_max_samples: int = ROI_MAX_SAMPLES,
                 signal_max_samples: int = SIGNAL_MAX_SAMPLES,
                 peak_max_samples: int = PEAK_MAX_SAMPLES,
                 *,
                 color_channel: SignalColorChannel = SIGNAL_COLOR_CHANNEL,
                 processing_methods: list[SignalProcessingMethod] | None = None,
                 spectrum_transform: SignalSpectrumTransform = SIGNAL_SPECTRUM_TRANSFORM,
                 butter_order: int = FILTER_BUTTER_ORDER,
                 butter_min_bw: float = FILTER_BUTTER_MIN_BW,
                 fir_taps: int = FILTER_FIR_TAPS,
                 fir_df: float = FILTER_FIR_DF,
                 min_freq: float = FILTER_MIN_FREQ,
                 max_freq: float = FILTER_MAX_FREQ,
                 min_mag: float = SPECTRUM_MIN_MAG,
                 max_mag: float = SPECTRUM_MAX_MAG,
                 min_lag: float = SIGNALS_MIN_LAG,
                 max_lag: float = SIGNALS_MAX_LAG,
                 min_corr: float = SIGNALS_MIN_CORR,
                 max_corr: float = SIGNALS_MAX_CORR) -> None:
        self.selected_roi_configs = selected_roi_configs if selected_roi_configs is not None else roi.SELECTED_ROI_CONFIGS
        self.num_signals = len(self.selected_roi_configs)
        self.roi_max_samples = roi_max_samples
        self.signal_max_samples = signal_max_samples
        self.peak_max_samples = peak_max_samples
        self.store = SignalStore(self.num_signals, self.roi_max_samples, self.signal_max_samples, self.peak_max_samples)
        self.color_channel = color_channel
        self.processing_methods = processing_methods if processing_methods is not None else SIGNAL_PROCESSING_METHODS
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
    def calc_rois(self, model_results: 'inference_runner.InferenceResults') -> list[roi.Location]:
        rois = []
        for roi_config in self.selected_roi_configs:
            if roi_config.model_type is model.ModelType.FACE_LANDMARKER:
                landmarks = model_results.face_landmarker.detections
            elif roi_config.model_type is model.ModelType.HAND_LANDMARKER:
                landmarks = model_results.hand_landmarker.detections
            else:
                raise NotImplementedError
            if len(landmarks) > 0:
                bbox, points = landmarks[0]
                pp = np.squeeze(np.mean([points[i] for i in roi_config.landmark_indices], axis=0))
                x, y = pp.round().astype(int)
                left_m, top_m, right_m, bottom_m = roi_config.relative_bbox
                x_0 = int(round(x + left_m * (bbox[2] - bbox[0])))
                y_0 = int(round(y + top_m * (bbox[3] - bbox[1])))
                x_1 = int(round(x + right_m * (bbox[2] - bbox[0])))
                y_1 = int(round(y + bottom_m * (bbox[3] - bbox[1])))
                sroi = (x, y, x_0, y_0, x_1, y_1)
            else:
                sroi = (np.nan,) * 6
            rois.append(sroi)
        return rois

    @profiler.timeit
    def make_filter(self, signal_processing_method: SignalProcessingMethod, sampling_freq: float) -> np.ndarray:
        if signal_processing_method is SignalProcessingMethod.FILTER_BUTTER:
            bands = [min(self.min_freq, sampling_freq / 2 - 2 * self.butter_min_bw),
                     min(self.max_freq, sampling_freq / 2 - self.butter_min_bw)]
            filt = scipy.signal.butter(self.butter_order, bands, btype='bandpass', output='sos', fs=sampling_freq)
        elif signal_processing_method is SignalProcessingMethod.FILTER_FIR:
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
    def sample_signal(self, frame: cv2.typing.MatLike, sroi: roi.Location) -> signal_data.YType:
        if not np.isnan(sroi).any():
            _, _, x_0, y_0, x_1, y_1 = sroi
            roi_bgr = frame[y_0:y_1, x_0:x_1, :]
            if self.color_channel is SignalColorChannel.GREEN:
                pixel_values = roi_bgr[..., 1]
            elif self.color_channel is SignalColorChannel.CHROM_GREEN:
                pixel_values = roi_bgr[..., 1] / 2 - roi_bgr[..., 0] / 4 - roi_bgr[..., 2] / 4 + 0.5
            else:
                raise NotImplementedError
            value = np.mean(pixel_values)
        else:
            value = np.nan
        return value

    @profiler.timeit
    def sample_signals(self, frame: cv2.typing.MatLike, rois: list[roi.Location]) -> list[signal_data.YType]:
        return [self.sample_signal(frame, r) for r in rois]

    @profiler.timeit
    def process_signal(self, signal_raw: signal_data.Signal) -> signal_data.Signal:
        x, y = np.array(signal_raw.x), np.array(signal_raw.y)
        block, valid = signal_raw.v, signal_raw.w
        fs = signal_raw.get_fs()
        if valid.sum() >= 2 and np.isfinite(fs):
            for method in self.processing_methods:
                if method is SignalProcessingMethod.DIFF_1:
                    y[valid] = np.diff(y[valid], n=1, axis=0, prepend=y[valid][0])
                elif method is SignalProcessingMethod.DIFF_2:
                    y[valid] = np.diff(y[valid], n=2, axis=0, prepend=y[valid][:2])
                elif method is SignalProcessingMethod.INTERP_LINEAR:
                    x_interp_block, ts = np.linspace(x[block][0], x[block][-1], block.sum(), retstep=True)
                    y_interp_block = np.interp(x_interp_block, x[valid], y[valid])
                    x[block], y[block] = x_interp_block, y_interp_block
                    valid = block
                    fs = 1 / ts
                elif method is SignalProcessingMethod.INTERP_CUBIC:
                    cs = scipy.interpolate.CubicSpline(x[valid], y[valid], axis=0)
                    x_interp_block, ts = np.linspace(x[block][0], x[block][-1], block.sum(), retstep=True)
                    y_interp_block = cs(x_interp_block)
                    x[block], y[block] = x_interp_block, y_interp_block
                    valid = block
                    fs = 1 / ts
                elif method is SignalProcessingMethod.DETREND_CONST:
                    y_detrended_valid = scipy.signal.detrend(y[valid], type='constant')
                    y[valid] = y_detrended_valid
                elif method is SignalProcessingMethod.DETREND_LINEAR:
                    y_detrended_valid = scipy.signal.detrend(y[valid], type='linear')
                    y[valid] = y_detrended_valid
                elif method is SignalProcessingMethod.FILTER_BUTTER:
                    butter = self.make_filter(method, fs)
                    default_padlen = 3 * (2 * len(butter) + 1 - min((butter[:, 2] == 0).sum(), (butter[:, 5] == 0).sum()))
                    padlen = valid.sum() - 1 if valid.sum() <= default_padlen else default_padlen
                    y_filtered_valid = scipy.signal.sosfiltfilt(butter, y[valid], padlen=padlen)
                    y[valid] = y_filtered_valid
                elif method is SignalProcessingMethod.FILTER_FIR:
                    fir = self.make_filter(method, fs)
                    default_padlen = 3 * len(fir)
                    padlen = valid.sum() - 1 if valid.sum() <= default_padlen else default_padlen
                    y_filtered_valid = scipy.signal.filtfilt(fir, 1.0, y[valid], padlen=padlen)
                    y[valid] = y_filtered_valid
                else:
                    raise NotImplementedError
        signal_proc = signal_data.Signal(x, y, len(x))
        signal_proc.set_range()
        return signal_proc

    @profiler.timeit
    def process_signals(self, signals_raw: signal_data.SignalGroup) -> signal_data.SignalGroup:
        return signal_data.SignalGroup(signals=[self.process_signal(s) for s in signals_raw])

    @profiler.timeit
    def transform_signal(self, signal_proc: signal_data.Signal) -> signal_data.Signal:
        x, y = np.array(signal_proc.x), np.array(signal_proc.y)
        valid = signal_proc.w
        fs = signal_proc.get_fs()
        if valid.sum() >= 2 and np.isfinite(fs):
            if self.spectrum_transform is SignalSpectrumTransform.DFT_RFFT:
                num_samples = len(x[valid])
                sampling_period = 1 / fs
                freqs = scipy.fft.rfftfreq(num_samples, sampling_period)
                spectrum = scipy.fft.rfft(y[valid], n=num_samples) # norm='ortho'
                mags = 2 * np.abs(spectrum) / num_samples
            elif self.spectrum_transform is SignalSpectrumTransform.PGRAM_WELCH:
                freqs, pgram = scipy.signal.welch(y[valid], fs)
                mags = pgram
            elif self.spectrum_transform is SignalSpectrumTransform.PGRAM_LS:
                num_samples = len(x[valid])
                freqs = np.linspace(self.min_freq, self.max_freq, num_samples)
                pgram = scipy.signal.lombscargle(x[valid], y[valid], freqs=freqs*2*np.pi, floating_mean=True, normalize=True)
                mags = pgram
            else:
                raise NotImplementedError
        else:
            freqs, mags = [], []
        signal_spectrum = signal_data.Signal(freqs, mags, s_maxlen=len(freqs))
        signal_spectrum.set_range((self.min_freq, self.max_freq), (self.min_mag, self.max_mag))
        return signal_spectrum

    @profiler.timeit
    def transform_signals(self, signals_proc: signal_data.SignalGroup) -> signal_data.SignalGroup:
        return signal_data.SignalGroup(signals=[self.transform_signal(s) for s in signals_proc])

    @profiler.timeit
    def correlate_signal_pair(self, signal_a: signal_data.Signal, signal_b: signal_data.Signal) -> signal_data.Signal:
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
        signal_corr = signal_data.Signal(lags, corr, s_maxlen=len(lags))
        signal_corr.set_range((self.min_lag, self.max_lag), (self.min_corr, self.max_corr))
        return signal_corr

    @profiler.timeit
    def correlate_signals(self, signals_proc: signal_data.SignalGroup) -> signal_data.SignalGroup:
        return signal_data.SignalGroup(signals=[self.correlate_signal_pair(s_a, s_b) for s_a, s_b in itertools.combinations(signals_proc, 2)])

    @profiler.timeit
    def process(self, frame_data: 'video_reader.FrameData', model_results: 'inference_runner.InferenceResults') -> SignalStore:
        rois = self.calc_rois(model_results)
        self.store.sg_roi.add_samples(frame_data.timestamp, rois)
        rois = self.store.sg_roi.get_means(as_int=True)
        samples = self.sample_signals(frame_data.frame, rois)
        self.store.sg_raw.add_samples(frame_data.timestamp, samples)
        self.store.sg_proc = self.process_signals(self.store.sg_raw)
        self.store.sg_spec = self.transform_signals(self.store.sg_proc)
        self.store.sg_bpm.add_samples(frame_data.timestamp, [f * 60 for f, _ in self.store.sg_spec.get_peaks()])
        self.store.sg_corr = self.correlate_signals(self.store.sg_proc)
        self.store.sg_ptt.add_samples(frame_data.timestamp, [t * 1000 for t, _ in self.store.sg_corr.get_peaks()])
        return copy.deepcopy(self.store)

    run = process

    def cleanup(self):
        pass
