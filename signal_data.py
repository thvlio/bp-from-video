import collections

import numpy as np

from config import Config as c


class Signal:

    def __init__(self,
                 x0: c.SignalXs | list[c.SignalXs] = np.nan,
                 y0: c.SignalYs | list[c.SignalYs] = np.nan,
                 s_maxlen: int | None = None) -> None:
        self.x = collections.deque(x0 if isinstance(x0, (list, np.ndarray)) else [x0] * s_maxlen if s_maxlen is not None else [], s_maxlen)
        self.y = collections.deque(y0 if isinstance(y0, (list, np.ndarray)) else [y0] * s_maxlen if s_maxlen is not None else [], s_maxlen)
        self.v: np.ndarray[bool]
        self.w: np.ndarray[bool]
        self.range_x: tuple[c.SignalXs, c.SignalXs]
        self.range_y: tuple[c.SignalYs, c.SignalYs]
        self.reset_mask()
        self.reset_range()

    def __repr__(self):
        with np.printoptions(legacy='1.25'):
            return f'{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in vars(self).items()])})'

    def add_sample(self, xp: c.SignalXs, yp: c.SignalYs) -> None:
        self.x.append(xp)
        self.y.append(yp)
        self.reset_mask()
        self.reset_range()

    def set_data(self, data_x: list[c.SignalXs] | None = None, data_y: list[c.SignalYs] | None = None) -> None:
        self.x = collections.deque(data_x, self.x.maxlen) if data_x is not None else self.x
        self.y = collections.deque(data_y, self.y.maxlen) if data_y is not None else self.y
        self.reset_mask()
        self.reset_range()

    def reset_mask(self) -> None:
        self.v = np.isfinite(self.x)
        self.w = np.isfinite(self.y).all(axis=1) if np.ndim(self.y) == 2 else np.isfinite(self.y)

    def reset_range(self) -> None:
        self.range_x = (np.nanmin(self.x), np.nanmax(self.x)) if self.v.sum() >= 2 else (np.nan, np.nan)
        self.range_y = (np.nanmin(self.y), np.nanmax(self.y)) if self.w.sum() >= 2 else (np.nan, np.nan)

    def set_range(self, range_x: tuple[c.SignalXs, c.SignalXs] | None = None, range_y: tuple[c.SignalYs, c.SignalYs] | None = None) -> None:
        self.range_x = range_x if range_x is not None else self.range_x
        self.range_y = range_y if range_y is not None else self.range_y

    def get_fs(self, only_valid: bool = False) -> c.SignalXs:
        x = np.array(self.x)
        u = self.w if only_valid else self.v
        return 1 / np.nanmean(np.diff(x[u])) if u.sum() >= 2 else np.nan

    def get_mean(self, as_int: bool = False) -> c.SignalYs:
        y = np.array(self.y)
        y_mean = np.squeeze(np.nanmean(y, axis=0)) if self.w.any() else y[-1]
        return y_mean.round().astype(int) if as_int and self.w.any() else y_mean

    def get_peak(self, min_x: c.SignalXs | None = None, max_x: c.SignalXs | None = None) -> tuple[c.SignalXs, c.SignalYs]:
        x, y = np.array(self.x), np.array(self.y)
        min_x = min_x if min_x is not None else self.range_x[0]
        max_x = max_x if max_x is not None else self.range_x[1]
        u = (min_x <= x) & (x <= max_x) & self.w
        return (x[u][np.argmax(y[u])], np.max(y[u])) if u.sum() >= 2 else ((np.nan,)*y.shape[-1], np.nan) if np.ndim(y) == 2 else (np.nan, np.nan)


class SignalCollection:

    # TODO: rename to SignalGroup maybe

    def __init__(self,
                 num_signals: int | None = None,
                 x0: c.SignalXs | list[c.SignalXs] = np.nan,
                 y0: c.SignalYs | list[c.SignalYs] = np.nan,
                 s_maxlen: int | None = None,
                 *,
                 signals: list[Signal] | None = None) -> None:
        self.num_signals = num_signals if signals is None else len(signals)
        self.signals = [Signal(x0, y0, s_maxlen) for _ in range(self.num_signals)] if signals is None else signals
        self.range_x: tuple[c.SignalXs, c.SignalXs]
        self.range_y: tuple[c.SignalYs, c.SignalYs]
        self.reset_ranges()

    def __repr__(self):
        return f'{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in vars(self).items()])})'

    def __iter__(self):
        return (s for s in self.signals)

    def add_samples(self, xps: c.SignalXs | list[c.SignalXs], yps: list[c.SignalYs]) -> None:
        xps = xps if isinstance(xps, (list, np.ndarray)) else [xps] * self.num_signals
        for signal, xp, yp in zip(self.signals, xps, yps):
            signal.add_sample(xp, yp)
        self.reset_ranges()

    def reset_ranges(self) -> None:
        for signal in self.signals:
            signal.reset_range()
        lower_xs, upper_xs, lower_ys, upper_ys = zip(*((*s.range_x, *s.range_y) for s in self.signals))
        self.range_x = (np.nanmin(lower_xs), np.nanmax(upper_xs)) if np.isfinite([lower_xs, upper_xs]).any(axis=1).all() else (np.nan, np.nan)
        self.range_y = (np.nanmin(lower_ys), np.nanmax(upper_ys)) if np.isfinite([lower_ys, upper_ys]).any(axis=1).all() else (np.nan, np.nan)

    def set_ranges(self, range_x: tuple[c.SignalXs, c.SignalXs] | None = None, range_y: tuple[c.SignalYs, c.SignalYs] | None = None) -> None:
        for signal in self.signals:
            signal.set_range(range_x, range_y)
        self.range_x = range_x if range_x is not None else self.range_x
        self.range_y = range_y if range_y is not None else self.range_y

    def get_means(self, as_int: bool = False) -> list[c.SignalYs]:
        return [s.get_mean(as_int) for s in self.signals]

    def get_peaks(self, min_x: c.SignalXs | None = None, max_x: c.SignalXs | None = None) -> list[tuple[c.SignalXs, c.SignalYs]]:
        return [s.get_peak(min_x, max_x) for s in self.signals]
