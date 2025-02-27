import datetime
from pathlib import Path

import matplotlib
import matplotlib.axes
import matplotlib.figure
import numpy as np
import scipy.fft
import scipy.io
import scipy.signal
from matplotlib import pyplot as plt


def dft(signal, sampling_freq, num_samples):
    freqs = scipy.fft.rfftfreq(num_samples, 1 / sampling_freq)
    spectrum = scipy.fft.rfft(signal, n=num_samples)
    mags = 2 * np.abs(spectrum) / num_samples
    # mags = np.angle(spectrum, True)
    return freqs, mags


def main():

    plt.rcParams.update({'font.size': 22})

    signal_path = Path('/home/thulio/Downloads/3314767_0004m.mat')
    header_path = signal_path.with_suffix('.hea')

    header_lines = header_path.read_text().split('\n')
    header = header_lines[0].strip().split()
    specs = [li.strip().split() for li in header_lines[1:] if not li.startswith('#') and len(li.strip()) > 0]

    signal_info = []
    num_signals = int(header[1])
    sampling_freq = float(header[2])
    num_samples = int(header[3])
    start_time = datetime.datetime.strptime(header[4]+'000', '%H:%M:%S.%f')
    # start_time = datetime.datetime.fromisoformat(header[4])
    for ss in range(num_signals):
        name = specs[ss][-1]
        gain_base, unit = specs[ss][2].split('/')
        gain, base = map(float, gain_base.replace('(', ' ').replace(')', ' ').strip().split())
        signal_info.append((gain, base, unit, name))

    timestamps = np.linspace(start_time, start_time + datetime.timedelta(seconds=num_samples/sampling_freq), num_samples)
    signals = scipy.io.loadmat(signal_path)

    ax_t: matplotlib.axes.Axes
    ax_s: matplotlib.axes.Axes
    fig_ppg, (ax_t, ax_s) = plt.subplots(2, 1, figsize=(16, 16))
    alpha = 1.0

    fs = sampling_freq
    min_freq = 0.8
    max_freq = 3.0
    butter_order = [4, 8, 16]
    butter_bands = [min_freq, max_freq]
    fir_taps = [255, 511, 1023, 4093]
    fir_df = 0.3
    fir_bands = [0, min_freq - fir_df, min_freq, max_freq, max_freq + fir_df, fs / 2]
    fir_gains = [0, 0, 1, 1, 0, 0]
    detrend_methods = ['linear']

    ppg = (signals['val'][2] - signal_info[2][1]) / signal_info[2][0]
    ax_t.plot(timestamps, ppg, alpha=alpha)
    freqs, mags = dft(ppg, sampling_freq, num_samples)
    ax_s.plot(freqs, mags, alpha=alpha)

    for method in detrend_methods:
        ppg_detrended = scipy.signal.detrend(ppg, type=method)
        ax_t.plot(timestamps, ppg_detrended, alpha=alpha)
        freqs, mags = dft(ppg_detrended, sampling_freq, num_samples)
        ax_s.plot(freqs, mags, alpha=alpha)

    for order in butter_order:
        butter = scipy.signal.butter(order, butter_bands, btype='bandpass', output='sos', fs=fs)
        ppg_butter = scipy.signal.sosfiltfilt(butter, ppg)
        ax_t.plot(timestamps, ppg_butter, alpha=alpha)
        freqs, mags = dft(ppg_butter, sampling_freq, num_samples)
        ax_s.plot(freqs, mags, alpha=alpha)

    for taps in fir_taps:
        fir = scipy.signal.firls(taps, fir_bands, fir_gains, fs=fs)
        ppg_fir = scipy.signal.filtfilt(fir, 1, ppg)
        ax_t.plot(timestamps, ppg_fir, alpha=alpha)
        freqs, mags = dft(ppg_fir, sampling_freq, num_samples)
        ax_s.plot(freqs, mags, alpha=alpha)

    ax_t.set(title=f'{signal_info[2][3]} ({signal_info[2][2]})', xlabel='Time', ylabel=f'{signal_info[2][3]} ({signal_info[2][2]})')
    ax_t.legend(['raw'] + [f'detrend-{m}' for m in detrend_methods] + [f'butter-{o}' for o in butter_order] + [f'fir-{t}' for t in fir_taps])
    ax_t.grid(True)
    ax_s.set(xlabel='Frequency', ylabel='Magnitude', xlim=[0.01, sampling_freq/10], ylim=[0.0, 0.2])
    # ax_s.set(xlabel='Frequency', ylabel='Phase')
    ax_s.legend(['raw'] + [f'detrend-{m}' for m in detrend_methods] + [f'butter-{o}' for o in butter_order] + [f'fir-{t}' for t in fir_taps])
    ax_s.grid(True)
    fig_ppg.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
