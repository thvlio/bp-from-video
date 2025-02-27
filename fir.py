import numpy as np
import scipy.signal
from matplotlib import pyplot as plt


def main():

    plt.rcParams.update({'font.size': 22})

    min_freq = 0.8
    max_freq = 6.0
    fs = 25

    butter_order = [2, 4, 8, 16]
    butter_bands = [min_freq, max_freq]

    fir_taps = [31, 63, 127, 255, 511, 1023]
    fir_df = 0.3
    fir_bands = [0, min_freq - fir_df, min_freq, max_freq, max_freq + fir_df, fs / 2]
    fir_gains = [0, 0, 1, 1, 0, 0]

    fig, ax = plt.subplots(figsize=(16, 16))

    hs = []

    for order in butter_order:
        butter = scipy.signal.butter(order, butter_bands, btype='bandpass', output='sos', fs=fs)
        freq, response = scipy.signal.freqz_sos(butter)
        hs.append(ax.semilogy(0.5*fs*freq/np.pi, np.abs(response))[0])
    ax.semilogy(butter_bands, np.maximum([1, 1], 1e-7), 'k--', linewidth=2)

    for taps in fir_taps:
        fir = scipy.signal.firls(taps, fir_bands, fir_gains, fs=fs)
        freq, response = scipy.signal.freqz(fir)
        hs.append(ax.semilogy(0.5*fs*freq/np.pi, np.abs(response))[0])
    for band, gains in zip(zip(fir_bands[::2], fir_bands[1::2]), zip(fir_gains[::2], fir_gains[1::2])):
        ax.semilogy(band, np.maximum(gains, 1e-7), 'k--', linewidth=2)

    labels = [f'butter-{o}' for o in butter_order] + [f'fir-{t}' for t in fir_taps]
    ax.legend(hs, labels, loc='lower center', frameon=False)
    ax.set_xlabel('Frequency (Hz)')
    ax.grid(True)
    ax.set(title=f'Band-pass {min_freq}-{max_freq} Hz', ylabel='Magnitude')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
