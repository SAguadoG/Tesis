import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def load_signal(datos_path):
    data = scipy.io.loadmat(datos_path)
    signal = data['signal'][0]
    fs = data['Fs'][0, 0]
    return signal, fs

def plot_signal(signal, fs):
    # Plotear la señal en el dominio del tiempo y la frecuencia.
    N = len(signal)
    t = np.arange(0, N) / fs

    # Calcular la transformada de Fourier de la señal
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/fs)

    # Plotear la señal y su transformada de Fourier
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    ax[0].plot(t, signal)
    ax[0].set_title('Señal original')

    ax[1].plot(freqs[:N//2], np.abs(fft_signal[:N//2]))
    ax[1].set_xlim(0, fs / 10)
    ax[1].set_title('Transformada de Fourier de la señal')

    plt.show()

def signal_feature_extraction(signal, fs):


    mean_value = np.mean(signal)
    unbias_data = signal - mean_value
    N = len(unbias_data)

    unbias_data_2 = unbias_data ** 2
    unbias_data_3 = unbias_data_2 * unbias_data
    unbias_data_4 = unbias_data_3 * unbias_data
    variance = np.var(unbias_data)
    skewness = np.mean(unbias_data_3) / (variance ** 1.5)
    kurtosis = np.mean(unbias_data_4) / (variance ** 2) - 3
    thd = np.sqrt(np.sum(np.abs(np.fft.fft(signal)[2:4])) / np.abs(np.fft.fft(signal)[1]))
    rms = np.sqrt(np.mean(signal ** 2))
    crest_factor = np.max(signal) / rms

    return variance, skewness, kurtosis, thd, crest_factor