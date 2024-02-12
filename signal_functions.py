import scipy.io
import numpy as np

def load_signal(datos_path):
    data = scipy.io.loadmat(datos_path)
    signal = data['signal'][0]
    fs = data['Fs'][0, 0]
    return signal, fs

def signal_feature_extraction(signal, fs):

    # Normalizar la señal
    max_theoretical_value = 230 * np.sqrt(2)
    normalized_signal = signal / max_theoretical_value

    # Calcular características de la señal normalizada
    mean_value = np.mean(normalized_signal)
    unbias_data = normalized_signal - mean_value
    # n = len(unbias_data)

    unbias_data_2 = unbias_data ** 2
    unbias_data_3 = unbias_data_2 * unbias_data
    unbias_data_4 = unbias_data_3 * unbias_data
    variance = np.var(unbias_data)
    skewness = np.mean(unbias_data_3) / (variance ** 1.5)
    kurtosis = np.mean(unbias_data_4) / (variance ** 2) - 3
    thd = np.sqrt(np.sum(np.abs(np.fft.fft(normalized_signal)[2:4])) / np.abs(np.fft.fft(normalized_signal)[1]))
    rms = np.sqrt(np.mean(normalized_signal ** 2))
    crest_factor = np.max(normalized_signal) / rms

    return variance, skewness, kurtosis, thd, crest_factor
