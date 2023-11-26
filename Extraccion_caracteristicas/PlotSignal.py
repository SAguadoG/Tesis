import scipy.io
import matplotlib.pyplot as plt
import numpy as np

data = scipy.io.loadmat('2022_5_3_2_0_2.8_lab_politecnica_3a.mat')

signal = data['signal'][0]
fs = data['Fs'][0, 0]

# Calcular la transformada de Fourier de la señal
fft_signal = np.fft.fft(signal)

# Calcular la frecuencia correspondiente a cada componente de la transformada de Fourier
freqs = np.fft.fftfreq(len(signal)) * fs

# Plotear la señal y su transformada de Fourier
fig, ax = plt.subplots(2, 1, figsize=(8, 8))

ax[0].plot(signal)
ax[0].set_title('Señal original')

ax[1].plot(freqs, np.abs(fft_signal))
ax[1].set_xlim(0, fs / 2)
ax[1].set_title('Transformada de Fourier de la señal')

plt.show()