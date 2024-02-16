import os
import numpy as np
import matplotlib.pyplot as plt

# Directorio y nombre del archivo a cargar
data_dir = 'data/val/flicker_signals'
file_name = 'flck_s_3456.npy'
file_path = os.path.join(data_dir, file_name)

# Cargar el archivo .npy
signal_data = np.load(file_path)

# Crear eje x con el número de iteraciones
num_iterations = len(signal_data)
iterations = range(1, num_iterations + 1)

# Plotear la señal
plt.plot(iterations, signal_data)
plt.title('Señal de harmonic_signals (hrc_s_3456)')
plt.xlabel('Iteraciones')
plt.ylabel('Tensión')
plt.grid(True)
plt.show()
