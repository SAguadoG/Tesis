import os
import numpy as np

# Directorio donde se encuentra el archivo .npy
data_dir = 'data/train/flicker_signals'
file_name = 'flck_s_1.npy'
file_path = os.path.join(data_dir, file_name)

# Verificar si el archivo .npy está vacío o no
try:
    if os.path.exists(file_path):
        data = np.load(file_path)
        if data.size == 0:
            print(f"El archivo {file_name} en {data_dir} está vacío.")
        else:
            print(f"El archivo {file_name} en {data_dir} no está vacío.")
    else:
        print(f"El archivo {file_name} no fue encontrado en {data_dir}.")
except Exception as e:
    print(f"Error al procesar el archivo {file_name}: {e}")