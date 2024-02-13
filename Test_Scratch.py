import os

# Directorio de datos
data_dir = 'data'

# Tipos de señales
signal_types = ['flicker_signals', 'harmonic_signals', 'interruption_signals', 'original_signals', 'sag_signals',
                'swell_signals']

# Función para contar archivos en cada carpeta
def contar_archivos_por_carpeta(tipo_de_senal, conjunto_de_datos):
    carpeta = os.path.join(data_dir, conjunto_de_datos, tipo_de_senal)
    num_archivos = len([nombre for nombre in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, nombre))])
    return num_archivos

# Imprimir número de archivos por carpeta para cada tipo de señal y conjunto de datos
for conjunto_de_datos in ['train', 'test', 'val']:
    print(f"Conjunto de datos: {conjunto_de_datos}")
    for tipo_de_senal in signal_types:
        num_archivos = contar_archivos_por_carpeta(tipo_de_senal, conjunto_de_datos)
        print(f"Carpeta '{tipo_de_senal}': {num_archivos} archivos")

