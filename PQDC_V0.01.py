import os
from Extraccion_caracteristicas.signal_functions import load_signal, plot_signal, signal_feature_extraction


def procesar_archivo(ruta_archivo):
    try:
        signal, fs = load_signal(ruta_archivo)
        plot_signal(signal, fs)
        variance, skewness, kurtosis, thd, crest_factor = signal_feature_extraction(signal, fs)
        # Se muestran las características de la señal
        print("Variance:", variance)
        print("Skewness:", skewness)
        print("Kurtosis:", kurtosis)
        print("THD:", thd)
        print("Factor de Cresta:", crest_factor)

        return True

    except FileNotFoundError:
        return False


def obtener_fecha():
    # Aqui se pide la fecha en modo cadena
    # En modo int o float para el decimal no funciona
    year = input("Ingrese el año: ")
    mes = input("Ingrese el mes: ")
    dia = input("Ingrese el día: ")
    hora = input("Ingrese la hora: ")
    minuto = input("Ingrese el minuto: ")
    segundo = input("Ingrese el segundo: ")

    # Obtener el directorio actual del script
    directorio_actual = os.path.abspath(os.path.dirname(__file__))

    # Flag para indicar si se encontró al menos un archivo
    se_encontro_archivo = False

    # Aqui se construye el nombre del archivo
    # la fracción al ser aleatoria, se hará un bucle de 0 a 9
    for fraccion in range(10):
        nombre_archivo = f"{year}_{mes}_{dia}_{hora}_{minuto}_{segundo}.{fraccion}_lab_politecnica_3a.mat"
        # Obtener la ruta completa al archivo
        ruta_archivo = os.path.join(directorio_actual, 'Datos', nombre_archivo)

        # Procesar el archivo y actualizar el flag si se encuentra
        se_encontro_archivo |= procesar_archivo(ruta_archivo)

    # Mensaje si no se encontraron archivos
    if not se_encontro_archivo:
        print(f"No se encontraron archivos para la fecha proporcionada.")


# Llamada a la función principal
if __name__ == "__main__":
    obtener_fecha()
