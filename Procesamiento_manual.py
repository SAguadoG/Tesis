import os
import time
from Extraccion_caracteristicas.signal_functions import load_signal, plot_signal, signal_feature_extraction

# Este programa es el que uso para obtener las características y ver si tienen
# perturbación o no, este programa cambiará o se creará uno distinto en cuanto tenga
# sufientes datos para poder crear el clasificador.


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
    # segundo = input("Ingrese el segundo: ")

    # Obtener el directorio actual del script
    directorio_actual = os.path.abspath(os.path.dirname(__file__))

    # Flag para indicar si se encontró al menos un archivo
    se_encontro_archivo = False

    # Aqui se construye el nombre del archivo
    # la fracción al ser aleatoria, se hará un bucle de 0 a 9 para barrer el segundo
    # porque la centésima es aleatoria
    # la siguiente linea (primer bucle) se eliminara, simplemente es para comprobar en iteracion
    # un conjunto de segundos
    # para hacer más rápida la comprobación, para coger archivos de 30 en 30 y no de 1 en 1.
    for segundo in range(31):
        # Pongo una pequeña pausa, si intento generar tantas imágenes seguidas.
        # Pycharm me da error en las últimas iteraciones
        time.sleep(0.20)
        for fraccion in range(10):
            nombre_archivo = f"{year}_{mes}_{dia}_{hora}_{minuto}_{segundo}.{fraccion}_lab_politecnica_3a.mat"
            # Obtener la ruta completa al archivo
            ruta_archivo = os.path.join(directorio_actual, 'Manual_test', nombre_archivo)

            # Procesar el archivo y actualizar el flag si se encuentra
            se_encontro_archivo |= procesar_archivo(ruta_archivo)

            # Una pequeña pausa, ajustar para que no sea ni demasiado lento ni se cree problemas al plotear
            time.sleep(0.20)

    # Mensaje si no se encontraron archivos
        if not se_encontro_archivo:
            print(f"No se encontraron archivos para la fecha proporcionada.")

    # Para saber cuando acaba de buscar
    print("Se termina de buscar")

# Llamada a la función principal


if __name__ == "__main__":
    obtener_fecha()
