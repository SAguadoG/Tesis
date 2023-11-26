clear
clc

% Cargar la señal original
data = load('2022_5_3_2_0_2.8_lab_politecnica_3a.mat');
signal_original = data.signal;

% Normalizar la señal al rango [-1, 1]
signal_original = signal_original / max(abs(signal_original));

% Parámetros de la amplificación aleatoria
longitud_signal = length(signal_original);
duracion_amplificacion = 3000;  % en muestras (0.3 segundos a una tasa de muestreo de 10,000 Hz)
factor_amplitud_min = 1.2;
factor_amplitud_max = 1.7;

% Generar la posición aleatoria para la amplificación
inicio_amplificacion = randi([1, longitud_signal - duracion_amplificacion]);

% Generar el factor de amplificación aleatorio
factor_amplitud = (factor_amplitud_max - factor_amplitud_min) * rand() + factor_amplitud_min;

% Aplicar la amplificación en la parte aleatoria de la señal original
signal_amplificada = signal_original;
signal_amplificada(inicio_amplificacion:(inicio_amplificacion + duracion_amplificacion - 1)) = ...
    signal_amplificada(inicio_amplificacion:(inicio_amplificacion + duracion_amplificacion - 1)) * factor_amplitud;

% Visualizar la señal original y la señal amplificada
figure;
subplot(2, 1, 1);
plot(signal_original);
title('Señal Original');
xlabel('Muestras');
ylabel('Amplitud');

subplot(2, 1, 2);
plot(signal_amplificada);
title('Señal Amplificada Aleatoriamente');
xlabel('Muestras');
ylabel('Amplitud');
