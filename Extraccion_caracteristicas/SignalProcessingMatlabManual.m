clear all
clc

tic

% Cargamos la señal y obtenemos los valores de amplitud y frecuencia.
data = load('2021_11_30_5_0_3.7_lab_politecnica_3a.mat');

signal = data.signal;
fs = data.Fs;

% Plotear la señal en el dominio del tiempo y la frecuencia.

% Calcular la transformada de Fourier de la señal
fft_signal = fft(signal);

% Calcular la frecuencia correspondiente a cada componente de la transformada de Fourier
freqs = fftshift((-fs/2):(fs/length(signal)):(fs/2-fs/length(signal)));

% Plotear la señal y su transformada de Fourier
figure;
subplot(2, 1, 1);
plot(signal);
title('Señal original');

subplot(2, 1, 2);
plot(freqs, abs(fft_signal));
xlim([0, fs/10]);
title('Transformada de Fourier de la señal');

% Guardar la imagen en formato PNG
% saveas(gcf, 'Imagen.png');

% Mostrar el gráfico
% pause; % Descomenta esta línea si deseas pausar la ejecución después de mostrar el gráfico

% Obtener las características de la señal

mean_value = mean(signal);
unbias_data = signal - mean_value;
N = numel(unbias_data);

% Calcular power 2 3 4 de los datos, necesarios para los cumulantes
unbias_data_2 = unbias_data .* unbias_data;
unbias_data_3 = unbias_data_2 .* unbias_data;
unbias_data_4 = unbias_data_3 .* unbias_data;
Variance = sum(unbias_data_2) / N;
Skewness = (sum(unbias_data_3) / N) / (Variance^(3/2));
Kurtosis = (sum(unbias_data_4) / N - 3 * (Variance^2)) / (Variance^2);

% Calcular el THD
fund_freq = find(abs(fft_signal(1:length(fft_signal)/2)) == max(abs(fft_signal(1:length(fft_signal)/2))));
fund_freq = fund_freq / length(signal) * fs;
fund_amp = abs(fft_signal(round(fund_freq * length(signal) / fs)));
harm_amps = abs(fft_signal(2:2:round(fs/fund_freq)*2));
thd = sqrt(sum(harm_amps.^2) / fund_amp.^2);

% Calcular el factor de cresta
rms = sqrt(mean(signal.^2));
crest_factor = max(signal) / rms;

% Mostrar los resultados
disp(['Variance: ', num2str(Variance)]);
disp(['Skewness: ', num2str(Skewness)]);
disp(['Kurtosis: ', num2str(Kurtosis)]);
disp(['THD: ', num2str(thd)]);
disp(['Factor de Cresta: ', num2str(crest_factor)]);

% Guardar todas las salidas en un vector que nos servirá para introducirlas en el clasificador
features = zeros(1, 5);
features(1) = Variance;
features(2) = Skewness;
features(3) = Kurtosis;
features(4) = thd;
features(5) = crest_factor;

toc