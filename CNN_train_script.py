import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Definir el dataset personalizado para cargar las señales
class SignalDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.signal_types = ['flicker_signals', 'harmonic_signals', 'interruption_signals',
                             'original_signals', 'sag_signals', 'swell_signals']
        self.file_paths = []
        for signal_type in self.signal_types:
            type_dir = os.path.join(data_dir, signal_type)
            files = [os.path.join(type_dir, f) for f in os.listdir(type_dir) if f.endswith('.npy')]
            self.file_paths.extend(files)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        signal_data = np.load(file_path)
        signal_label = self.get_signal_label(file_path)
        return signal_data, signal_label

    def get_signal_label(self, file_path):
        signal_type = file_path.split('/')[-2]
        return self.signal_types.index(signal_type)

# Directorios de datos
train_data_dir = 'data/train'
test_data_dir = 'data/test'
val_data_dir = 'data/val'

# Crear datasets y dataloaders para entrenamiento, prueba y validación
train_dataset = SignalDataset(train_data_dir)
test_dataset = SignalDataset(test_data_dir)
val_dataset = SignalDataset(val_data_dir)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Definir la arquitectura de la red convolucional
class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 2500, 128)  # sequence_length calculada dinámicamente
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x.float()))  # Convertir datos de entrada a float
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # Calcular la longitud de la secuencia dinámicamente
        sequence_length = x.size(2)
        x = x.view(x.size(0), -1)  # Aplanar la salida de las capas convolucionales
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Parámetros del modelo
input_channels = 1  # Número de canales de entrada (1 para señales unidimensionales)
num_classes = len(train_dataset.signal_types)  # Número de clases (tipos de señales)

# Instanciar el modelo CNN
model = CNN(input_channels, num_classes)


# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar el modelo
num_epochs = 1
loss_history = []
total_iterations = num_epochs * len(train_dataloader)
completed_iterations = 0
for epoch in range(num_epochs):
    for i, (signals, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(signals.unsqueeze(1))  # Agregar dimensión de canal a las señales
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        # Incrementar el número de iteraciones completadas
        completed_iterations += 1

        # Calcular el porcentaje de entrenamiento completado
        progress_percentage = (completed_iterations / total_iterations) * 100

        # Imprimir el progreso
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Iteration [{i + 1}/{len(train_dataloader)}], Progress: {progress_percentage:.2f}%')

print("Entrenamiento completado.")

# Evaluar el modelo en el conjunto de validación
correct_per_class = {signal_type: 0 for signal_type in val_dataset.signal_types}
total_per_class = {signal_type: 0 for signal_type in val_dataset.signal_types}

with torch.no_grad():
    for signals, labels in val_dataloader:
        outputs = model(signals.unsqueeze(1))  # Agregar dimensión de canal a las señales
        _, predicted = torch.max(outputs, 1)
        for pred, true_label in zip(predicted, labels):
            true_signal_type = val_dataset.signal_types[true_label]
            total_per_class[true_signal_type] += 1
            if pred == true_label:
                correct_per_class[true_signal_type] += 1

print("Accuracy por tipo de señal en el conjunto de validación:")
for signal_type in val_dataset.signal_types:
    accuracy = correct_per_class[signal_type] / total_per_class[signal_type] * 100
    print(f"{signal_type}: {accuracy:.2f}%")

# Graficar la pérdida durante el entrenamiento
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
