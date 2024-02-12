import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class SignalDataset(Dataset):
    def __init__(self, data_dir, signal_type):
        self.data_dir = data_dir
        self.signal_type = signal_type
        print("Directorio de datos:", data_dir)
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        try:
            # Cargar los datos
            signal_data = np.load(file_path)
            return torch.tensor(signal_data, dtype=torch.float32), self.signal_type
        except Exception as e:
            print(f"Error al cargar el archivo {file_name}: {e}")
            return None

# Directorio de datos
data_dir = 'data'

# Tipos de señales
signal_types = ['flicker_signals', 'harmonic_signals', 'interruption_signals', 'original_signals', 'sag_signals', 'swell_signals']

# Crear dataloaders para cada conjunto de datos
train_datasets = {signal_type: SignalDataset(os.path.join(data_dir, 'train', signal_type), signal_type) for signal_type in signal_types if SignalDataset(os.path.join(data_dir, 'train', signal_type), signal_type)}
test_datasets = {signal_type: SignalDataset(os.path.join(data_dir, 'test', signal_type), signal_type) for signal_type in signal_types if SignalDataset(os.path.join(data_dir, 'test', signal_type), signal_type)}
val_datasets = {signal_type: SignalDataset(os.path.join(data_dir, 'val', signal_type), signal_type) for signal_type in signal_types if SignalDataset(os.path.join(data_dir, 'val', signal_type), signal_type)}

# Crear dataloaders para cada conjunto de datos
train_dataloaders = {signal_type: DataLoader(dataset, batch_size=32, shuffle=True) for signal_type, dataset in train_datasets.items()}
test_dataloaders = {signal_type: DataLoader(dataset, batch_size=32, shuffle=False) for signal_type, dataset in test_datasets.items()}
val_dataloaders = {signal_type: DataLoader(dataset, batch_size=32, shuffle=False) for signal_type, dataset in val_datasets.items()}

# Visualizar un ejemplo de datos de entrenamiento para la señal flicker_signals
for signal_type, dataset in train_datasets.items():
    if len(dataset) > 0:
        sample_data, sample_type = dataset[0]
        print(f"Ejemplo de datos de entrenamiento para la señal {signal_type}:")
        print(sample_data)
        print(sample_type)
        break  # Mostramos solo un ejemplo para la primera señal

# Crear modelo RNN
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        print("Dimensiones de x:", x.size())  # Agregar esta línea para imprimir las dimensiones de x
        out, _ = self.rnn(x)
        print(out.shape)
        out = self.fc(out)
        return out

# Parámetros del modelo
input_size = 10000
hidden_size = 64
output_size = len(signal_types)

# Crear dataset y dataloaders para el modelo RNN
model_dataset = SignalDataset(os.path.join(data_dir, 'train', signal_types[0]), signal_types[0])
model_dataset = [data for data in model_dataset if data is not None]  # Filtrar datos nulos
model_dataloader = DataLoader(model_dataset, batch_size=32, shuffle=True)


# Crear instancia del modelo RNN
model = RNNModel(input_size, hidden_size, output_size)

# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar el modelo
num_epochs = 200
loss_history = []
for epoch in range(num_epochs):
    for data, labels in model_dataloader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, torch.tensor([signal_types.index(t) for t in labels]))
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Guardar el modelo entrenado
torch.save(model.state_dict(), "rnn_model.pth")

# Hacer predicciones con el modelo entrenado (simulado)
# Cargar algunas señales de la carpeta de validación para usar como nuevas señales
nuevas_señales = []
val_data_dir = 'data/val'
val_file_list = [f for f in os.listdir(val_data_dir) if f.endswith('.npy')]
num_signals_to_use = 5  # Por ejemplo, puedes ajustar este valor según tus necesidades
val_file_list = val_file_list[:num_signals_to_use]

for file_name in val_file_list:
    file_path = os.path.join(val_data_dir, file_name)
    signal_data = np.load(file_path)
    nuevas_señales.append(signal_data)

# Convertir la lista de arrays de numpy en un tensor de tamaño [batch_size, input_size]
nuevas_señales = torch.tensor(nuevas_señales, dtype=torch.float32)

# Verificar las dimensiones del tensor
print("Dimensiones de nuevas_señales:", nuevas_señales.size())


# Graficar la pérdida durante el entrenamiento
plt.plot(loss_history)
plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Iteración")
plt.ylabel("Pérdida")
plt.show()
