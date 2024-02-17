import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader


class SignalDataset(Dataset):
    def __init__(self, data_dir, signal_type):
        self.data_dir = data_dir
        self.signal_type = signal_type
        # print("Directorio de datos:", data_dir)
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        try:
            # Cargar los datos
            signal_data = np.load(file_path)
            return torch.tensor(signal_data, dtype=torch.float32), signal_types.index(self.signal_type)
        except Exception as e:
            print(f"Error al cargar el archivo {file_name}: {e}")
            return None


# Directorio de datos
data_dir = 'data'

# Tipos de señales
signal_types = ['flicker_signals', 'harmonic_signals', 'interruption_signals', 'original_signals', 'sag_signals',
                'swell_signals']

# Crear dataloaders para cada conjunto de datos
train_datasets = {signal_type: SignalDataset(os.path.join(data_dir, 'train', signal_type), signal_type) for signal_type
                  in signal_types if SignalDataset(os.path.join(data_dir, 'train', signal_type), signal_type)}
test_datasets = {signal_type: SignalDataset(os.path.join(data_dir, 'test', signal_type), signal_type) for signal_type in
                 signal_types if SignalDataset(os.path.join(data_dir, 'test', signal_type), signal_type)}
val_datasets = {signal_type: SignalDataset(os.path.join(data_dir, 'val', signal_type), signal_type) for signal_type in
                signal_types if SignalDataset(os.path.join(data_dir, 'val', signal_type), signal_type)}

# Crear dataloaders para cada conjunto de datos
train_dataloaders = {signal_type: DataLoader(dataset, batch_size=32, shuffle=True) for signal_type, dataset in
                     train_datasets.items()}
test_dataloaders = {signal_type: DataLoader(dataset, batch_size=32, shuffle=False) for signal_type, dataset in
                    test_datasets.items()}
val_dataloaders = {signal_type: DataLoader(dataset, batch_size=32, shuffle=False) for signal_type, dataset in
                   val_datasets.items()}

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
        # print("Dimensiones de x:", x.size())  # Agregar esta línea para imprimir las dimensiones de x
        out, _ = self.rnn(x)
        # print(out.shape)
        out = self.fc(out)
        return out


# Parámetros del modelo
input_size = 10000
hidden_size = 128
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

# Inicializar lista para almacenar la historia de la pérdida
loss_history = []

# Entrenar el modelo
num_epochs = 200
total_loaded_files = 0  # Inicializar contador de archivos cargados

for epoch in range(num_epochs):
    for data, labels in model_dataloader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        total_loaded_files += len(labels)  # Incrementar el contador de archivos cargados
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

# Imprimir el número total de archivos cargados durante el entrenamiento
print(f"Total de archivos cargados durante el entrenamiento: {total_loaded_files}")

# Guardar el modelo entrenado con el nombre que incluye la fecha y hora actual
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = f"rnn_model_{current_datetime}.pth"
torch.save(model.state_dict(), model_save_path)

# Hacer predicciones con el modelo entrenado (simulado)
# Cargar algunas señales de la carpeta de validación para usar como nuevas señales
nuevas_señales = []
val_data_dir = 'data/val'

# Recorrer cada tipo de señal en la carpeta de validación
for signal_type in signal_types:
    # Obtener la lista de archivos para el tipo de señal actual
    val_signal_dir = os.path.join(val_data_dir, signal_type)
    val_file_list = [f for f in os.listdir(val_signal_dir) if f.endswith('.npy')]

    # Seleccionar algunos archivos para usar como nuevas señales
    num_signals_to_use = min(5, len(val_file_list))  # Limitar a 5 señales por tipo de señal
    val_file_list = val_file_list[:num_signals_to_use]

    # Cargar los datos de cada archivo y agregarlos a la lista de nuevas señales
    for file_name in val_file_list:
        file_path = os.path.join(val_signal_dir, file_name)
        signal_data = np.load(file_path)
        nuevas_señales.append(signal_data)

# Convertir la lista de arrays de numpy en un tensor de tamaño [batch_size, input_size]
nuevas_señales = torch.tensor(np.array(nuevas_señales), dtype=torch.float32)

# Verificar las dimensiones del tensor
print("Dimensiones de nuevas_señales:", nuevas_señales.size())

# Graficar la pérdida durante el entrenamiento
plt.plot(loss_history)
plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Iteración")
plt.ylabel("Pérdida")
plt.show()


# Función para evaluar el modelo en un conjunto de datos
def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data, labels in dataloader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    average_loss = total_loss / len(dataloader)
    accuracy = (correct_predictions / total_predictions) * 100.0
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    confusion = confusion_matrix(all_labels, all_predictions)

    print(
        f'Average Loss: {average_loss}, Accuracy: {accuracy}%, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
    print('Confusion Matrix:')
    print(confusion)


# Evaluar el modelo en el conjunto de datos de prueba
print("Evaluación en el conjunto de datos de prueba:")
for signal_type, dataloader in test_dataloaders.items():
    print(f"Signal Type: {signal_type}")
    evaluate_model(model, dataloader)

# Evaluar el modelo en el conjunto de datos de validación
print("Evaluación en el conjunto de datos de validación:")
for signal_type, dataloader in val_dataloaders.items():
    print(f"Signal Type: {signal_type}")
    evaluate_model(model, dataloader)
