import numpy as np
import torch
import time
import os
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from joblib import dump
from src.classes.MultipleRegressionModel import MultipleRegressionModel  # Ajusta esta importación según tu estructura de directorios

# Cargar tensores y convertir a numpy
X_tensor = torch.load('X_tensor.pth')
y_tensor = torch.load('y_tensor.pth')
X = X_tensor.numpy().reshape((X_tensor.shape[0], -1))  # Aplana las imágenes
y = y_tensor.numpy()

# División entre entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar modelo
model = MultipleRegressionModel()

# Inicializar variables para early stopping
patience = 10
best_val_loss = float('inf')
counter = 0

# Registro del tiempo de inicio
start_time = time.time()

# Ciclo de entrenamiento
for epoch in range(100):  # Puedes ajustar el número de épocas
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_loss = mean_absolute_error(y_val, val_pred)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # Guardar el mejor modelo
        dump(model, 'best_MultipleRegressionModel.pkl')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# Registro del tiempo de finalización y cálculo del tiempo de entrenamiento
end_time = time.time()
training_time = end_time - start_time

# Métricas de desempeño
mae = mean_absolute_error(y_val, val_pred)

# Generar informe
report = f"Model: MultipleRegressionModel\nTraining Time: {training_time}s\nMAE: {mae}"
print(report)

# Guardar el informe en un archivo

# Asegurar que el directorio 'reports' exista
if not os.path.exists('reports'):
    os.makedirs('reports')

file_number = 0
report_filename_base = f"reports/MultipleRegressionModel_report{file_number}.txt"

while os.path.exists(report_filename_base):
    file_number += 1
    report_filename_base = f"reports/MultipleRegressionModel_report{file_number}.txt"

with open(report_filename_base, "w") as f:
    f.write(report)
