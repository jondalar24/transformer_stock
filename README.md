# Transformer Time Series Forecasting

Este repositorio contiene una implementación completa de un modelo Transformer para predicción de series temporales, construido con TensorFlow y Keras. Se entrena sobre un dataset sintético de precios bursátiles y se puede adaptar fácilmente a otros conjuntos de datos reales.

---

##  Descripción general del flujo en 8 pasos

### 1. **Generación del Dataset Sintético**
- Se simula una serie temporal con una tendencia lineal ascendente y ruido gaussiano.
- Se guarda como CSV para simular una carga real.

### 2. **Carga y Normalización**
- Se carga el archivo `stock_prices.csv`.
- Se aplica `MinMaxScaler` de Scikit-learn para escalar los valores al rango [0, 1].

### 3. **Preparación del Dataset (Ventanas Temporales)**
- Se generan secuencias de 100 valores (`time_step`) y se asocia cada secuencia con su siguiente valor como etiqueta.
- Este proceso convierte una serie unidimensional en entradas `X` y salidas `Y` para el modelo.

### 4. **Definición de la Capa Multi-Head Attention**
- Se implementa manualmente la atención con múltiples cabezas.
- Permite al modelo enfocar distintos contextos simultáneamente dentro de la secuencia.

### 5. **Bloque Transformer y Encoder**
- Se define un `TransformerBlock` con atención + red feedforward + normalización + dropout.
- Se apilan múltiples bloques para construir el `TransformerEncoder`.

### 6. **Construcción del Modelo Keras**
- Se define la arquitectura:
  - `Input` → `Dense` → `TransformerEncoder` → `Flatten` → `Dropout` → `Dense` (salida).
- Se compila con `Adam` y `loss='mse'` (regresión).

### 7. **Entrenamiento**
- El modelo se entrena durante 20 épocas con `batch_size=32`.
- Puedes modificar el `batch_size` para observar cambios en estabilidad y velocidad.

### 8. **Predicción y Visualización**
- Se invierte la normalización (`inverse_transform`) para comparar con los valores originales.
- Se grafica la serie real y la predicción en un mismo gráfico con `matplotlib`.

---

##  ¿Qué es una ventana temporal?

Una ventana temporal es una forma de preparar los datos para que el modelo pueda aprender relaciones temporales. Por ejemplo:

```text
Input (X):  [ día 1 → día 100 ]
Label (Y):         día 101
```

Con `time_step=100`, el modelo ve 100 valores pasados y aprende a predecir el siguiente.

---

##  Cómo probar el modelo

### 1. Instala los requisitos:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

### 2. Ejecuta el script:
```bash
python transformer_stock_forecast.py
```

Esto:
- Generará el dataset
- Entrenará el modelo
- Mostrará el gráfico comparativo entre predicción y valores reales

---

##  Cómo adaptar el código a otro dataset

1. **Reemplaza `stock_prices.csv` por tu propio archivo** que contenga una columna numérica con la serie temporal.

2. Asegúrate de cambiar esta línea:
```python
data = data[['Close']].values  # cambia 'Close' por tu columna
```

3. El resto del flujo funcionará igual si:
   - Tu serie es continua (por ejemplo, temperaturas, consumo, etc.)
   - Tiene suficientes muestras (mínimo 200–300 para pruebas)

### ❗ Consideraciones clave:
- Escala siempre los datos antes de entrenar.
- Ajusta `time_step` si quieres que el modelo observe más o menos contexto.
- Evalúa visualmente la predicción para detectar overfitting.

---

## ✏️ Modificaciones útiles

- Puedes cambiar la función de activación final a `tanh` si tus datos están centrados:
```python
outputs = Dense(1, activation='tanh')(x)
```

- También puedes usar `validation_split=0.2` durante `model.fit(...)` para evaluar generalización.

---

## 📌 Créditos

Creado para prácticas de aprendizaje profundo con Transformers aplicados a series temporales.\
