# Transformer for Time Series Forecasting
# ---------------------------------------------------
# This script generates a synthetic stock price dataset,
# trains a Transformer-based model, and visualizes the results.
# Suitable for educational purposes.

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout

# ----------------------------
# 1. Generate synthetic dataset
# ----------------------------
np.random.seed(42)
data_length = 2000
trend = np.linspace(100, 200, data_length)  # linear upward trend
noise = np.random.normal(0, 2, data_length)  # gaussian noise
synthetic_data = trend + noise

# Save to CSV and reload (simulating real-world pipeline)
pd.DataFrame({'Close': synthetic_data}).to_csv('stock_prices.csv', index=False)
data = pd.read_csv('stock_prices.csv')
data = data[['Close']].values  # shape: (2000, 1)

# ----------------------------
# 2. Normalize the data
# ----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# ----------------------------
# 3. Create time series sequences (windows)
# ----------------------------
def create_dataset(data, time_step=100):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])  # shape: (100,)
        Y.append(data[i + time_step, 0])      # scalar
    return np.array(X), np.array(Y)

time_step = 100
X, Y = create_dataset(data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)  # shape: (samples, time_step, features)

# ----------------------------
# 4. Define Multi-Head Attention layer
# ----------------------------
class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        attention, _ = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

# ----------------------------
# 5. Define Transformer Block
# ----------------------------
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# ----------------------------
# 6. Define Transformer Encoder
# ----------------------------
class TransformerEncoder(Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.enc_layers = [
            TransformerBlock(embed_dim, num_heads, ff_dim, rate)
            for _ in range(num_layers)
        ]
        self.dropout = Dropout(rate)

    def call(self, inputs, training=False):
        x = inputs
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training)
        return x

# ----------------------------
# 7. Build the model
# ----------------------------
embed_dim = 128
num_heads = 8
ff_dim = 512
num_layers = 4

inputs = tf.keras.Input(shape=(time_step, 1))
x = Dense(embed_dim)(inputs)
encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim)
x = encoder(x)
x = tf.keras.layers.Flatten()(x)
x = Dropout(0.5)(x)
outputs = Dense(1)(x)  # you could also use activation="tanh" here
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='mse')
model.summary()

# ----------------------------
# 8. Train the model
# ----------------------------
model.fit(X, Y, epochs=20, batch_size=32)

# ----------------------------
# 9. Make and plot predictions
# ----------------------------
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)
true_values = scaler.inverse_transform(data.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(true_values, label='True Data')
plt.plot(np.arange(time_step, time_step + len(predictions)), predictions, label='Predictions')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Transformer Model Predictions vs True Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
