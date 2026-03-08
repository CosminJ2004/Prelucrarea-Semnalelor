import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

# 1. Datele: Antrenament (Istoric) vs Test (Prezent 2025-2026)
df_train = yf.download('^GSPC', start='2003-01-01', end='2015-01-01', progress=False)
df_test = yf.download('^GSPC', start='2000-01-01', end='2026-03-01', progress=False)

# CALCULĂM RANDAMENTELE (Diferența procentuală de la o zi la alta)
# Asta face ca datele din 2003 să "arate" la fel ca cele din 2026
train_returns = df_train['Close'].pct_change().dropna().values.reshape(-1, 1)
test_returns = df_test['Close'].pct_change().dropna().values.reshape(-1, 1)

# Scalare (StandardScaler e mai bun aici pentru că randamentele au distribuție normală)
scaler = StandardScaler()
scaled_train = scaler.fit_transform(train_returns)
scaled_test = scaler.transform(test_returns)

# [Image of price series vs log returns visualization]

def create_sequences(data, window=20): # Fereastră de 20 de zile (o lună de trading)
    x = []
    for i in range(len(data) - window):
        x.append(data[i:i+window])
    return np.array(x)

window_size = 60
X_train = create_sequences(scaled_train, window_size)
X_test = create_sequences(scaled_test, window_size)

# 2. Modelul LSTM Autoencoder
model = models.Sequential([
    layers.LSTM(16, activation='tanh', input_shape=(window_size, 1), return_sequences=False),
    layers.RepeatVector(window_size),
    layers.LSTM(16, activation='tanh', return_sequences=True),
    layers.TimeDistributed(layers.Dense(1))
])

model.compile(optimizer='adam', loss='mse')
print("Antrenăm pe randamente (Log-Returns)...")
model.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)

# 3. Pragul de anomalie (bazat pe erorile de antrenament)
train_pred = model.predict(X_train, verbose=0)
train_mse = np.mean(np.square(X_train - train_pred), axis=(1,2))
threshold = np.percentile(train_mse, 98) # Acceptăm 2% anomalii în trecut

# 4. Evaluare pe prezent (2025-2026)
test_pred = model.predict(X_test, verbose=0)
test_mse = np.mean(np.square(X_test - test_pred), axis=(1,2))
anomalies = test_mse > threshold

# 5. Vizualizare
plt.figure(figsize=(16, 8))
test_dates = df_test.index[window_size+1:] # +1 din cauza pct_change
test_prices_real = df_test['Close'].values[window_size+1:]

plt.plot(test_dates, test_prices_real, label='S&P 500 (Prezent)', color='darkblue', alpha=0.7)
plt.scatter(test_dates[anomalies], test_prices_real[anomalies], 
            color='red', label='Anomalie Volatilitate', s=25)

plt.title('Detecția Anomaliilor pe Randamente (LSTM Autoencoder)')
plt.grid(True, alpha=0.2)
plt.legend()
plt.show()