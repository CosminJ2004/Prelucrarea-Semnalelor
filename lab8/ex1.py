import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

# -----------------------------
# GENERARE SERIE DE TIMP
# -----------------------------

N = 1000
np.random.seed(42)

t = np.linspace(0, N-1, N)

# Trend: polinom de grad 2
a, b, c =  0.000001, -0.000002, -0.000003
trend = a * t**2 + b * t + c

 
period1 = 0.01   
period2 = 0.005

seasonal = 0.1 * np.sin(2 * np.pi * t * period1) + \
           0.2 * np.sin(2 * np.pi * t * period2)

# Zgomot alb gaussian
sigma = 0.2
noise = np.random.normal(0, sigma, N)

# Seria totală
series = trend + seasonal + noise

# -----------------------------
# PLOTURI
# -----------------------------

plt.figure(figsize=(14, 10))

# 1. Seria totală
plt.subplot(4, 1, 1)
plt.plot(t, series)
plt.title("Seria de timp totală (N=1000)")
plt.grid(True)

# 2. Trend
plt.subplot(4, 1, 2)
plt.plot(t, trend, color="orange")
plt.title("Componenta trend (polinom grad 2)")
plt.grid(True)

# 3. Sezon
plt.subplot(4, 1, 3)
plt.plot(t, seasonal, color="green")
plt.title("Componenta sezonieră (2 sinusoide)")
plt.grid(True)

# 4. Zgomot
plt.subplot(4, 1, 4)
plt.plot(t, noise, color="red")
plt.title("Componenta zgomot (Gaussian)")
plt.grid(True)

plt.tight_layout()
# plt.show()


def autocorrelation(x):
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode='full')
    corr = corr[corr.size // 2:]      # păstrăm doar lag >= 0
    return corr / corr[0]             # normalizare (rho(k))


# Calculează autocorelația
rho = autocorrelation(series)

# Vectorul lagurilor
lags = np.arange(len(rho))

# Plot autocorelație
plt.figure(figsize=(12,4))
plt.stem(lags, rho)
plt.title("Autocorelația seriei")
plt.xlabel("Lag")
plt.ylabel("rho(k)")
plt.grid(True)
# plt.show()



p = 5

X = np.column_stack([series[i:len(series)-p+i] for i in range(p)])
y = series[p:]

phi = np.linalg.lstsq(X, y, rcond=None)[0]
print("Coeficienti AR(p):", phi)

predictions = X @ phi

plt.figure(figsize=(12,6))
plt.plot(series, label="Seria originala")
plt.plot(range(p, len(series)), predictions, linestyle='--', label="Predicții AR(p) manual")
plt.xlabel("Timp")
plt.ylabel("Valoare")
plt.title("Seria originala vs Predictii AR")
plt.legend()
plt.show()



N = len(series)
max_p = 20 
best_p = 0
best_mse = float('inf')


train_size = int(0.8 * N)
train, test = series[:train_size], series[train_size:]


def ar_predict_one_step(train_series, p):
    X = np.column_stack([train_series[i:len(train_series)-p+i] for i in range(p)])
    y = train_series[p:]
    phi = np.linalg.lstsq(X, y, rcond=None)[0]
    return phi


for p in range(1, max_p+1):
    if len(train) <= p:
        break
    phi = ar_predict_one_step(train, p)
    # Predictii pe test (un pas înainte)
    preds = []
    history = train[-p:].tolist()  
    for t in range(len(test)):
        x_input = np.array(history[-p:])
        y_pred = np.dot(phi, x_input)
        preds.append(y_pred)
        history.append(test[t]) 
    mse = mean_squared_error(test, preds)
    if mse < best_mse:
        best_mse = mse
        best_p = p

print(f"Cel mai bun p: {best_p}, MSE: {best_mse}")