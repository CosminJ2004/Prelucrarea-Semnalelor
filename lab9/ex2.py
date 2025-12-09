import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


N = 400
np.random.seed(42)

t = np.linspace(0, N-1, N)

# Trend: polinom de grad 2
a, b, c =  0.000001, -0.000002, -0.000003
trend = a * t**2 + b * t + c

 
period1 = 0.01   
period2 = 0.005

seasonal = 0.1 * np.sin(2 * np.pi * t * period1) + \
           0.2 * np.sin(2 * np.pi * t * period2)

sigma = 0.01
noise = np.random.normal(0, sigma, N)


series = trend + seasonal + noise



q = 5  # orizontul modelului MA

# Calculam media seriei
mu = np.mean(series)

# Termeni de eroare: deviatia fata de medie
epsilon = series - mu

ma_series = np.zeros_like(series)

theta = np.random.uniform(-0.5, 0.5, q)


for t in range(len(series)):
    ma_series[t] = mu + epsilon[t]  # termenul curent
    for k in range(1, q+1):
        if t-k >= 0:
            ma_series[t] += theta[k-1] * epsilon[t-k]

# Plot pentru comparatie
plt.figure(figsize=(12,6))
plt.plot(series, label="Seria originala")
plt.plot(ma_series, label=f"Seria MA({q})")
plt.legend()
plt.title(f"Model MA({q}) aplicat seriei de timp")
plt.show()


