import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Generăm seria (N mai mare, ex: 500)
N = 300
np.random.seed(42)
t = np.linspace(0, N-1, N)
trend = 0.00001*t**2 - 0.00002*t + 0.1
seasonal = 0.1 * np.sin(2*np.pi*0.01*t) + 0.2 * np.sin(2*np.pi*0.005*t)
noise = np.random.normal(0, 0.2, N)
series = trend + seasonal + noise

# Split train/test
train_size = int(0.8 * N)
train, test = series[:train_size], series[train_size:]

# Funcție pentru mediere exponențială
def exp_smooth(series, alpha):
    s_smooth = np.zeros_like(series)
    s_smooth[0] = series[0]
    for t in range(1, len(series)):
        s_smooth[t] = alpha * series[t-1] + (1 - alpha) * s_smooth[t-1]
    return s_smooth


def find_best_alpha(train, test):
    alphas = np.linspace(0.01, 0.99, 99)
    best_alpha = 0
    best_mse = float('inf')
    for alpha in alphas:
        smooth_train = exp_smooth(train, alpha)
        history = smooth_train[-1]
        preds = []
        for t in range(len(test)):
            pred = alpha * history + (1 - alpha) * history
            preds.append(pred)
            history = test[t]  # actualizare cu valoarea reala
        mse = mean_squared_error(test, preds)
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    return best_alpha

# α optim pentru mediere simpla
alpha_simple = find_best_alpha(train, test)
smooth_simple = exp_smooth(series, alpha_simple)

# α optim pentru mediere dubla
smooth_double = exp_smooth(smooth_simple, alpha_simple)

# α optim pentru mediere tripla
smooth_triple = exp_smooth(smooth_double, alpha_simple)

# Plot comparativ
plt.figure(figsize=(12,8))
plt.plot(series, label="Seria originala", alpha=0.6)
plt.plot(smooth_simple, label="Mediere exponențială simpla", linestyle='--')
plt.plot(smooth_double, label="Mediere exponentiala dubla", linestyle='-.')
plt.plot(smooth_triple, label="Mediere exponentiala tripla", linestyle=':')
plt.legend()
plt.title("Suavizarea exponentiala: simpla, dubla si tripla (a optim)")
plt.show()

print(f" optim folosit: {alpha_simple:.3f}")
