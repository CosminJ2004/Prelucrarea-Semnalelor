import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, DotProduct

# --- PASUL 1 (a din laborator): Date Lunare ---
print("1. Descarcare si Resampling (Lunar)...")
df = yf.download("^GSPC", start="2015-01-01", end="2024-01-01", progress=False)
# 'M' inseamna Month end frequency - facem media lunara ca in laborator
df_monthly = df['Close'].resample('ME').mean()

# Pregatim datele pentru Scikit-Learn (X trebuie sa fie 2D array)
X = np.arange(len(df_monthly)).reshape(-1, 1) # Timpul (luni numerotate 0, 1, 2...)
y = df_monthly.values.reshape(-1, 1)          # Pretul

# --- PASUL 2 (b din laborator): Calcul Trend si Eliminare ---
print("2. Calcul Trend Liniar si Detrending...")
# Folosim Regresie Liniara simpla pentru Trend (Drift)
lr = LinearRegression()
lr.fit(X, y)
trend = lr.predict(X)

# Eliminam trendul pentru a ramane cu "Zgomotul" (Componenta Browniana/Stocastica)
residuals = y - trend

# --- PASUL 3 (c din laborator): GP Regression pe ultimele 12 luni ---
print("3. Proces Gaussian pe Reziduuri...")

# Definim un Kernel. 
# La laborator probabil ati folosit RBF (Squared Exponential).
# Aici folosim: Constant * RBF + WhiteNoise (pentru a permite zgomot la bursa)
kernel = C(1.0) * RBF(length_scale=10.0) + WhiteKernel(noise_level=1)

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

# Antrenam pe TOATE datele fara trend (nu doar ultimele 12, ca sa invete volatilitatea istorica)
gp.fit(X, residuals)

# Facem predictie pentru viitor (urmatoarele 12 luni)
X_future = np.arange(len(X) + 12).reshape(-1, 1)
y_pred_resid, sigma = gp.predict(X_future, return_std=True)

# Re-adaugam Trendul (extrapolat) pentru a vedea pretul final
# Trebuie sa prezicem trendul liniar si pentru viitor
trend_future = lr.predict(X_future)
y_pred_final = trend_future + y_pred_resid.reshape(-1, 1)

# --- VIZUALIZARE ---
plt.figure(figsize=(14, 8))

# 1. Datele Reale
plt.plot(X, y, 'k.', markersize=10, label='Date Reale (Medie Lunara)')

# 2. Trendul Liniar
plt.plot(X_future, trend_future, 'b--', label='Trend Global (Drift)', alpha=0.5)

# 3. Predictia GP (Pret + Incertitudine)
plt.plot(X_future, y_pred_final, 'r-', label='Predictie GP (Brownian Logic)')

# Zona de incredere (95% - 1.96 sigma)
# Aici se vede "pâlnia" specifica miscarii browniene (incertitudinea creste)
upper = (y_pred_final + 1.96 * sigma.reshape(-1, 1)).flatten()
lower = (y_pred_final - 1.96 * sigma.reshape(-1, 1)).flatten()

plt.fill_between(X_future.flatten(), lower, upper, alpha=0.2, color='red', label='Interval de Incredere (Volatilitate)')

plt.title("Aplicarea Logicii 'Mauna Loa' pe S&P 500 (Trend + Proces Gaussian)")
plt.xlabel("Luni de la start")
plt.ylabel("Pret ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()