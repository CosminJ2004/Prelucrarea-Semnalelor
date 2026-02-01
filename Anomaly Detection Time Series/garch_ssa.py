import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# --- 1. SSA ENGINE (Trend Extraction) ---
def ssa_trend_direct(series, L=50):
    # (Codul SSA standard pe care l-am validat deja)
    N = len(series)
    series_padded = np.pad(series, (L, L), mode='reflect')
    N_pad = len(series_padded)
    K = N_pad - L + 1
    X = np.column_stack([series_padded[i : i + L] for i in range(K)])
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    X_trend = Sigma[0] * np.outer(U[:, 0], VT[0, :])
    
    rows, cols = X_trend.shape
    reconstructed = np.zeros(N_pad)
    count = np.zeros(N_pad)
    for r in range(rows):
        for c in range(cols):
            reconstructed[r+c] += X_trend[r, c]
            count[r+c] += 1
            
    trend = (reconstructed / count)[L : -L]
    if len(trend) > N: trend = trend[:N]
    return trend

# --- 2. PREPARARE DATE ---
print("Descarcare date S&P 500...")
df = yf.download("^GSPC", start="2018-01-01", end="2024-01-01", progress=False)
price = df['Close']
if hasattr(price, 'squeeze'): price = price.squeeze()
price_values = price.values

# A. Calculam Trendul SSA
print("Extragere Trend SSA...")
trend = ssa_trend_direct(price_values, L=60)

# B. Calculam Reziduurile (Noise)
residuals = price_values - trend

# --- 3. GARCH ENGINE (Dynamic Volatility) ---
print("Antrenare GARCH pe Reziduuri...")

# NOTA IMPORTANTA: GARCH lucreaza prost cu numere mari (mii de dolari).
# Solutia inginereasca: Scalam reziduurile la procente sau le impartim la 100 pt stabilitate.
scale_factor = 100
residuals_scaled = residuals / scale_factor

# Definim modelul GARCH(1,1) pe reziduuri
# Mean='Zero' pentru ca am scos deja trendul cu SSA, deci media e aprox 0.
garch = arch_model(residuals_scaled, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
res_garch = garch.fit(disp='off')

# Extragem Volatilitatea Conditionata (Sigma_t) si o rescalam inapoi
conditional_vol = res_garch.conditional_volatility * scale_factor

# --- 4. CONSTRUCTIA TUNELULUI DINAMIC ---
# Definim pragul de anomalie (ex: 3 Sigma)
z_score_threshold = 3.0

# Limitele Dinamice
upper_band = z_score_threshold * conditional_vol
lower_band = -z_score_threshold * conditional_vol

# --- 5. DETECTIE ANOMALII ---
# O anomalie e cand Reziduul real sparge limita GARCH
anomalies_mask = (residuals > upper_band) | (residuals < lower_band)
anomalies = residuals[anomalies_mask]
anomaly_dates = df.index[anomalies_mask]

# --- 6. VIZUALIZARE ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# Panel 1: Context General
ax1.plot(df.index, price, color='black', alpha=0.3, label='Pret Real')
ax1.plot(df.index, trend, color='blue', linewidth=2, label='SSA Trend')
# Punem si punctele de anomalie pe graficul de pret
ax1.scatter(anomaly_dates, price[anomaly_dates], color='red', s=50, zorder=5, label='Anomalii Validate (SSA+GARCH)')
ax1.set_title("1. S&P 500: Trend SSA + Anomalii filtrate prin GARCH")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Radiografia Anomaliilor (Respiratia Pietei)
ax2.plot(df.index, residuals, color='purple', alpha=0.5, label='Reziduuri SSA (Zgomot)')

# Desenam Tunelul GARCH (Care respira!)
ax2.plot(df.index, upper_band, color='orange', linestyle='--', linewidth=1.5, label=f'Limita Dinamica GARCH (+/- {z_score_threshold}$\sigma_t$)')
ax2.plot(df.index, lower_band, color='orange', linestyle='--', linewidth=1.5)

# Coloram zona dintre benzi
ax2.fill_between(df.index, lower_band, upper_band, color='orange', alpha=0.1)

# Punctele Rosii
ax2.scatter(anomaly_dates, anomalies, color='red', s=30, zorder=5, label='Incalcare Limita Dinamica')

ax2.set_title("2. Tunelul de Volatilitate: Observati cum limitele se lărgesc automat în crize (2020, 2022)")
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Numar total anomalii detectate: {len(anomalies)}")
print(f"Procent zile anormale: {len(anomalies)/len(df)*100:.2f}%")