import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.linear_model import LinearRegression

# --- 1. SSA ENGINE (Trend Extraction) ---
def ssa_trend_direct(series, L=50):
    series_np = series.values if hasattr(series, 'values') else series
    N = len(series_np)
    series_padded = np.pad(series_np, (L, L), mode='reflect')
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

# --- 2. OU ENGINE (Ornstein-Uhlenbeck Calibration) ---
def fit_ornstein_uhlenbeck(residuals, dt=1.0):
    """
    Calibreaza procesul OU pe reziduuri pentru a gasi parametrii globali.
    Returneaza: Theta (viteza), Mu (media), Sigma_Eq (Deviatia standard de echilibru)
    """
    x_t = residuals[:-1].reshape(-1, 1)
    x_t1 = residuals[1:].reshape(-1, 1)
    
    reg = LinearRegression().fit(x_t, x_t1)
    a = reg.coef_[0][0]
    b = reg.intercept_[0]
    
    # Reziduurile regresiei (pentru zgomotul instantaneu)
    epsilon = x_t1 - reg.predict(x_t)
    sigma_epsilon = np.std(epsilon)
    
    # Conversie la parametri OU
    theta = -np.log(a) / dt
    mu = b / (1 - a)
    
    # Sigma de Echilibru (Limita teoretica pe termen lung)
    # Formula: Sigma_Eq = Sigma_instant / sqrt(2 * theta)
    # Daca theta e foarte mic (aproape de random walk), sigma_eq explodeaza
    if theta < 1e-5: theta = 1e-5
    
    sigma_ou_instant = sigma_epsilon * np.sqrt(-2 * np.log(a) / (1 - a**2) * dt)
    sigma_eq = sigma_ou_instant / np.sqrt(2 * theta)
    
    return theta, mu, sigma_eq

# --- 3. PREPARARE DATE ---
print("Descarcare date S&P 500...")
df = yf.download("^GSPC", start="2019-01-01", end="2024-01-01", progress=False)
price = df['Close']
if hasattr(price, 'squeeze'): price = price.squeeze()
price_values = price.values

# A. SSA Trend
print("Extragere Trend SSA...")
trend = ssa_trend_direct(price_values, L=60)
residuals = price_values - trend

# --- 4. CALIBRARE ORNSTEIN-UHLENBECK (Global Physics) ---
print("Calibrare OU pe Reziduuri...")
theta, mu, sigma_eq = fit_ornstein_uhlenbeck(residuals)

# Definim Tunelul OU (Static / Echilibru pe termen lung)
# De obicei 3 sigma echilibru acopera 99.7% din cazuri intr-o lume ideala
ou_upper = mu + 3 * sigma_eq
ou_lower = mu - 3 * sigma_eq

print(f"--- Parametri OU ---")
print(f"Theta (Viteza revenire): {theta:.4f}")
print(f"Sigma Echilibru (Limita Statica): {sigma_eq:.2f}")

# --- 5. GARCH ENGINE (Local Volatility) ---
print("Antrenare GARCH...")
scale_factor = 100
residuals_scaled = residuals / scale_factor

garch = arch_model(residuals_scaled, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
res_garch = garch.fit(disp='off')
conditional_vol = res_garch.conditional_volatility * scale_factor

# Definim Tunelul GARCH (Dinamic)
z_score_threshold = 3.0
garch_upper = z_score_threshold * conditional_vol
garch_lower = -z_score_threshold * conditional_vol

# --- 6. DETECTIE SI COMPARATIE ---
# Anomaliile sunt punctele care sparg GARCH-ul (cel mai permisiv model)
anomalies_mask = (residuals > garch_upper) | (residuals < garch_lower)
anomalies = residuals[anomalies_mask]
anomaly_dates = df.index[anomalies_mask]

# --- 7. VIZUALIZARE COMPARATIVA ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# Panel 1: Pret si Anomalii
ax1.plot(df.index, price, color='black', alpha=0.3, label='Pret Real')
ax1.plot(df.index, trend, color='blue', linewidth=2, label='SSA Trend')
ax1.scatter(anomaly_dates, price[anomaly_dates], color='red', s=50, zorder=5, label='Anomalii (GARCH Breakout)')
ax1.set_title(f"1. S&P 500: SSA Trend + Anomalii Hibride")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Bătălia Modelelor (OU vs GARCH)
ax2.plot(df.index, residuals, color='purple', alpha=0.5, label='Reziduuri (Zgomot SSA)')

# TUNELUL OU (Verde - Static)
ax2.axhline(ou_upper, color='green', linestyle='--', linewidth=2, label=f'Limita OU (Termen Lung +/- 3$\sigma_{{eq}}$)')
ax2.axhline(ou_lower, color='green', linestyle='--', linewidth=2)
ax2.fill_between(df.index, ou_lower, ou_upper, color='green', alpha=0.05)

# TUNELUL GARCH (Portocaliu - Dinamic)
ax2.plot(df.index, garch_upper, color='orange', linestyle='-', linewidth=2, label='Limita GARCH (Adaptivă)')
ax2.plot(df.index, garch_lower, color='orange', linestyle='-', linewidth=2)
ax2.fill_between(df.index, garch_lower, garch_upper, color='orange', alpha=0.1)

ax2.set_title("2. Comparație: Limita OU (Statică/Verde) vs. Limita GARCH (Dinamică/Portocalie)")
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Anomalii detectate: {len(anomalies)}")