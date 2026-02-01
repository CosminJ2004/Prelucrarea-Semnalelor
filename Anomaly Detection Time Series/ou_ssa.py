import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- 1. FUNCTIA SSA (Direct Method - deja validata) ---
def ssa_trend_direct(series, L=50):
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

# --- 2. CALIBRARE ORNSTEIN-UHLENBECK ---
def fit_ornstein_uhlenbeck(residuals, dt=1.0):
    """
    Estimeaza parametrii OU (Theta, Mu, Sigma) din datele reziduale
    folosind Regresie Liniara pe formula discreta:
    x(t+1) = a * x(t) + b + e
    """
    x_t = residuals[:-1].reshape(-1, 1) # Azi
    x_t1 = residuals[1:].reshape(-1, 1) # Maine
    
    # Regresie x(t) -> x(t+1)
    reg = LinearRegression().fit(x_t, x_t1)
    
    a = reg.coef_[0][0]  # Panta (autoregressive)
    b = reg.intercept_[0] # Intercept
    
    # Calculam erorile (reziduurile regresiei) pentru a gasi sigma zgomotului
    pred_x_t1 = reg.predict(x_t)
    epsilon = x_t1 - pred_x_t1
    sigma_epsilon = np.std(epsilon)
    
    # Conversie la parametrii fizici OU: dX = theta(mu - X)dt + sigma dW
    # Relatia matematica intre discret si continuu:
    # a = exp(-theta * dt) => theta = -ln(a) / dt
    theta = -np.log(a) / dt
    
    # mu = b / (1 - a)
    mu = b / (1 - a)
    
    # sigma_ou = sigma_epsilon / sqrt((1 - exp(-2*theta*dt)) / (2*theta))
    # Pentru dt mic, simplificam: sigma_ou approx sigma_epsilon / sqrt(dt)
    sigma_ou = sigma_epsilon * np.sqrt( -2 * np.log(a) / (1 - a**2) * dt )
    
    return theta, mu, sigma_ou

# --- 3. EXECUTIE ---
print("Descarcare date S&P 500...")
df = yf.download("^GSPC", start="2000-01-01", end="2024-01-01", progress=False)
price = df['Close']
if hasattr(price, 'squeeze'): price = price.squeeze()
price = price.values

# A. Extragem Trendul SSA
print("Calculare SSA Trend...")
trend = ssa_trend_direct(price, L=60)

# B. Calculam Reziduul (Zgomotul) - Asta e candidatul OU
residuals = price - trend

# C. Calibram Modelul OU pe Reziduuri
print("Calibrare parametri Ornstein-Uhlenbeck...")
theta, mu, sigma_ou = fit_ornstein_uhlenbeck(residuals)

# D. Calculam Deviatia Standard de Echilibru (Limita teoretica a procesului)
# La infinit, varianta unui proces OU este sigma^2 / (2 * theta)
std_equilibrium = sigma_ou / np.sqrt(2 * theta)

print(f"\n--- REZULTATE CALIBRARE OU ---")
print(f"Theta (Viteza de revenire): {theta:.4f} (Cu cat e mai mare, cu atat elasticul e mai tare)")
print(f"Mu (Media): {mu:.2f} (Ar trebui sa fie aproape de 0)")
print(f"Sigma (Volatilitatea Instantanee): {sigma_ou:.2f}")
print(f"Sigma Echilibru (Limita pe termen lung): {std_equilibrium:.2f}")

# E. VIZUALIZARE SI DETECTIE
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Panel 1: Pret + SSA
ax1.plot(df.index, price, 'k.', alpha=0.3, label='Pret Real')
ax1.plot(df.index, trend, 'b-', linewidth=2, label='SSA Trend (Componenta Neteda)')
ax1.set_title("1. Descompunerea: Scoaterea Trendului pentru a izola Procesul OU")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Procesul OU (Reziduuri)
ax2.plot(df.index, residuals, color='purple', alpha=0.6, label='Reziduuri (Pret - Trend)')

# Desenam "Tunelul OU" (3 Sigma Echilibru)
boundary = 3 * std_equilibrium
ax2.axhline(boundary, color='red', linestyle='--', label='Limita Superioara OU (+3σ)')
ax2.axhline(-boundary, color='red', linestyle='--', label='Limita Inferioara OU (-3σ)')
ax2.axhline(mu, color='green', linestyle='-', alpha=0.5, label='Media (Attractor)')

# Flag Anomaliile
anomalies = residuals[np.abs(residuals) > boundary]
anom_indices = df.index[np.abs(residuals) > boundary]
ax2.scatter(anom_indices, anomalies, color='red', s=30, zorder=5, label='ANOMALIE (Rupere de elastic)')

ax2.set_title(f"2. Detectie Anomalii OU (Theta={theta:.3f}) - Punctele ies din 'Elastic'")
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()