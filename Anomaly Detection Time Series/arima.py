import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 1. DATE
print("Descarcare date S&P 500...")
df = yf.download("^GSPC", start="2000-01-01", end="2024-01-01", progress=False)

# Extragem Close si ne asiguram ca e Serie (nu DataFrame)
price = df['Close']
if hasattr(price, 'squeeze'): 
    price = price.squeeze()

# Convertim indexul la frecventa de business (zile lucratoare)
price.index = pd.to_datetime(price.index)

# --- FIX AICI ---
# Folosim .ffill() direct in loc de fillna(method='ffill')
price = price.asfreq('B').ffill()
# ----------------

# 2. ANTRENARE ARIMA
print("Antrenare model ARIMA(5,1,0)...")
# Ordinul (5,1,0) = AR(5), I(1), MA(0)
model = ARIMA(price, order=(5, 1, 0))
model_fit = model.fit()

# 3. EXTRAGERE REZIDUURI
print("Calculare reziduuri si anomalii...")
predictions = model_fit.predict(typ='levels')
residuals = price - predictions

# Aruncam primele 5 zile (start-up noise)
residuals = residuals.iloc[5:]

# 4. DEFINIRE PRAG ANOMALIE (3 Sigma)
sigma = np.std(residuals)
threshold = 3 * sigma
anomalies = residuals[np.abs(residuals) > threshold]

# 5. VIZUALIZARE
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Panel 1: Real vs Predictie
ax1.plot(price.index, price, label='Pret Real', color='black', alpha=0.5)
ax1.plot(predictions.index, predictions, label='Predictie ARIMA', color='blue', linestyle='--', alpha=0.7)
ax1.set_title("1. ARIMA(5,1,0): Predicția liniară a prețului")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Erorile (Anomaliile)
ax2.plot(residuals.index, residuals, color='purple', alpha=0.6, label='Eroare (Reziduu)')
ax2.axhline(threshold, color='red', linestyle='--', label=f'Limita (+{threshold:.0f})')
ax2.axhline(-threshold, color='red', linestyle='--', label=f'Limita (-{threshold:.0f})')

# Punctele Rosii
ax2.scatter(anomalies.index, anomalies, color='red', s=30, zorder=5, label='ANOMALIE (Eroare > 3σ)')

ax2.set_title(f"2. Detectie Anomalii ARIMA: Șocuri care rup corelația istorică")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nNumar anomalii detectate: {len(anomalies)}")
print(f"Pragul de detectie (3 Sigma): +/- {threshold:.2f}")