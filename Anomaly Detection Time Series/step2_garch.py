import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from arch import arch_model

# # 1. Date S&P 500
print("Descarcare date...")
df = yf.download("^GSPC", start="2010-01-01", end="2024-01-01", progress=False)

# --- FIX AICI ---
# Extragem Close. Daca e DataFrame (Tabel), il facem Serie (Lista) cu .squeeze()
price = df['Close']
if isinstance(price, pd.DataFrame):
    price = price.squeeze()
# ----------------

# 2. Calculam Randamentele (Returns)
returns = 100 * price.pct_change().dropna()
# 3. Antrenam Modelul GARCH(1,1)
# Acesta invata cat de "nervoasa" e piata bazat pe zilele anterioare
print("Antrenare GARCH...")
model = arch_model(returns, vol='Garch', p=1, q=1)
res = model.fit(disp='off') # disp='off' ascunde detaliile tehnice din consola

# 4. Extragem Volatilitatea Dinamica (Sigma Conditionata)
# Asta e "Deviatia Standard a Zilei"
conditional_volatility = res.conditional_volatility

# 5. Vizualizare
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# --- GRAFIC 1: Volatilitatea in sine (Nivelul de Frica) ---
ax1.plot(conditional_volatility, color='orange', linewidth=1.5)
ax1.set_title("1. Volatilitatea Condiționată (GARCH) - 'Termometrul de Frică'")
ax1.set_ylabel("Volatilitate (%)")
ax1.grid(True, alpha=0.3)
# Observa cum sare in 2020 si 2008, dar e mica in 2017

# --- GRAFIC 2: Randamente vs Limite Dinamice ---
# Aici e "Anomalie vs Context"
ax2.plot(returns.index, returns, color='gray', alpha=0.4, label='Randamente Zilnice')

# Construim "Tunelul de Volatilitate" (+/- 2.5 Sigma)
# Daca randamentul iese din acest tunel, e o anomalie REALA (Shock), nu doar volatilitate mare
upper_band = 2.5 * conditional_volatility
lower_band = -2.5 * conditional_volatility

ax2.plot(upper_band, color='red', linestyle='--', linewidth=1, label='Limita Dinamica (+/- 2.5σ)')
ax2.plot(lower_band, color='red', linestyle='--', linewidth=1)

# Flag-uim Anomaliile (Punctele care ies din tunel)
anomalies = returns[(returns > upper_band) | (returns < lower_band)]
ax2.scatter(anomalies.index, anomalies, color='red', s=15, zorder=5, label='Anomalie Contextuala')

ax2.set_title("2. Detecție Dinamică: Anomalii care sparg 'Tunelul de Volatilitate'")
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()