import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# 1. Obținere Date (Din 2000 ca să prindem marile crize)
print("Descarcare date S&P 500...")
df = yf.download("^GSPC", start="2000-01-01", end="2024-01-01", progress=False)

# Păstrăm doar prețul de închidere
price = df['Close']

# 2. Calculăm Diferențele (Price Returns)
# Diferenta = Pretul Azi - Pretul Ieri
# Asta transforma graficul dintr-o linie care urca, intr-o oscilatie in jurul lui 0
diff = price.diff().dropna()

# 3. Calculăm Z-Score Simplu
# Formula: (Valoare - Medie) / Deviatie Standard
mean_val = diff.mean()
std_val = diff.std()

z_score = (diff - mean_val) / std_val

# 4. Vizualizare
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Grafic 1: Prețul Brut (Context)
ax1.plot(price.index, price, color='black', alpha=0.8, label='Pret S&P 500')
ax1.set_title("1. Date Originale (Non-Staționare - Au Trend)")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Grafic 2: Z-Score pe Diferențe
ax2.plot(diff.index, z_score, color='blue', linewidth=0.8, label='Z-Score (Diferențe)')

# Adăugăm pragurile de anomalie (Standardul este +/- 3 Sigma)
ax2.axhline(3, color='red', linestyle='--', linewidth=1.5, label='Prag Anomalie (+3σ)')
ax2.axhline(-3, color='red', linestyle='--', linewidth=1.5, label='Prag Anomalie (-3σ)')

# Evidențiem punctele care depășesc pragul
anomalies = z_score[abs(z_score) > 3]
ax2.scatter(anomalies.index, anomalies, color='red', s=10, zorder=5)

ax2.set_title("2. Z-Score pe Diferențe (Staționar)")
ax2.set_ylabel("Deviații Standard (Sigma)")
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show() # Sau plt.savefig('step1_zscore.png')