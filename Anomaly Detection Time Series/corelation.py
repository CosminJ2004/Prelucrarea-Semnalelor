import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# --- 1. FUNCTIA DTW (Algoritmul Elastic) ---
def simple_dtw(series_a, series_b):
    """
    Calculeaza distanta DTW intre doua serii de timp.
    Foloseste matricea de costuri si programare dinamica.
    """
    # Calculam matricea de distante euclidiene intre fiecare punct
    # Aici folosim 'cdist' pentru viteza (scipy)
    manhattan_dist = cdist(series_a.reshape(-1, 1), series_b.reshape(-1, 1), metric='cityblock')
    
    n, m = manhattan_dist.shape
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[1:, 0] = np.inf
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[0, 0] = 0
    
    # Umplem matricea de acumulare a costului minim
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = manhattan_dist[i-1, j-1]
            # Luam minimul dintre vecini (stanga, jos, diagonala)
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # Insertie
                                          dtw_matrix[i, j-1],    # Stergere
                                          dtw_matrix[i-1, j-1])  # Potrivire
            
    return dtw_matrix[n, m]

# --- 2. DATE SI SABLON ---
print("Descarcare date...")
df = yf.download("^GSPC", start="2005-01-01", end="2024-01-01", progress=False)
price = df['Close']
if hasattr(price, 'squeeze'): price = price.squeeze()

# Definim SABLONUL (CRASH 2008 - Partea cea mai violenta)
# Sept 2008 (Lehman) -> Dec 2008
start_template = "2008-09-01"
end_template = "2008-12-30"

template = price[start_template:end_template].values
# Normalizare Z-Score (ESENTIALA pentru DTW)
# Vrem sa comparam FORMA, nu pretul absolut (1000$ vs 4000$)
template_norm = (template - np.mean(template)) / np.std(template)

window_size = len(template)
print(f"Lungime Sablon 2008: {window_size} zile. Rulare DTW (poate dura 30 sec)...")

# --- 3. SLIDING WINDOW DTW ---
dtw_distances = []
dates_scanned = []

# Optimizare: Sarim cate 5 zile ca sa mearga mai repede (stride)
step = 5 
price_values = price.values

for i in range(0, len(price_values) - window_size, step):
    window = price_values[i : i + window_size]
    
    # Normalizam fereastra curenta
    if np.std(window) == 0: 
        dtw_distances.append(np.inf)
    else:
        window_norm = (window - np.mean(window)) / np.std(window)
        
        # Calculam distanta DTW
        # Distanta mica = Similaritate mare
        dist = simple_dtw(window_norm, template_norm)
        dtw_distances.append(dist)
        
    dates_scanned.append(price.index[i])

# Convertim la Pandas Series pentru plotare usoara
dtw_series = pd.Series(dtw_distances, index=dates_scanned)

# --- 4. VIZUALIZARE ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Panel 1: Pretul
ax1.plot(price.index, price, 'k', alpha=0.6, label='S&P 500')
ax1.set_title("1. Istoric S&P 500")
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Highlight zona Sablon
ax1.axvspan(pd.to_datetime(start_template), pd.to_datetime(end_template), color='red', alpha=0.3, label='Sablon 2008 (Target)')
ax1.legend()

# Panel 2: Distanta DTW (Inversul similaritatii)
# Cu cat e mai JOS graficul, cu atat seamana mai mult cu 2008
ax2.plot(dtw_series.index, dtw_series, color='blue', linewidth=1.5)
ax2.set_title("2. Distanța DTW (Graficul 'Fricii Structurale')\nValorile MICI (Văile) înseamnă că piața arată ca în 2008")
ax2.set_ylabel("Distanță DTW (Mai mic = Mai rău)")
ax2.grid(True, alpha=0.3)

# Punem un prag vizual (Top 5% cele mai mici distante)
threshold = dtw_series.quantile(0.05) 
ax2.axhline(threshold, color='red', linestyle='--', label='Prag de Alerta (Top 5% Similaritate)')

# Marcam momentele de alerta
alerts = dtw_series[dtw_series < threshold]
ax2.scatter(alerts.index, alerts, color='red', s=10, label='Alerta Structura 2008')

# Highlight pe graficul de pret unde sunt alerte
for date in alerts.index:
    # Desenam doar cateva linii ca sa nu incarcam
    if np.random.rand() > 0.8: # Desenam aleator doar 20% din linii pt claritate
        ax1.axvline(date, color='orange', alpha=0.2)

ax2.legend()
plt.tight_layout()
plt.show()

print(f"\nCele mai similare perioade cu 2008 (Distanta minima):")
print(dtw_series.sort_values().head(5))