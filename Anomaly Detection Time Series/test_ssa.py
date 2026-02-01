import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# 1. Date
df = yf.download("^GSPC", start="2023-01-01", end="2024-01-01", progress=False)
price = df['Close'].values.flatten()

# VERIFICARE 1: Datele brute sunt inversate?
print(f"Data start: {df.index[0]} | Pret: {price[0]}")
print(f"Data final: {df.index[-1]} | Pret: {price[-1]}")
if df.index[0] > df.index[-1]:
    print("ALERTĂ: Datele din yfinance au venit inversate cronologic!")
else:
    print("OK: Datele sunt în ordine corectă (Vechi -> Nou).")

# 2. SSA Simplu (Functia Robustă)
def ssa_robust(series, L=50):
    """
    SSA Reconstructie Directa (Fara fliplr - Elimina riscul de inversare)
    """
    N = len(series)
    
    # 1. Padding Simetric (pentru margini)
    series_padded = np.pad(series, (L, L), mode='reflect')
    N_pad = len(series_padded)
    K = N_pad - L + 1
    
    # 2. Matricea Hankel
    X = np.column_stack([series_padded[i : i + L] for i in range(K)])
    
    # 3. SVD
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    
    # 4. Extragere Trend (Componenta 0)
    # Reconstruim matricea doar cu prima componenta
    X_trend = Sigma[0] * np.outer(U[:, 0], VT[0, :])
    
    # 5. Diagonal Averaging (Metoda Directa i+j=t)
    # Nu mai rotim matricea. Calculam media direct pe indicii corecti.
    reconstructed = np.zeros(N_pad)
    count = np.zeros(N_pad)
    
    # X_trend are dimensiuni (L, K)
    rows, cols = X_trend.shape
    
    for r in range(rows):
        for c in range(cols):
            t = r + c # Timpul t corespunde sumei indicilor r si c
            reconstructed[t] += X_trend[r, c]
            count[t] += 1
            
    trend_padded = reconstructed / count
    
    # 6. Taiem Padding-ul
    trend = trend_padded[L : -L]
    
    # Siguranta dimensiune
    if len(trend) > N: trend = trend[:N]
    elif len(trend) < N: trend = np.pad(trend, (0, N-len(trend)), mode='edge')
        
    return trend

trend = ssa_robust(price)

# 3. Plotare Comparativă
plt.figure(figsize=(10, 6))
plt.plot(price, label='Pret (Original)', color='black', alpha=0.3)
plt.plot(trend, label='SSA Trend', color='red', linewidth=2)

# Verificam corelatia
corr = np.corrcoef(price, trend)[0, 1]
plt.title(f"Verificare Orientare: Corelație = {corr:.2f}\n(Daca e -1, e inversat sus-jos. Daca e mică, e inversat stanga-dreapta)")
plt.legend()
plt.show()