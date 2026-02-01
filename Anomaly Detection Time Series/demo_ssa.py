import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# --- 1. FUNCTIA SSA ENGINE ---
def run_ssa_decomposition(series, L):
    """
    Executa SSA complet si returneaza elementele matematice interne.
    """
    N = len(series)
    K = N - L + 1
    
    # Pas 1: Matricea Hankel (Trajectory Matrix)
    # 
    X = np.column_stack([series[i : i + L] for i in range(K)])
    
    # Pas 2: SVD (Descompunerea)
    # 
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    
    return X, U, Sigma, VT

def reconstruct_component(U, Sigma, VT, component_idx):
    """
    Reconstruieste o componenta folosind Metoda Directa (Safe).
    Garanteaza ca nu inverseaza timpul.
    """
    # 1. Izolam componenta elementara (Matricea componenta)
    # X_elem = sigma * u * v.T
    X_elem = Sigma[component_idx] * np.outer(U[:, component_idx], VT[component_idx, :])
    
    # 2. Diagonal Averaging (Metoda Directa: i + j = t)
    # Aceasta metoda este imuna la erori de orientare (flip)
    rows, cols = X_elem.shape
    N_rec = rows + cols - 1
    
    reconstructed = np.zeros(N_rec)
    count = np.zeros(N_rec)
    
    # Parcurgem matricea si adunam elementele care apartin aceleiasi zile
    for r in range(rows):
        for c in range(cols):
            t = r + c
            reconstructed[t] += X_elem[r, c]
            count[t] += 1
            
    # Facem media
    return reconstructed / count

# --- 2. PREGATIREA DATELOR ---
print("Descarcare date S&P 500...")
df = yf.download("^GSPC", start="2023-01-01", end="2024-01-01", progress=False)

# Fix pentru yfinance (in caz ca returneaza DataFrame)
price = df['Close']
if hasattr(price, 'squeeze'):
    price = price.squeeze()
price = price.values

# L = 20 zile (Fereastra de analiza)
L = 20 

print(f"Rulare SSA cu Fereastra L={L}...")
X_hankel, U, Sigma, VT = run_ssa_decomposition(price, L)

# --- 3. RECONSTRUCTIE ---
# Reconstruim folosind metoda sigura
trend = reconstruct_component(U, Sigma, VT, 0)
comp1 = reconstruct_component(U, Sigma, VT, 1)
comp2 = reconstruct_component(U, Sigma, VT, 2)

# --- 4. VIZUALIZARE MATEMATICA ---
fig = plt.figure(figsize=(16, 10))
plt.subplots_adjust(hspace=0.3)

# A. SCREE PLOT (Energia)
# 
ax1 = plt.subplot(2, 2, 1)
ax1.plot(Sigma[:15], 'o-', color='purple', linewidth=2)
ax1.set_title("1. Scree Plot: Energia Componentelor\n(Componenta 0 domină totul)")
ax1.set_xlabel("Index Componenta")
ax1.set_ylabel("Valoare Singulară (Log Scale)")
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# B. FORMELE VECTORILOR (Eigenvectors)
ax2 = plt.subplot(2, 2, 2)
ax2.plot(U[:, 0], label='U0 (Trend Shape)', color='red', linewidth=2)
ax2.plot(U[:, 1], label='U1 (Oscilatie)', color='blue', alpha=0.6)
ax2.plot(U[:, 2], label='U2 (Oscilatie)', color='green', alpha=0.6)
ax2.set_title(f"2. Vectorii Proprii (Formele găsite în fereastra de {L} zile)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# C. RECONSTRUCTIA FINALA
ax3 = plt.subplot(2, 1, 2)
ax3.plot(price, color='black', alpha=0.3, label='Pret Original', linewidth=3)
ax3.plot(trend, color='red', linewidth=2, label='Componenta 0 (Trend)')
# Adunam si primele 2 oscilatii ca sa vedem cum se apropie de pret
ax3.plot(trend + comp1 + comp2, color='blue', linestyle='--', label='Trend + Comp 1 + Comp 2')

ax3.set_title("3. Reconstrucția: Cum adunăm piesele pentru a obține prețul")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- INFO TEXT ---
print("\n--- REZULTATE MATEMATICE ---")
print(f"Forma Matricei Hankel: {X_hankel.shape}")
print(f"Procent energie Trend: {(Sigma[0]**2 / np.sum(Sigma**2))*100:.2f}%")