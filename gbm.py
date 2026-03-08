import numpy as np
import matplotlib.pyplot as plt

# --- 1. SETĂRILE DE BAZĂ (Zilele obișnuite) ---
S0 = 100
mu = 0.10      # Trend normal de 10%
sigma = 0.20   # Volatilitate normală de 20%
T = 1.0
N = 252
M = 10         # Generăm 10 scenarii paralele
dt = T / N

# --- 2. PARAMETRII NOI (Setările pentru Anomalii/Salturi) ---
lambda_ = 3.0    # Frecvența: Ne așteptăm la 3 anomalii (salturi) pe an
mu_J = -0.15     # Direcția: În medie, anomalia înseamnă un Crash de -15%
sigma_J = 0.05   # Variația crash-ului: Poate fi -10% sau -20%

np.random.seed(42) # (Opțional) Pentru a avea același grafic la fiecare rulare

# --- 3. CREAREA "ZILELOR NORMALE" (Difuzia Geometric Brownian) ---
Z = np.random.normal(0, 1, size=(N, M))
difuzie = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

# --- 4. CREAREA "ANOMALIILOR" (Procesul Poisson) ---
# Funcția poisson ne va da de cele mai multe ori 0. Rar, ne va da 1 (o anomalie azi!)
salturi_poisson = np.random.poisson(lambda_ * dt, size=(N, M))

# Dacă există o anomalie (1), cât de mare e? (Extragem mărimea dintr-o distribuție)
marime_salt = np.random.normal(mu_J, sigma_J, size=(N, M))
total_salturi = salturi_poisson * marime_salt

# --- 5. ADUNĂM ZILELE NORMALE CU ANOMALIILE ---
randamente_zilnice = difuzie + total_salturi

# --- 6. CALCULĂM PREȚUL FINAL ---
drumuri_logaritmice = np.cumsum(randamente_zilnice, axis=0)
S = np.zeros((N + 1, M))
S[0] = S0
S[1:] = S0 * np.exp(drumuri_logaritmice) # Transformăm logaritmul înapoi în Dolari

# --- 7. VIZUALIZARE ---
plt.figure(figsize=(10, 6))
plt.plot(S, linewidth=1.5)
plt.title('Merton Jump-Diffusion Model (Simulare cu Anomalii/Salturi)')
plt.xlabel('Zile de tranzacționare')
plt.ylabel('Prețul acțiunii ($)')
plt.axhline(S0, color='black', linestyle='--', label='Preț inițial')
plt.grid(True, alpha=0.3)
plt.show()