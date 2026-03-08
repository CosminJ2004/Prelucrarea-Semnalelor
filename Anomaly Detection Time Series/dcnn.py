import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- 1. SETĂRI DE BAZĂ ȘI ANTRENAMENT (Același ca al tău) ---
def normalize(data):
    return np.array(data) / 1000.0

base_pattern = np.array([80, 400, -100, 320, 600, -332, 40, 103, -22, 111])
seq_len = len(base_pattern)

X_train_raw = np.array([base_pattern + np.random.normal(0, 5, seq_len) for _ in range(1000)])
X_train = normalize(X_train_raw)

model = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=(seq_len,)),
    layers.Dense(4, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(seq_len, activation='linear')
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)

# --- 2. GENERAREA FLUXULUI DE DATE (LONG STREAM) ---
flux_lungime = 100
flux_date = np.random.normal(10, 50, flux_lungime) # Zgomot de fond (trafic random)

# Inserția 1: Pattern-ul nostru apare la indexul 20 (cu puțin zgomot normal)
flux_date[20:30] = base_pattern + np.random.normal(0, 5, seq_len)

# Inserția 2: Pattern-ul apare la indexul 70, dar primește un pachet PING (lag/zgomot parazit) în mijloc!
# Luăm primele 5 pachete din pattern
flux_date[65:70] = base_pattern[0:5] 
# Pachet PING parazit la index 70
flux_date[70] = 800 
# Restul de 5 pachete din pattern (decalate)
flux_date[71:76] = base_pattern[5:10] 

# --- 3. SCANNER-UL CU FEREASTRĂ GLISANTĂ ---
print("Începem scanarea fluxului de date...\n")

erori_pe_parcurs = []
prag = 0.015 # Setăm un prag pentru recunoaștere

for i in range(flux_lungime - seq_len + 1):
    # Extragem fereastra curentă (10 pachete)
    fereastra_curenta_raw = flux_date[i : i + seq_len]
    fereastra_norm = normalize([fereastra_curenta_raw])
    
    # Reconstruim și calculăm eroarea
    reconstructie = model.predict(fereastra_norm, verbose=0)
    eroare = np.mean(np.square(fereastra_norm - reconstructie))
    erori_pe_parcurs.append(eroare)
    
    # Dacă eroarea e sub prag, modelul recunoaște forma!
    if eroare < prag:
        print(f"[!] MATCH GĂSIT la indexul {i}! Eroare: {eroare:.5f}")

# --- 4. VIZUALIZARE (Opțional, dar super util) ---
plt.figure(figsize=(15, 6))

# Graficul 1: Fluxul de date brut
plt.subplot(2, 1, 1)
plt.plot(flux_date, label="Trafic Rețea", color='gray', alpha=0.7)
plt.axvspan(20, 30, color='green', alpha=0.3, label='Pattern Introdus (Curat)')
plt.axvspan(65, 76, color='orange', alpha=0.3, label='Pattern Introdus (Cu Lag/Ping)')
plt.title("Fluxul de date RAW")
plt.legend()

# Graficul 2: Radarul de eroare
plt.subplot(2, 1, 2)
plt.plot(erori_pe_parcurs, label="Eroare MSE (Mai mic = Pattern Recunoscut)", color='red')
plt.axhline(y=prag, color='blue', linestyle='--', label='Prag de detecție')
plt.title("Radar Autoencoder (Eroare de Reconstrucție)")
plt.legend()

plt.tight_layout()
plt.show()