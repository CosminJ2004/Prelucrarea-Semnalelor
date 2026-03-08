import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# =====================================================================
# 1. SETUP ȘI ANTRENAMENT (Modelul tău C-LSTM)
# =====================================================================
seq_length = 20
timp = np.linspace(0, 4 * np.pi, seq_length)
base_pattern = np.sin(timp) # Unda noastră țintă

# Generăm datele de antrenament
X_train = np.array([base_pattern + np.random.normal(0, 0.05, seq_length) for _ in range(1000)])
X_train_3D = X_train.reshape(-1, seq_length, 1)

model = models.Sequential([
    layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same', input_shape=(seq_length, 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.LSTM(12, activation='tanh'),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(seq_length, activation='linear')
])
model.compile(optimizer='adam', loss='mse')

print("Antrenăm C-LSTM (va dura câteva secunde)...")
model.fit(X_train_3D, X_train, epochs=30, batch_size=32, verbose=0)
print("Antrenament finalizat!\n")

# =====================================================================
# 2. GENERAREA FLUXULUI DE REȚEA (Cu Anomalii și Lag)
# =====================================================================
flux_lungime = 150
# Trafic de fond aleatoriu (zgomot normal de rețea care NU e pattern-ul nostru)
flux_date = np.random.normal(0, 0.4, flux_lungime) 

# INSERȚIA 1: Pattern curat la index 30
flux_date[30:50] = base_pattern + np.random.normal(0, 0.05, seq_length)

# INSERȚIA 2: Pattern cu LAG și PING masiv la index 90
flux_date[90:100] = base_pattern[0:10]        # Prima jumătate a pattern-ului
flux_date[100:102] = [3.0, -2.5]              # 2 pachete parazite uriașe (PING / Zgomot)
flux_date[102:112] = base_pattern[10:20]      # A doua jumătate (acum e decalată cu 2 pași!)

# =====================================================================
# 3. SCANNER-UL CU FEREASTRĂ GLISANTĂ
# =====================================================================
print("Începem scanarea fluxului cu C-LSTM...\n")
erori_pe_parcurs = []

# Scanăm pas cu pas
for i in range(flux_lungime - seq_length + 1):
    fereastra_curenta = flux_date[i : i + seq_length]
    
    # Formatăm 3D pentru C-LSTM: (1 exemplu, 20 pași, 1 feature)
    fereastra_3d = fereastra_curenta.reshape(1, seq_length, 1)
    
    # Predicție și calcul eroare (folosim flatten() pentru a le aduce pe ambele la 1D)
    reconstructie = model.predict(fereastra_3d, verbose=0)
    eroare = np.mean(np.square(fereastra_curenta - reconstructie.flatten()))
    erori_pe_parcurs.append(eroare)

# =====================================================================
# 4. VIZUALIZARE (Radarul de Detecție)
# =====================================================================
plt.figure(figsize=(15, 8))

# Graficul de sus: Traficul RAW
plt.subplot(2, 1, 1)
plt.plot(flux_date, label="Trafic Rețea Brut", color='gray')
plt.axvspan(30, 50, color='green', alpha=0.2, label='Pattern Curat')
plt.axvspan(90, 112, color='orange', alpha=0.2, label='Pattern cu Lag & Ping')
plt.title("Fluxul de date (Ce vede modelul pe sârmă)")
plt.legend()

# Graficul de jos: Eroarea MSE
plt.subplot(2, 1, 2)
plt.plot(erori_pe_parcurs, label="Eroare de Reconstrucție (C-LSTM)", color='red')
plt.axhline(y=0.1, color='blue', linestyle='--', label='Prag (Threshold) aproximativ')
plt.title("Radar Autoencoder (Eroarea scade dramatic când recunoaște forma)")
plt.legend()

plt.tight_layout()
plt.show()