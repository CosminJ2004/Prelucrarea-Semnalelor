import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Funcții de scalare (Esențiale pentru rețele neurale)
def normalize(data):
    return np.array(data) / 1000.0  # Mapăm valorile în intervalul [-1, 1] aprox.

def denormalize(data):
    return np.array(data) * 1000.0

# Pattern-ul de bază
base_pattern = np.array([80, 400, -100, 320, 600, -332, 40, 103, -22, 111])
sequence_length = len(base_pattern)

# Generăm datele de antrenament și LE NORMALIZĂM
# Este critic ca X_train să fie între -1 și 1 pentru ca ReLU/Adam să funcționeze optim
X_train_raw = np.array([base_pattern + np.random.normal(0, 2, sequence_length) for _ in range(2000)])
X_train = normalize(X_train_raw) 

# 2. Autoencoder-ul
model = models.Sequential([
    # Encoder
    layers.Dense(16, activation='relu', input_shape=(sequence_length,)),
    layers.Dense(4, activation='relu'), # Bottleneck ușor mai larg pentru stabilitate
    
    # Decoder
    layers.Dense(16, activation='relu'),
    layers.Dense(sequence_length, activation='linear') # 'linear' e ok aici pentru că datele sunt normalizate
])

model.compile(optimizer='adam', loss='mse')

print("Antrenăm modelul pe date NORMALIZATE...")
model.fit(X_train, X_train, epochs=100, batch_size=32, verbose=0)
print("Antrenament finalizat!\n")

# =====================================================================
def testeaza_secventa(nume, secventa_raw):
    # PASUL 1: Normalizăm secvența de test
    secventa_norm = normalize(secventa_raw)
    
    # PASUL 2: Predicție
    reconstructie_norm = model.predict(secventa_norm, verbose=0)
    
    # PASUL 3: Denormalizăm pentru a vedea numerele reale (opțional, pentru print)
    reconstructie_raw = denormalize(reconstructie_norm)
    
    # PASUL 4: Calculăm MSE pe valorile NORMALIZATE 
    # (E mult mai ușor să pui un prag gen 0.01 pe date scalate)
    eroare = np.mean(np.square(secventa_norm - reconstructie_norm))
    
    print(f"--- {nume} ---")
    print(f"Original (Raw):    {secventa_raw[0]}")
    print(f"Reconstruit (Raw): {np.round(reconstructie_raw[0], 1)}")
    print(f"Eroare (MSE Norm): {eroare:.6f}")
    
    # Pragul: Dacă eroarea e mai mare de 0.005, e clar o anomalie
    prag = 0.005 
    if eroare > prag:
        print(f"Verdict: ANOMALIE DETECTATĂ 🚨 (Eroare > {prag})\n")
    else:
        print(f"Verdict: PATTERN RECUNOSCUT ✅\n")

# Datele de test (raw)
secventa_buna = np.array([[85, 395, -98, 325, 590, -335, 42, 105, -21, 115]])
secventa_anormala = np.array([[800, 10, -500, 10, 10, -10, 500, 10, -10, 10]]) # Valori haotice
secventa_plata = np.array([[70, 440, -70, 305, 410, -230, 40, 120, -55, 175]])

# Rulăm testele
testeaza_secventa("Test 1 - Pattern SIMILAR", secventa_buna)
testeaza_secventa("Test 2 - Atac/Anomalie", secventa_anormala)
testeaza_secventa("Test 3 - Pattern PLAT", secventa_plata)