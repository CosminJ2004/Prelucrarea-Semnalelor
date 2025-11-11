import numpy as np
import matplotlib.pyplot as plt

# === 1. Citire doar coloana 3 ===
# Nota: Asigurati-va ca calea de fisier este corecta!
x = np.genfromtxt(r"D:\Prelucrarea-Semnalelor\lab5\Train.csv", delimiter=",", skip_header=1, usecols=2)
N = len(x)

print(f"Numar de esantioane initial: {N}")

# === 2. Eliminare NaN (Aceasta schimba N, deci trebuie recalculat) ===
x = x[~np.isnan(x)]
N = len(x)
print(f"Numar de esantioane dupa eliminarea NaN: {N}")
# Calculeaza media semnalului
x_mean = np.mean(x)

# Elimina componenta continua
x_centered = x - x_mean 

print(f"Media semnalului original: {x_mean:.2f}")
print(f"Media semnalului centrat: {np.mean(x_centered):.2e}")

# Inlocuiti x cu x_centered in sectiunea "3. Calcul FFT" din codul de la subpunctul (d)
# X = np.fft.fft(x_centered)

x=x_centered
# === 3. Calcul FFT ===
X = np.fft.fft(x)

# Normalizarea si Modulul:
X_modulus = np.abs(X) / N  # Modulul normalizat
X_half = X_modulus[:N // 2]  # Doar jumatate (spectrul unilateral)

# === 4. Definire Frecventa de Esantionare si Vectorul de Frecvente ===
# Fs = 1 / Ts. Presupunand ca esantionarea este la o ora (3600 secunde).
Fs = 1 / 3600  # Frecventa de esantionare [Hz]
N_half = N // 2

# Calculul vectorului de frecvente [Hz] de la 0 la Fs/2
# Folosim endpoint=False pentru a evita dubla numarare a frecventei Nyquist (Fs/2)
f = np.linspace(0, Fs / 2, N_half, endpoint=False)

# === 5. Afisare Grafica ===
plt.figure(figsize=(12, 6))
plt.plot(f, X_half)
plt.title("Modulul Transformatei Fourier (Spectrul de Amplitudine)")
plt.xlabel("Frecvență [Hz]")
plt.ylabel("|X(f)|")
plt.grid(True)
plt.show()