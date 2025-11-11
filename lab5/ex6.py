import numpy as np
import matplotlib.pyplot as plt

# --- 1. SETUP SI INCARCARE DATE ---
FS = 1 / 3600  # Fs = 1 / 3600 secunde
FILE_PATH = r"D:\Prelucrarea-Semnalelor\lab5\Train.csv"

try:
    x_raw = np.genfromtxt(FILE_PATH, delimiter=",", skip_header=1, usecols=2)
except FileNotFoundError:
    print(f"EROARE: Fisierul '{FILE_PATH}' nu a fost gasit.")
    exit()

# Prelucrare
x = x_raw[~np.isnan(x_raw)]
N = len(x)
x_mean = np.mean(x)
x_centered = x - x_mean 

# Functie FFT
def compute_fft_spectrum(signal):
    N_sig = len(signal)
    X = np.fft.fft(signal)
    X_modulus = np.abs(X) / N_sig
    N_half = N_sig // 2
    X_half = X_modulus[:N_half]
    f = np.linspace(0, FS / 2, N_half, endpoint=False)
    return X, X_half, f

print("\n--- (a) & (b) Fs si Interval Timp ---")
print(f"Frecventa de esantionare (Fs): {FS:.6e} Hz")
T_total_hours = N
print(f"Interval total: {T_total_hours} ore ({T_total_hours/24:.2f} zile)")

print("\n--- (c) Frecventa Maxima ---")
F_max = FS / 2
print(f"Frecventa maxima (Nyquist): {F_max:.6e} Hz")

# Calcul FFT pe semnalul centrat
X, X_half_centered, f = compute_fft_spectrum(x_centered)

print("\n--- (d) Calcul FFT si Plot Spectru ---")
plt.figure(figsize=(10, 5))
plt.plot(f, X_half_centered)
plt.title("Spectru FFT (Semnal Centrat)")
plt.xlabel("Frecventa [Hz]")
plt.ylabel("Amplitudine")
plt.grid(True)
plt.xlim(0, 0.00003)
plt.show()
print("-> Plot FFT realizat.")

print("\n--- (e) Componenta Continua (Plot) ---")
# Setari pentru plot comparativ
N_seg = 720  
i_start = 1008
i_stop = i_start + N_seg
time_axis = np.arange(N_seg) / 24

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(time_axis, x[i_start:i_stop], label=f"Original (Media: {np.mean(x[i_start:i_stop]):.2f})", color='blue')
plt.hlines(np.mean(x[i_start:i_stop]), time_axis[0], time_axis[-1], color='red', linestyle='--')
plt.title("Semnal Original (Cu DC Offset)")
plt.ylabel("Nr masini")

plt.subplot(2, 1, 2)
plt.plot(time_axis, x_centered[i_start:i_stop], label=f"Centrat (Media: {np.mean(x_centered[i_start:i_stop]):.2e})", color='green')
plt.hlines(0, time_axis[0], time_axis[-1], color='black', linestyle='--')
plt.title("Semnal Centrat (DC Eliminat)")
plt.xlabel("Timp [Zile]")
plt.ylabel("Amplitudine Centrata")
plt.tight_layout()
plt.show()
print("-> Plot comparativ DC realizat.")

print("\n--- (f) Frecvente Principale ---")
# Identificare 4 varfuri
top_indices_raw = np.argsort(X_half_centered)[-4:]
top_indices = top_indices_raw[np.argsort(f[top_indices_raw])]
top_frequencies = f[top_indices]

for freq in top_frequencies:
    period_hours = 1 / (freq * 3600)
    phenomenon = "Zile/Saptamani"
    print(f"Frecventa: {freq:.6e} Hz | Perioada: {period_hours:.2f} ore -> {phenomenon}")


print("\n--- (g) Vizualizare 1 Luna (Plot) ---")
i_start = 1008
i_stop = i_start + 720
x_month = x[i_start:i_stop]
time_axis = np.arange(len(x_month)) / 24

plt.figure(figsize=(10, 5))
plt.plot(time_axis, x_month, label="Trafic pe 30 de zile")
plt.title("Trafic pe o Luna")
plt.xlabel("Timp [Zile]")
plt.ylabel("Numar de masini")
plt.grid(True)
plt.show()
print("-> Plot 1 Luna realizat.")

print("\n--- (h) Metoda Determinare Ziua de Inceput ---")
print("Metoda: Corelatie Incrucisata cu un sablon saptamanal (5 zile sus, 2 zile jos). Varful corelatiei indica decalajul fata de inceputul saptamanii.")

print("\n--- (i) Filtrare Semnal (Plot) ---")
F_cut = 3.0 * 1e-5  # Filtru trece-jos (Low-Pass)

# 1. Filtrare in domeniu frecventa
X_filtered = np.copy(X)
N_full = len(X)
f_full = np.fft.fftfreq(N_full, d=1/FS) 
cutoff_indices = np.where(np.abs(f_full) > F_cut)
X_filtered[cutoff_indices] = 0 

# 2. Transformata Inversa
x_filtered = np.fft.ifft(X_filtered)
x_filtered_final = np.real(x_filtered) + x_mean

# Plot comparativ
plt.figure(figsize=(10, 5))
plt.plot(time_axis, x_month, label="Brut", alpha=0.5) 
plt.plot(time_axis, x_filtered_final[i_start:i_stop], label="Filtrat", color='red', linewidth=2) 
plt.title(f"Semnal Filtrat (Taiere la {F_cut:.6f} Hz)")
plt.xlabel("Timp [Zile]")
plt.ylabel("Numar de masini")
plt.legend()
plt.grid(True)
plt.show()
print("-> Plot Semnal Filtrat realizat.")