import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# 1. Date S&P 500 (2 ani pentru claritate vizuala)
print("Descarcare date...")
df = yf.download("^GSPC", start="2022-01-01", end="2024-01-01", progress=False)
data = df['Close'].values.flatten()
N = len(data)

# 2. Aplicam FFT (Transformata Fourier)
# Trecem din Timp -> Frecventa
fft_coeffs = np.fft.fft(data)
freqs = np.fft.fftfreq(N)

# 3. FILTRARE (Denoising)
# Pastram doar primele 'k' componente (frecventele joase/lente)
# Restul le facem zero (High Frequency Noise Removal)
keep_fraction = 0.05  # Pastram doar top 5% cele mai lente frecvente (Trendul)
cutoff_index = int(N * keep_fraction)

fft_coeffs_filtered = fft_coeffs.copy()
# Resetam la zero tot ce e peste indexul de cutoff (si simetric la final)
fft_coeffs_filtered[cutoff_index:-cutoff_index] = 0

# 4. Reconstructie (iFFT - Inverse FFT)
# Trecem inapoi din Frecventa -> Timp
filtered_data = np.fft.ifft(fft_coeffs_filtered).real

# --- EXTRA: Extragem Top 3 Unde Individuale ---
# Vrem sa vedem "din ce e facut" trendul
indices = np.argsort(np.abs(fft_coeffs))[: : -1] # Sortam dupa putere
# Ignoram indexul 0 (Componenta DC/Media) pentru vizualizare
top_indices = [i for i in indices if i != 0][:3] 

waves = []
for idx in top_indices:
    # Cream un spectru gol si punem doar acea frecventa
    single_freq_coeffs = np.zeros_like(fft_coeffs)
    single_freq_coeffs[idx] = fft_coeffs[idx]
    single_freq_coeffs[-idx] = fft_coeffs[-idx] # Simetricul (pentru numere reale)
    
    # Reconstruim unda
    wave = np.fft.ifft(single_freq_coeffs).real
    # O centram pe zero pentru vizualizare (fara media pretului)
    waves.append((freqs[idx], wave))


# 5. VIZUALIZARE COMPLEXA
fig = plt.figure(figsize=(14, 12))
plt.subplots_adjust(hspace=0.4)

# --- PANEL 1: Original vs Denoised ---
ax1 = plt.subplot(3, 1, 1)
ax1.plot(df.index, data, color='lightgray', label='Pret Brut (Cu Zgomot)')
ax1.plot(df.index, filtered_data, color='red', linewidth=2.5, label=f'FFT Denoised (Top {keep_fraction*100}% Frecvente)')
ax1.set_title("1. Rezultatul Final: Trendul extras prin filtrare Low-Pass")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- PANEL 2: Spectrul de Frecvente (Ce am pastrat vs Ce am taiat) ---
ax2 = plt.subplot(3, 1, 2)
power = np.abs(fft_coeffs[:N//2]) # Luam doar jumatate (simetrie)
freqs_half = freqs[:N//2]
# Plotam tot spectrul cu gri
ax2.plot(freqs_half, power, color='lightgray', label='Zgomot (Eliminat)')
# Plotam partea pastrata cu rosu
ax2.plot(freqs_half[:cutoff_index], power[:cutoff_index], color='red', label='Trend (Pastrat)')
ax2.set_yscale('log')
ax2.set_title("2. Spectrul de Putere: Am păstrat doar frecvențele joase (Stânga)")
ax2.set_ylabel("Putere (Log)")
ax2.set_xlabel("Frecvență")
ax2.legend()

# --- PANEL 3: Descompunerea (Top 3 Unde) ---
ax3 = plt.subplot(3, 1, 3)
colors = ['blue', 'green', 'purple']
for i, (freq, wave) in enumerate(waves):
    # Calculam perioada in zile (1 / frecventa)
    period = 1 / abs(freq)
    ax3.plot(df.index, wave, color=colors[i], label=f'Unda {i+1} (Ciclu ~{period:.0f} zile)', linewidth=1.5)

ax3.set_title("3. 'Ingredientele' Principale: Cele mai puternice 3 unde care compun prețul")
ax3.set_ylabel("Amplitudine (Oscilatie in jurul trendului)")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()