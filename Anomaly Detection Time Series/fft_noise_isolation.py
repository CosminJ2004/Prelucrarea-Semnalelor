import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# 1. Date S&P 500 (Perioada recenta pentru a vedea detaliile)
print("Descarcare date...")
df = yf.download("^GSPC", start="2022-01-01", end="2024-01-01", progress=False)
data = df['Close'].values.flatten()
N = len(data)

# 2. FFT (Transformata Fourier)
fft_coeffs = np.fft.fft(data)

# 3. FILTRARE INVERSA (High-Pass Filter)
# Vrem sa vedem DOAR zgomotul (ce am aruncat data trecuta)
keep_fraction = 0.05  # Pragul de 5% (aceeasi granita ca inainte)
cutoff_index = int(N * keep_fraction)

# Cream un array de zero-uri
fft_high_freq = np.zeros_like(fft_coeffs)

# COPIEM DOAR FRECVENTELE INALTE (Zgomotul)
# Sarim peste primele 'cutoff' (care sunt trendul) si copiem restul
fft_high_freq[cutoff_index:-cutoff_index] = fft_coeffs[cutoff_index:-cutoff_index]

# 4. Reconstructie (iFFT)
noise_data = np.fft.ifft(fft_high_freq).real

# Calculam si Trendul pentru comparatie (Low Pass)
fft_low_freq = fft_coeffs.copy()
fft_low_freq[cutoff_index:-cutoff_index] = 0
trend_data = np.fft.ifft(fft_low_freq).real

# 5. VIZUALIZARE FOCUSATA PE ZGOMOT
fig = plt.figure(figsize=(14, 10))
plt.subplots_adjust(hspace=0.4)

# --- PANEL 1: Contextul (Ce scoatem din ce) ---
ax1 = plt.subplot(3, 1, 1)
ax1.plot(df.index, data, color='black', alpha=0.3, label='Pret Brut')
ax1.plot(df.index, trend_data, color='blue', linewidth=2, label='Trend (Low Freq)')
ax1.set_title("1. Separarea: Preț = Trend (Albastru) + Zgomot (Restul)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- PANEL 2: ZGOMOTUL PUR (High Frequency) ---
ax2 = plt.subplot(3, 1, 2)
ax2.plot(df.index, noise_data, color='red', linewidth=1)
ax2.set_title(f"2. Zgomotul de Înaltă Frecvență (Extras prin FFT High-Pass)")
ax2.set_ylabel("Deviația față de Trend ($)")
# Adaugam o banda vizuala de "Normalitate"
std_noise = np.std(noise_data)
ax2.axhline(2*std_noise, color='green', linestyle='--', alpha=0.5, label='+2 Std Dev')
ax2.axhline(-2*std_noise, color='green', linestyle='--', alpha=0.5, label='-2 Std Dev')
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- PANEL 3: Distribuția Zgomotului (Histograma) ---
ax3 = plt.subplot(3, 1, 3)
ax3.hist(noise_data, bins=50, color='red', alpha=0.7, edgecolor='black')
ax3.set_title("3. Histograma Zgomotului: Este distribuția Normală (Gaussiană)")
ax3.set_xlabel("Amplitudine Zgomot ($)")
ax3.set_ylabel("Număr de Zile")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()