import pywt # PyWavelets
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# 1. Date
print("Descarcare date...")
df = yf.download("^GSPC", start="2002-01-01", end="2024-01-01", progress=False)

price = df['Close']
if hasattr(price, 'squeeze'): 
    price = price.squeeze()

# --- FIXUL ESTE AICI (.copy()) ---
# Facem o copie explicita pentru a scapa de eroarea "buffer source array is read-only"
x = price.values.copy() 
# ---------------------------------

# 2. APLICARE WAVELET (DWT)
# Folosim wavelet-ul 'sym4' (Symlet) care e bun pentru date financiare (simetric)
wavelet_type = 'sym4' 
level = 4 # Cat de adanc mergem cu descompunerea

print(f"Aplicare Wavelet Transform ({wavelet_type})...")
# Descompunere: coeffs = [cA_n, cD_n, cD_n-1, ..., cD_1]
coeffs = pywt.wavedec(x, wavelet_type, level=level)

# 3. RECONSTRUCTIE TREND (Doar din Aproximare - cA)
# Setam toate detaliile (zgomotul) la zero
coeffs_trend = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
trend_wavelet = pywt.waverec(coeffs_trend, wavelet_type)

# Ajustam lungimea (Wavelet poate adauga 1-2 puncte la margini)
if len(trend_wavelet) > len(x): 
    trend_wavelet = trend_wavelet[:len(x)]

# 4. RECONSTRUCTIE ZGOMOT (Doar Detalii - cD1)
# Nivelul 1 de detalii contine cea mai inalta frecventa (High Frequency Noise)
# Asta e echivalentul "reziduurilor" pe care aplicam GARCH/OU
coeffs_noise = [np.zeros_like(coeffs[0])] + [np.zeros_like(c) for c in coeffs[1:-1]] + [coeffs[-1]]
noise_wavelet = pywt.waverec(coeffs_noise, wavelet_type)

if len(noise_wavelet) > len(x): 
    noise_wavelet = noise_wavelet[:len(x)]

# 5. VIZUALIZARE
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Panel 1: Trend
ax1.plot(df.index, x, color='gray', alpha=0.4, label='Pret Original')
ax1.plot(df.index, trend_wavelet, color='blue', linewidth=2, label=f'Wavelet Trend (Level {level})')
ax1.set_title(f"1. Descompunere Wavelet ({wavelet_type}): Trend")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Detalii (Zgomotul izolat)
ax2.plot(df.index, noise_wavelet, color='purple', alpha=0.7, label='Wavelet Details (High Freq)')
# Prag simplu pentru vizualizare
threshold = 2 * np.std(noise_wavelet)
ax2.axhline(threshold, color='red', linestyle='--', label='+2 Std Dev')
ax2.axhline(-threshold, color='red', linestyle='--', label='-2 Std Dev')
ax2.set_title("2. Detaliile de Înaltă Frecvență (Unde se ascund șocurile)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()