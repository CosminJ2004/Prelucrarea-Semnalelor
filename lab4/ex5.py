import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# citire audio
x, fs = sf.read("D:/Prelucrarea-Semnalelor/lab4/audio.wav")

# daca e stereo -> luam un canal
if len(x.shape) > 1:
    x = x[:, 0]

N = len(x)
L = int(0.01 * N)      # 1% din semnal
hop = L // 2           # overlap 50%

spectrogram = []

for start in range(0, N - L, hop):
    frame = x[start:start+L] * np.hamming(L)
    F = np.abs(np.fft.fft(frame))
    spectrogram.append(F)

spectrogram = np.array(spectrogram).T   # coloane FFT

# afisare
plt.imshow(20*np.log10(spectrogram[:L//2]),
           origin='lower', aspect='auto')
plt.title("Spectrograma (cu FFT)")
plt.xlabel("Frame")
plt.ylabel("Frecventa")
plt.colorbar(label="dB")
plt.show()
