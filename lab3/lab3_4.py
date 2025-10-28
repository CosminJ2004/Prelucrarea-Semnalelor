import numpy as np
import matplotlib.pyplot as plt

d = 1
N = 300
t = np.linspace(0, d, N, endpoint=False) 

F = [3, 10, 20] 
A = [1.0, 0.5, 2.0] 

x = (
    A[0] * np.sin(2 * np.pi * F[0] * t) +
    A[1] * np.sin(2 * np.pi * F[1] * t) +
    A[2] * np.sin(2 * np.pi * F[2] * t)
)

def calculate_dft_explicit(x_n, N):
    X_k = np.zeros(N, dtype=complex)
    n = np.arange(N) 
    for k in range(N):
        exponent = -1j * 2 * np.pi * k * n / N
        X_k[k] = np.sum(x_n * np.exp(exponent))
    return X_k

X = calculate_dft_explicit(x, N)
X_modul = np.abs(X)

Fs = N / d
delta_f = Fs / N

f_pozitive = np.arange(N // 2) * delta_f
modul_pozitiv = X_modul[:N // 2]
modul_normalizat = modul_pozitiv * 2.0 / N

fig, ax = plt.subplots(figsize=(10, 5))
ax.stem(f_pozitive, modul_normalizat, basefmt=" ")

for f_peak in F:
    ax.axvline(x=f_peak, color='r', linestyle=':', linewidth=1)
    
ax.set_xlabel('Frecventa (Hz)')
ax.set_ylabel('Amplitudine')
ax.set_xlim(0, 50)
ax.grid(True, linestyle='--', alpha=0.6)
plt.show()