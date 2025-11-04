import numpy as np
import matplotlib.pyplot as plt
import time


def DFT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X


def FFT(x):
    N = len(x)
    if N <= 1:
        return x
    even = FFT(x[::2])
    odd = FFT(x[1::2])
    terms = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([even + terms[:N // 2] * odd,
                           even + terms[N // 2:] * odd])


Ns = [128, 256, 512, 1024, 2048, 4096, 8192]

times_dft = []
times_fft = []
times_npfft = []

for N in Ns:
    x = np.random.rand(N)

    # DFT
    start = time.time()
    DFT(x)
    times_dft.append(time.time() - start)

    # FFT
    start = time.time()
    FFT(x)
    times_fft.append(time.time() - start)

    # numpy.fft
    start = time.time()
    np.fft.fft(x)
    times_npfft.append(time.time() - start)


plt.figure(figsize=(10, 6))
plt.plot(Ns, times_dft, 'o-', label='DFT (manual)')
plt.plot(Ns, times_fft, 's-', label='FFT (manual)')
plt.plot(Ns, times_npfft, '^-', label='NumPy FFT')

plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Dimensiunea vectorului N (log₂)')
plt.ylabel('Timp de executie [s] (log)')
plt.title('Comparatie timpi de execuție: DFT vs FFT vs NumPy FFT')
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()
