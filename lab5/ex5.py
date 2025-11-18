import numpy as np
import matplotlib.pyplot as plt


x = np.genfromtxt(r"D:\Prelucrarea-Semnalelor\lab5\Train.csv", delimiter=",", skip_header=1, usecols=2)
N = len(x)

print(f"Numar de esantioane initial: {N}")

x = x[~np.isnan(x)]
N = len(x)
print(f"Numar de esantioane dupa eliminarea NaN: {N}")

x_mean = np.mean(x)

x_centered = x - x_mean 

print(f"Media semnalului original: {x_mean:.2f}")
print(f"Media semnalului centrat: {np.mean(x_centered):.2e}")


# X = np.fft.fft(x_centered)

x=x_centered

X = np.fft.fft(x)

X_modulus = np.abs(X) / N  
X_half = X_modulus[:N // 2]  


Fs = 1 / 3600  
N_half = N // 2


f = np.linspace(0, Fs / 2, N_half, endpoint=False)

plt.figure(figsize=(12, 6))
plt.plot(f, X_half)
plt.title("Modulul Transformatei Fourier (Spectrul de Amplitudine)")
plt.xlabel("Frecventa [Hz]")
plt.ylabel("|X(f)|")
plt.grid(True)
plt.show()