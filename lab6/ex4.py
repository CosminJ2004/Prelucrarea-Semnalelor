import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 0.1,20)    

x = np.sin(2 * np.pi * 400 * t)



d=5
y=np.roll(x,d)


X = np.fft.fft(x)
Y = np.fft.fft(y)

corr1 = np.fft.ifft(X * Y)

corr2 = np.fft.ifft(Y / (X + 1e-12))


print("Rezultat IFFT(FFT(x)*FFT(y)):\n", np.round(corr1.real, 4))
print("\nRezultat IFFT(FFT(y)/FFT(x)):\n", np.round(corr2.real, 4))

#fiind sinus corelatia nu functioneaza asa de bine, deconvolutia da