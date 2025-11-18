import numpy as np
import matplotlib.pyplot as plt


def rectangular_window(N):
    return np.ones(N)

def hanning_window(N):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))


def padded_rectangle(Nw, N):
    return np.pad(rectangular_window(Nw), (0, N - Nw))

def padded_hanning(Nw, N):
    return np.pad(hanning_window(Nw), (0, N - Nw))

import numpy as np
import matplotlib.pyplot as plt

Nw = 200
fs = 1000 
T = 1          
t = np.linspace(0, T, fs) 

f = 100     
A = 1
phi = 0

x = A * np.sin(2 * np.pi * f * t + phi)

w_rect = padded_rectangle(Nw,fs)
w_hann = padded_hanning(Nw,fs)

x_rect = x * w_rect
x_hann = x * w_hann



plt.figure(figsize=(12, 6))

plt.subplot(3,1,1)
plt.plot(t, x)
plt.title("Sinusoidă originala (f = 100 Hz)")
plt.grid()

plt.subplot(3,1,2)
plt.plot(t, x_rect)
plt.title("Sinusoidă cu fereastră dreptunghiulara")
plt.grid()

plt.subplot(3,1,3)
plt.plot(t, x_hann)
plt.title("Sinusoida cu fereastră Hanning")
plt.grid()

plt.tight_layout()
plt.show()
