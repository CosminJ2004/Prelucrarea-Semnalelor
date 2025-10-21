import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt


fs = 44100     
duration = 2.0 
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

f1 = 400
f2 = 800


x1 = np.sin(2 * np.pi * f1 * t)
x2 = np.sin(2 * np.pi * f2 * t)


x_concat = np.concatenate((x1, x2))

sd.play(x_concat, fs)
sd.wait()

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(t, x1)


plt.subplot(3, 1, 2)
plt.plot(t, x2)

t_concat = np.linspace(0, 2*duration, int(fs * 2 * duration), endpoint=False)
plt.subplot(3, 1, 3)
plt.plot(t_concat, x_concat)


plt.tight_layout()
plt.show()
