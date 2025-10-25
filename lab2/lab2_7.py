import numpy as np
import matplotlib.pyplot as plt


f=10
t=np.linspace(0,1,1000)

x=np.sin(2 * np.pi * f * t)
x = np.sin(2 * np.pi * f * t)


x_dec1 = x[::4]       
t_dec1 = t[::4]      


x_dec2 = x[1::4]
t_dec2 = t[1::4]

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, x)

plt.subplot(3, 1, 2)
plt.stem(t_dec1, x_dec1)

plt.subplot(3, 1, 3)
plt.stem(t_dec2, x_dec2)

plt.tight_layout()
plt.show()