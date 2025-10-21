import numpy as np
import matplotlib.pyplot as plt

fs = 100000      
t = np.linspace(0, 0.02, int(fs * 0.02), endpoint=False)  

f1 = 200 
x1 = np.sin(2 * np.pi * f1 * t)

f2 = 300  
x2 = 2 * (f2 * t - np.floor(0.5 + f2 * t))  

x_sum = x1 + x2


plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, x1)

plt.subplot(3, 1, 2)
plt.plot(t, x2)


plt.subplot(3, 1, 3)
plt.plot(t, x_sum)


plt.tight_layout()
plt.show()
