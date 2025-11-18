import numpy as np
import matplotlib.pyplot as plt

N = 100
t_start = 0
t_end = 1
x0 = np.random.rand(N) 

x1 = np.convolve(x0, x0, mode='full') # Lungime 199
x2 = np.convolve(x1, x0, mode='full') # Lungime 298
x3 = np.convolve(x2, x0, mode='full') # Lungime 397

signals = [x0, x1, x2, x3]
fig, axes = plt.subplots(4, 1, figsize=(12, 10))


for j, sig in enumerate(signals):
    L = len(sig)

    t_axis = np.arange(L)

    axes[j].plot(t_axis, sig)

    axes[j].grid(True, alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()


x_block = np.zeros(N)
x_block[40:60] = 1.0 

xb1 = np.convolve(x_block, x_block, mode='full') 
xb2 = np.convolve(xb1, x_block, mode='full') 
xb3 = np.convolve(xb2, x_block, mode='full') 

signals_block = [x_block, xb1, xb2, xb3]

fig_b, axes_b = plt.subplots(4, 1, figsize=(12, 10))

for j, sig in enumerate(signals_block):
    L = len(sig)
    t_axis = np.arange(L)
    
    axes_b[j].plot(t_axis, sig)
    axes_b[j].grid(True, alpha=0.5)
    
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()
