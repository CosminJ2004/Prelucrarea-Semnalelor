import numpy as np
import matplotlib.pyplot as plt



A=5
f1 = 5
f2 = 10
f3 = 100
f4 = 1000         
phi1 = np.pi/2
phi2 =  np.pi
phi3= np.pi*2
phi4= np.pi*3
t = np.linspace(0, 1, 100000)  


x1 = A * np.sin(2 * np.pi * f1 * t + phi1)

x2 = A * np.sin(2 * np.pi * f1 * t + phi2)

x3 = A * np.sin(2 * np.pi * f1 * t + phi3)

x4 = A * np.sin(2 * np.pi * f1 * t + phi4)



figs,axs=plt.subplots(4)

axs[0].plot(t,x1)

axs[1].plot(t,x2)

axs[2].plot(t,x3)

axs[3].plot(t,x4)

plt.show()



z = np.random.normal(0, 1, len(x1))  


snr_values = [0.1, 1, 10, 100]

fig, axes = plt.subplots(4, 1, figsize=(10, 10))

for i, snr in enumerate(snr_values):
    gamma = np.linalg.norm(x1) / (np.linalg.norm(z) * np.sqrt(snr))
    x_noisy = x1 + gamma * z

    axes[i].plot(t, x_noisy)
    axes[i].set_title(f'SNR = {snr}')
    axes[i].set_ylabel('Amplitudine')
    axes[i].grid(True)

axes[-1].set_xlabel('Timp [s]')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()




