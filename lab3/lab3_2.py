import numpy as np
import matplotlib.pyplot as plt

fs=6
d=1
t=np.linspace(0,1,10000,endpoint=False)

x = np.sin(2 * np.pi * fs * t)

W_n = np.exp(-1j * 2 * np.pi * t)

z_complex = x * W_n

fig, axes = plt.subplots(2, 1, figsize=(12, 10))



axes[0].plot(t, x, color='dodgerblue', alpha=0.8, linewidth=1.5)

axes[0].set_xlabel('Timp (s)', fontsize=12)
axes[0].set_ylabel('Amplitudine', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.6)


axes[1].plot(z_complex.real, z_complex.imag, marker='o', linestyle='-', color='teal', markersize=3, alpha=0.6, label='Semnal Înfășurat: z[n]')

axes[1].set_aspect('equal', adjustable='box')
axes[1].set_xlabel('Partea Reala')
axes[1].set_ylabel('Partea Imaginara')
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.show()




winding_frequencies = [2, 5, 8, 13] 
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 16))

axes2_flat = axes2.flatten()

for idx, omega in enumerate(winding_frequencies):


    W_omega_n = np.exp(-1j * 2 * np.pi * omega * t)
    z_omega_complex = x * W_omega_n
    
 
    magnitude = np.abs(z_omega_complex) 

  
    scatter2 = axes2_flat[idx].scatter(z_omega_complex.real, z_omega_complex.imag,c=magnitude, cmap='coolwarm', s=20, alpha=0.8)


    axes2_flat[idx].set_aspect('equal', adjustable='box')
    
    if omega == fs:
        axes2_flat[idx].set_title(f'$\\omega$ = {omega} Hz (Egal cu Fs)', fontsize=14, color='green')
    else:
        axes2_flat[idx].set_title(f'$\\omega$ = {omega} Hz', fontsize=14)
        


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
