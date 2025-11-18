import numpy as np
import matplotlib.pyplot as plt

B = 1.0
t_range = 3.0
t_points = 1000
Fs_values = [1.0, 1.5, 2.0, 4.0] 
t_continuous = np.linspace(-t_range, t_range, t_points, endpoint=False)
x_t = np.sinc(B * t_continuous)**2


fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, Fs in enumerate(Fs_values):
    Ts = 1.0 / Fs 
    ax = axes[i]
    N_pos = int(np.floor(t_range / Ts)) 
    
    indices = np.arange(-N_pos, N_pos + 1)

    t_samples = indices * Ts

    x_samples = np.sinc(B * t_samples)**2

    x_reconstructed = np.zeros_like(t_continuous)
    
    for n_idx, val in zip(t_samples, x_samples):
        sinc_term = np.sinc((t_continuous - n_idx) / Ts)
        x_reconstructed += val * sinc_term
        

    ax.plot(t_continuous, x_t, 'g-', label='Original')
    

    ax.stem(t_samples, x_samples, linefmt='C1-', markerfmt='C1o', basefmt=" ", label='Esantioane $x[n]$')
    
    ax.plot(t_continuous, x_reconstructed, 'k--', label='Reconstruit')

    ax.set_title(f'$F_s = {Fs:.2f}$ Hz')
    ax.set_xlabel('$t$ [s]')
    ax.set_ylim(-0.2, 1.2)
    ax.grid(True, alpha=0.5)


handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3)
plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.show()