import numpy as np
import matplotlib.pyplot as plt

N = 8


F = np.zeros((N, N), dtype=complex) 

for k in range(N): 
    for n in range(N):  
     
        argument = -1j * 2 * np.pi * k * n / N
        F[k, n] = np.exp(argument)

print(f"Dimensiunea Matricei Fourier F: {F.shape}")

fig, axes = plt.subplots(N, 2, figsize=(12, 2 * N))

x_indices = np.arange(N)

for i in range(N):
  
    axes[i, 0].stem(x_indices, F[i, :].real, markerfmt='bo', linefmt='b-', basefmt='r-')
    axes[i, 0].set_title(f'Linia {i}: Reala', fontsize=10)
    axes[i, 0].set_ylim(-1.1, 1.1)
    axes[i, 0].set_xticks(x_indices)
    axes[i, 0].grid(True, linestyle='--')

  
    axes[i, 1].stem(x_indices, F[i, :].imag, markerfmt='ro', linefmt='r-', basefmt='b-')
    axes[i, 1].set_title(f'Linia {i}: Imaginara', fontsize=10)
    axes[i, 1].set_ylim(-1.1, 1.1)
    axes[i, 1].set_xticks(x_indices)
    axes[i, 1].grid(True, linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()


F_H = F.conj().T
I = np.identity(N)
N_I = N * I


FH_F = F_H @ F


is_unitary_multiple_allclose = np.allclose(FH_F, N_I)


norm_difference = np.linalg.norm(FH_F - N_I)


print(f"Verificare cu numpy.allclose:{is_unitary_multiple_allclose}")
print(f"Norma Diferentei: {norm_difference:.10f}")