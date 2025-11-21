import os

import matplotlib.pyplot as plt
import numpy as np


dir_path = f"{os.getcwd()}/plots"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def compute_image(shape: tuple, f, norm: bool = False) -> np.ndarray:
    eps = 1e-9
    matrix = np.zeros(shape)

    # Numarul de esantioane (N)
    N = shape[0] 

    for i in range(shape[0]):
        for j in range(shape[1]):
            matrix[i, j] = f(i, j)

    if norm:
        # Aplica normalizarea logaritmica
        return np.abs(10 * np.log10(np.abs(matrix) + eps))

    return matrix

shape = (256, 256)
cmap = "grey"

f1 = lambda x, y: np.sin(2 * np.pi * x + 3 * np.pi * y)
matrix1 = compute_image(shape, f1, True)

f2 = lambda x, y: np.sin(4 * np.pi * x) + np.cos(6 * np.pi * y)
matrix2 = compute_image(shape, f2, True)

# este depasita frecventa Nyquist cu x si y 
#Semnalele sunt puternic aliate, imaginea în domeniul timp (matrix1/matrix2) arata ca un zgomot aleator

plt.imshow(matrix1, cmap=cmap)
plt.savefig("plots/First Function.pdf", format="pdf")
plt.show()

plt.imshow(matrix2, cmap=cmap)
plt.savefig("plots/Second Function.pdf", format="pdf")
plt.show()

f3 = lambda x, y: 1 if (x == 0 and y == 5) or (x == 0 and y == shape[0] - 5) else 0
matrix3 = compute_image(shape, f3)
reverse_matrix3 = np.real(np.fft.ifft2(matrix3))

#Această operație transformă spectrele de frecvență înapoi în semnalul lor original în domeniul timp.

# Rezultat: Deoarece semnalul de intrare (matrix3) contine doua varfuri simetrice, rezultatul iFFT este o unda sinusoidala (cosinus)
#varfuri pe orizontala
plt.imshow(reverse_matrix3, cmap=cmap)
plt.savefig("plots/First Spectrum.pdf", format="pdf")
plt.show()

matrix4 = matrix3.T
reverse_matrix4 = np.real(np.fft.ifft2(matrix4))

#muta varfurile prin transpunere pe verticala

plt.imshow(reverse_matrix4, cmap=cmap)
plt.savefig("plots/Second Spectrum.pdf", format="pdf")
plt.show()

matrix5 = matrix3 + matrix4
reverse_matrix5 = np.real(np.fft.ifft2(matrix5))

# este o suma de doua unde sinusoidale ortogonale (una verticală si una orizontala), rezultand un model de interferenta in domeniul timp

plt.imshow(reverse_matrix5, cmap=cmap)
plt.savefig("plots/Third Spectrum.pdf", format="pdf")
plt.show()