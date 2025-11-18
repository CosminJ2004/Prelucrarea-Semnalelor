import numpy as np
import time
N = 5 
K = N + 1 

np.random.seed(42) 
p_coeffs = np.random.randint(1, 10, size=K)
q_coeffs = np.random.randint(1, 10, size=K)

print("Coeficientii p(x) (de la x^0 la x^N):", p_coeffs)
print("Coeficientii q(x) (de la x^0 la x^N):", q_coeffs)

L_produs = 2 * N + 1 

r_coeffs_conv = np.convolve(p_coeffs, q_coeffs)

print("Coeficientii produsului r(x):", r_coeffs_conv)
print(f"Lungime r(x): {len(r_coeffs_conv)}")




N_fft = 16 

p_padded = np.pad(p_coeffs, (0, N_fft - K), 'constant')
q_padded = np.pad(q_coeffs, (0, N_fft - K), 'constant')

P_fft = np.fft.fft(p_padded)
Q_fft = np.fft.fft(q_padded)

R_fft = P_fft * Q_fft


r_coeffs_fft_complex = np.fft.ifft(R_fft)

r_coeffs_fft = np.round(np.real(r_coeffs_fft_complex[:L_produs])).astype(int) 

print("Coeficientii produsului r(x):", r_coeffs_fft)
print(f"Lungime r(x): {len(r_coeffs_fft)}")



