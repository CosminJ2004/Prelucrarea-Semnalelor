import numpy as np
import matplotlib.pyplot as plt

alpha = np.linspace(-np.pi/2, np.pi/2, 1000)


sin_alpha = np.sin(alpha)

approx_linear = alpha

# aproximarea Pade
num = alpha - (7 * alpha**3) / 60.0
den = 1 + (alpha**2) / 20.0
approx_pade = num / den

# erorile (absolute)
err_linear = np.abs(sin_alpha - approx_linear)
err_pade   = np.abs(sin_alpha - approx_pade)

plt.figure(figsize=(10, 6))
plt.plot(alpha, sin_alpha, label='sin(x) (exact)', linewidth=2)
plt.plot(alpha, approx_linear, '--', label='aprox. liniara: α', linewidth=1.5)
plt.plot(alpha, approx_pade, '-.', label='aprox. Pade', linewidth=1.5)
plt.title('sin(x) vs aproximatii pe intervalul [-π/2, π/2]')
plt.xlabel('x [rad]')
plt.ylabel('valoare')
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.figure(figsize=(10, 5))
plt.plot(alpha, err_linear, label='|sin(x) - x|', linewidth=1.5)
plt.plot(alpha, err_pade,   label='|sin(x) - Pade(x)|', linewidth=1.5)
plt.yscale('log')
plt.title('Eroare absoluta (axa y logaritmica)')

plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

