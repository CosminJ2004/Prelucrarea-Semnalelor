import numpy as np
import matplotlib.pyplot as plt

fs = 10          
duration = 1
t_cont = np.linspace(0, duration, 1000, endpoint=False)   


f_a = fs / 2  
f_b = fs / 4  
f_c = 0       

x_a_cont = np.sin(2 * np.pi * f_a * t_cont)
x_b_cont = np.sin(2 * np.pi * f_b * t_cont)
x_c_cont = np.sin(2 * np.pi * f_c * t_cont)


fig, axes = plt.subplots(3, 1, figsize=(10, 8))

axes[0].plot(t_cont, x_a_cont, label='continuous (for visualization)', alpha=0.6)


axes[1].plot(t_cont, x_b_cont, label='continuous (for visualization)', alpha=0.6)

axes[2].plot(t_cont, x_c_cont, label='continuous (for visualization)', alpha=0.6)

#frecventa scade, sunt mai putine sinusoidale
#0 Hz -> o linie plata, nu avem semnal



plt.tight_layout()
plt.show()

