import numpy as np
import matplotlib.pyplot as plt


def x(t):
    return np.cos(520*np.pi*t + np.pi/3)

def y(t):
    return np.cos(280*np.pi*t - np.pi/3)

def z(t):
    return np.cos(120*np.pi*t + np.pi/3)


values = np.linspace(0, 0.03, 60)


x_vals = x(values)
y_vals = y(values)
z_vals = z(values)


plt.figure(figsize=(10, 8))


plt.subplot(3, 1, 1)
plt.plot(values, x_vals)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid(True)


plt.subplot(3, 1, 2)
plt.plot(values, y_vals)
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid(True)


plt.subplot(3, 1, 3)
plt.plot(values, z_vals)
plt.xlabel("t")
plt.ylabel("z(t)")
plt.grid(True)

plt.tight_layout()
plt.show()

                  
frecventa =  6
t_final = 0.03         
t_samples = np.linspace(0, t_final, frecventa)  


x_n = x(t_samples)
y_n = y(t_samples)
z_n = z(t_samples)

plt.figure(figsize=(10, 8))

# x[n]
plt.subplot(3, 1, 1)
plt.plot(t_samples, x_n)
plt.xlabel("t [s]")
plt.ylabel("x[n]")

# y[n]
plt.subplot(3, 1, 2)
plt.plot(t_samples, y_n)
plt.xlabel("t [s]")
plt.ylabel("y[n]")


# z[n]
plt.subplot(3, 1, 3)
plt.plot(t_samples, z_n)
plt.xlabel("t [s]")
plt.ylabel("z[n]")



plt.subplot(3, 1, 1)
plt.stem(t_samples, x_n)
plt.xlabel("t [s]")
plt.ylabel("x[n]")

# y[n]
plt.subplot(3, 1, 2)
plt.stem(t_samples, y_n)
plt.xlabel("t [s]")
plt.ylabel("y[n]")


# z[n]
plt.subplot(3, 1, 3)
plt.stem(t_samples, z_n)
plt.xlabel("t [s]")
plt.ylabel("z[n]")




plt.tight_layout()
plt.show()