import numpy as np
import matplotlib.pyplot as plt


f1 = 10  
f2 = 24 
fs = 38 
t_max = 1  


t_cont = np.linspace(0, t_max, 1000)
x_cont_1 = np.sin(2*np.pi*f1*t_cont)
x_cont_2 = 0.5*np.sin(2*np.pi*f2*t_cont)
x_cont_3 = np.sin(2*np.pi*fs*t_cont)


n = np.linspace(0, t_max, 16)
#cu una in plus fata de frecventa de esantionare corespunzatoare
x_samp_1 = np.sin(2*np.pi*f1*n)
x_samp_2 = 0.5*np.sin(2*np.pi*f2*n)
x_samp_3 = np.sin(2*np.pi*fs*n)


plt.figure(figsize=(10,8))


plt.subplot(3,1,1)
plt.plot(t_cont, x_cont_1)

plt.stem(n, x_samp_1)



plt.subplot(3,1,2)
plt.plot(t_cont,x_cont_2)

plt.stem(n,x_samp_2)


plt.subplot(3,1,3)
plt.plot(t_cont,x_cont_3)

plt.stem(n,x_samp_3)


plt.show()

