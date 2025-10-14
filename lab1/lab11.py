import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 0.1,1600)    

x = np.sin(2 * np.pi * 400 * t)

t1=np.linspace(0,3,2400)

x1 = np.sin(2 * np.pi * 800 * t1)

t2=np.linspace(0,0.01,1000)

x2 = 2 * (240* t2 - np.floor(0.5 + 240 * t2))

t3 = np.linspace(0,0.01,1000) 

x3 = np.sign(np.sin(2 * np.pi * 300 * t3))

I = np.random.rand(128, 128)
I2 = np.indices((128,128)).sum(axis=0) % 2

fig2, axs2 = plt.subplots(1, 2, figsize=(8, 4))

axs2[0].imshow(I, cmap='viridis')
axs2[0].set_title('Semnal 2D aleator')

axs2[1].imshow(I2, cmap='gray')
axs2[1].set_title('Semnal 2D propriu ')

plt.tight_layout()
plt.show()

fig,axs= plt.subplots(4)
axs[0].plot(t,x)
axs[1].plot(t1,x1)
axs[2].plot(t2,x2)
axs[3].plot(t3,x3)
plt.show()





