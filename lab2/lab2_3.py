import numpy as np
import sounddevice as sd
from scipy.io import wavfile

fs = 44100  
duration = 2.0  
f = 440  
t = np.linspace(0, duration, int(fs * duration), endpoint=False)



# t = np.linspace(0, 0.1,1600)    

# x = np.sin(2 * np.pi * 400 * t)

# t1=np.linspace(0,3,2400)

# x1 = np.sin(2 * np.pi * 800 * t1)

# t2=np.linspace(0,0.01,1000)

# x2 = 2 * (240* t2 - np.floor(0.5 + 240 * t2))

# t3 = np.linspace(0,0.01,1000) 

# x3 = np.sign(np.sin(2 * np.pi * 300 * t3))


print(len(t))

x = 3 * np.sin(2 * np.pi * f * t) 


sd.play(x, fs)
sd.wait()  


wavfile.write("sinus_float.wav", fs, x)

fs_read, x_read = wavfile.read("sinus_float.wav")

sd.play(x_read, fs_read)
sd.wait()