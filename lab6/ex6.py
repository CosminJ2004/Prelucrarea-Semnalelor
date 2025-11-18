import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x = np.genfromtxt(r"D:\Prelucrarea-Semnalelor\lab5\Train.csv", delimiter=",", skip_header=1, usecols=2)

x=x[:72] #primele 3 zile

windows = [5, 9, 13, 17]
smoothed_signals = {}

for w in windows:
    smoothed = np.convolve(x, np.ones(w), 'valid') / w
    smoothed_signals[w] = smoothed
    

plt.figure(figsize=(12, 6))
plt.plot(x, label="Original", marker='o')

for w, y in smoothed_signals.items():
    plt.plot(range(w-1, w-1+len(y)), y, label=f"Moving average w={w}", marker='o')

plt.xlabel("Ora (index esantion)")
plt.ylabel("Numar vehicule")
plt.title("Semnal trafic si filtrul medie alunecatoare")
plt.legend()
plt.grid(True)
plt.show()

#c)
#fs=1/3600
#f_nyquist=fs/2=1/7200
#alegem f_cut= 0.1*fs = 1/36000
#fc_normalizat=f_cut/f_nyquist= 0.2
#d,e)

from scipy.signal import butter, cheby1, filtfilt, freqz


fs = 1        
f_cut = 0.1   
f_nyquist = fs/2
Wn = f_cut / f_nyquist  

N = 5          
rp = 5        

b_butt, a_butt = butter(N, Wn, btype='low', analog=False)
x_butt = filtfilt(b_butt, a_butt, x)

b_cheb, a_cheb = cheby1(N, rp, Wn, btype='low', analog=False)
x_cheb = filtfilt(b_cheb, a_cheb, x)



plt.figure(figsize=(12,6))
plt.plot(x, label="Original", marker='o')
plt.plot(x_butt, label="Butterworth 5th order", linewidth=2)
plt.plot(x_cheb, label=f"Chebyshev I 5th order, rp={rp}dB", linewidth=2)
plt.xlabel("Ora (index eșantion)")
plt.ylabel("Număr vehicule")
plt.title("Filtre trece-jos: Butterworth vs Chebyshev Type I")
plt.legend()
plt.grid(True)
plt.show()

#as alege butterworth, nu aduce oscilatii nedorite, are tranzitii mai abrupte


#f


orders = [3, 5, 7]  
rp_values = [1, 3, 5, 8]  

plt.figure(figsize=(12,6))
plt.plot(x, label="Original", color='black', linewidth=2)

for N in orders:
    b, a = butter(N, Wn, btype='low')
    x_butt = filtfilt(b, a, x)
    plt.plot(x_butt, label=f"Butterworth N={N}")

plt.xlabel("Ora (index eșantion)")
plt.ylabel("Număr vehicule")
plt.title("Butterworth: efectul ordinului filtrului")
plt.legend()
plt.grid(True)
plt.show()
#cu cat creste ordinul filtrului cu atat devine mai abrupt, mai sensibil si mai oscilant

plt.figure(figsize=(12,6))
plt.plot(x, label="Original", color='black', linewidth=2)

N = 5  
for rp in rp_values:
    b, a = cheby1(N, rp, Wn, btype='low')
    x_cheb = filtfilt(b, a, x)
    plt.plot(x_cheb, label=f"Chebyshev I N={N}, rp={rp} dB")

plt.xlabel("Ora (index esantion)")
plt.ylabel("Numar vehicule")
plt.title("Chebyshev I: efectul rp")
plt.legend()
plt.grid(True)
plt.show()

#un ripple mai mic inseamna mai multe oscilatii, invers cumva de butterworth, aici cu cat creste ripple cu atat valorile mari afecteaza din ce in ce mai putin, ca un fel de filtru median
