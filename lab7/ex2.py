import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets, ndimage
import os
import time

# --- FUNCTII IMPLEMENTATE ---

def calculeaza_snr_db(semnal_original, semnal_filtrat):

    zgomot = semnal_original - semnal_filtrat
    
    putere_semnal = np.sum(semnal_original**2)
    
    putere_zgomot = np.sum(zgomot**2)
    
    if putere_zgomot == 0:
        return np.inf

    return 10 * np.log10(putere_semnal / putere_zgomot)


def filtreaza_trece_jos_2d(imagine_cu_zgomot, raza_taiere):
    """
    Aplica un filtru trece-jos (Low-Pass) in domeniul frecventa.
    Metoda: FFT -> Masca -> IFFT.
    """
    
    H, W = imagine_cu_zgomot.shape
    centru_H, centru_W = H // 2, W // 2
    
 
    spectru_brut = np.fft.fft2(imagine_cu_zgomot)
    
    spectru_centrat = np.fft.fftshift(spectru_brut)
    
    masca = np.zeros((H, W), dtype=float)
    
    # Iteram pentru a crea un disc de raza 'raza_taiere'
    for i in range(H):
        for j in range(W):
            distanta = np.sqrt((i - centru_H)**2 + (j - centru_W)**2)
            if distanta <= raza_taiere:
                masca[i, j] = 1.0
                
    # 4. Aplicarea mastii
    spectru_filtrat = spectru_centrat * masca
    
    # 5. De-centrarea spectrului (Inversul lui fftshift)
    spectru_decentrat = np.fft.ifftshift(spectru_filtrat)
    
    # 6. Transformata Fourier Inversa (IFFT)
    imagine_filtrata = np.real(np.fft.ifft2(spectru_decentrat))
    
    return imagine_filtrata


# Calea pentru salvari
cale_grafice = "grafice_filtrare"
if not os.path.exists(cale_grafice):
    os.makedirs(cale_grafice)

np.random.seed(0)

harta_culori = plt.colormaps["gray"]

# Incarcare imagine raton (folosind noua denumire)
imagine_curata = datasets.face(gray=True) 
H, W = imagine_curata.shape

# 3. ADĂUGAREA ZGOMOTULUI
amplitudine_zgomot = 200
zgomot = np.random.randint(-amplitudine_zgomot, high=amplitudine_zgomot + 1, size=np.shape(imagine_curata))
imagine_zgomotoasa = imagine_curata + zgomot

# 4. VIZUALIZARE INITIALA
plt.imshow(imagine_curata, cmap=harta_culori)
plt.savefig(f"{cale_grafice}/Initial_Clean_Face.pdf", format="pdf")
plt.show()

# --- 5. ANALIZA SNR INAINTE DE FILTRARE ---
# Calculam SNR-ul dintre imaginea curata si imaginea zgomotoasa
# In mod normal, aceasta este o masura a zgomotului adaugat (semnalul este imaginea curata)
snr_initial = calculeaza_snr_db(imagine_curata, imagine_zgomotoasa) 

print("Tinta SNR dorita: 7 Db")
print(f"SNR CALCULAT INAINTE de filtrare: {snr_initial:.2f} Db")

# --- 6. PROCESUL DE FILTRARE SI OPTIMIZARE ---

# Raze de testat (copiate din logica initiala)
raze_test = [47, 48, 49, 50, 51, 52, 53]
valori_snr_obtinute = []

# Iteram pentru a gasi raza care maximizeaza SNR-ul (eroarea minima)
for raza in raze_test:
    # Filtrare
    imagine_filtrata = filtreaza_trece_jos_2d(imagine_zgomotoasa, raza)
    
    # Calcul SNR (evaluam calitatea imaginii filtrate fata de cea curata)
    valori_snr_obtinute.append(calculeaza_snr_db(imagine_curata, imagine_filtrata))


# Gasim indexul care corespunde celei mai mari valori SNR (cea mai buna performanta)
index_raza_optima = np.argmax(valori_snr_obtinute) 
raza_optima = raze_test[index_raza_optima]
snr_obtinut_optim = valori_snr_obtinute[index_raza_optima]

print(f"SNR CALCULAT DUPA filtrare (Optim): {snr_obtinut_optim:.2f} Db")
print(f"RAZA OPTIMA de taiere: {raza_optima}")


imagine_filtrata_finala = filtreaza_trece_jos_2d(imagine_zgomotoasa, raza_optima)

plt.imshow(imagine_filtrata_finala, cmap=harta_culori)
plt.savefig(f"{cale_grafice}/Filtered_Optimal_Face.pdf", format="pdf")
plt.show()