import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
import matplotlib.style as style

# --- CONFIGURARE VIZUALA ---
# Folosim un stil curat pentru prezentare
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'

# --- GENERARE DATE ---
x = np.linspace(-6, 6, 1000)

# 1. Lumea Teoretică (Gauss / Bachelier)
# Distributie Normala Standard
y_gauss = norm.pdf(x, 0, 1)

# 2. Lumea Reală (Fat Tails / Mandelbrot)
# Folosim distributia Student-t cu grade mici de libertate (df=2.5)
# Aceasta simuleaza perfect "cozile grase" din finante
y_fat = t.pdf(x, df=2.5, loc=0, scale=1) 

# --- PLOTARE ---
fig, ax = plt.subplots(figsize=(12, 7))

# Plotam liniile
ax.plot(x, y_gauss, 'g--', linewidth=2.5, alpha=0.7, label='Teorie: Distribuție Normală (Gauss)')
ax.plot(x, y_fat, 'r-', linewidth=3, label='Realitate: "Fat Tails" (Piața Financiară)')

# --- EVIDENTIEREA ZONELOR CRITICE (Black Swans) ---
# Umplem cozile extreme (peste 3 sigma)
limit_high = 2.8
limit_low = -2.8

# Coada Dreapta (Extreme Pozitive)
ax.fill_between(x, y_fat, 0, where=(x >= limit_high), 
                color='red', alpha=0.3, interpolate=True)
ax.fill_between(x, y_gauss, 0, where=(x >= limit_high), 
                color='green', alpha=0.1, interpolate=True)

# Coada Stanga (Extreme Negative - Crash)
ax.fill_between(x, y_fat, 0, where=(x <= limit_low), 
                color='red', alpha=0.3, interpolate=True)
ax.fill_between(x, y_gauss, 0, where=(x <= limit_low), 
                color='green', alpha=0.1, interpolate=True)

# --- ANOTARI SI EXPLICATII ---
# Adaugam sageti catre cozile grase
ax.annotate('Zona "Black Swan"\n(Risc subestimat de modelele clasice)', 
            xy=(-3.5, 0.02), xytext=(-5.5, 0.15),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=11, fontweight='bold', color='darkred')

ax.annotate('Probabilitate Reală vs Teoretică', 
            xy=(3.5, 0.02), xytext=(2.5, 0.15),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=11, fontweight='bold', color='darkred')

# --- COSMETIZARE FINALĂ ---
ax.set_title('Ilustrare Conceptuală: Teoria Gaussiană vs. Realitatea Pieței', fontsize=16, pad=20)
ax.set_xlabel('Deviația Standard (Sigma)', fontsize=12)
ax.set_ylabel('Probabilitate', fontsize=12)

# Scoatem numerele de pe axa Y pentru ca e grafic conceptual
ax.set_yticks([]) 
ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.45)

# Legenda
ax.legend(loc='upper right', frameon=True, fontsize=12, shadow=True)

# Salvare
plt.tight_layout()
plt.savefig('concept_fat_tails.png', dpi=300)
plt.show()

print("Imaginea 'concept_fat_tails.png' a fost generată cu succes.")