import numpy as np
import matplotlib.pyplot as plt

# Parametri
n_steps = 1000
dt = 0.1
mu = 0      # Media la care vrea sa revina OU
theta = 0.1 # Forta elasticului (Mean Reversion Speed)
sigma = 1   # Zgomotul

# Initializare
brownian = np.zeros(n_steps)
ou_process = np.zeros(n_steps)

# Simulare
for i in range(1, n_steps):
    # Socul aleator (Zgomot Alb)
    shock = np.random.normal(0, np.sqrt(dt)) * sigma
    
    # 1. Brownian Motion: Doar adaugam socul
    brownian[i] = brownian[i-1] + shock
    
    # 2. Ornstein-Uhlenbeck: Adaugam socul + Elasticul
    # Elasticul trage valoarea anterioara inapoi spre 'mu'
    elastic_pull = theta * (mu - ou_process[i-1]) * dt
    ou_process[i] = ou_process[i-1] + elastic_pull + shock

# Vizualizare
plt.figure(figsize=(12, 6))

plt.plot(brownian, label='Brownian Motion (Random Walk - Preț)', alpha=0.7)
plt.plot(ou_process, label='Ornstein-Uhlenbeck (Mean Reverting - Volatilitate)', color='red')
plt.axhline(mu, color='black', linestyle='--', label='Media (Attractor)')

plt.title("Diferența: Brownian (Pleacă) vs. OU (Revine)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()