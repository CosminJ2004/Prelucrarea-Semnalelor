import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# 1. Descărcăm datele (Exemplu: S&P 500)
data = yf.download("^GSPC", start="2010-01-01", end="2024-01-01")
returns = data['Close'].pct_change().dropna().values.reshape(-1, 1)

returns = data['Close'].pct_change().dropna().values.reshape(-1, 1) * 100

# 2. Configurăm modelul HMM
# Presupunem 3 stări: Bull (Calm), Bear (Panică), Side (Tranziție)
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
model.fit(returns)

# 3. Predictibilitatea stărilor (Hidden States)
hidden_states = model.predict(returns)

# 4. Organizăm rezultatele pentru vizualizare
results = pd.DataFrame({'Returns': returns.flatten(), 'State': hidden_states}, 
                       index=data.index[1:])

# Vizualizare
plt.figure(figsize=(15, 8))
for i in range(model.n_components):
    state_data = results[results['State'] == i]
    plt.plot(state_data.index, data['Close'].loc[state_data.index], '.', label=f'Starea {i}')

plt.title("Detectarea Regimurilor de Piață cu HMM (S&P 500)")
plt.legend()
plt.show()