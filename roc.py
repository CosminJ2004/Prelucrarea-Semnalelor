import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# --- 1. DESCĂRCAREA DATELOR ---
# Folosim SPY (ETF-ul pentru S&P 500) ca reprezentant al pieței globale
print("Descarc datele...")
df = yf.download('SPY', start='2010-01-01', end='2023-01-01')

# --- 2. TRANSFORMAREA DATELOR (Secretul tău pentru licență) ---
# Aici facem datele staționare: calculăm cu cât la sută s-a modificat prețul față de ieri
df['Open_pct'] = df['Open'].pct_change()
df['High_pct'] = df['High'].pct_change()
df['Low_pct'] = df['Low'].pct_change()
df['Close_pct'] = df['Close'].pct_change()
df['Volume_pct'] = df['Volume'].pct_change()

# Adăugăm și faimosul RSI (Relative Strength Index) pe 14 zile pentru a testa analiza tehnică
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# --- 3. SETAREA ȚINTEI (Ce vrem să prezicem?) ---
# Target = 1 dacă prețul de închidere de MÂINE va fi mai mare strict decât cel de AZI
# Target = 0 dacă va scădea sau va stagna
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Curățăm rândurile cu date lipsă (NaN) apărute din cauza calculelor (RSI și pct_change)
df = df.dropna()

# --- 4. PREGĂTIREA PENTRU MACHINE LEARNING ---
# Alegem variabilele (features) pe care se bazează modelul
features = ['Open_pct', 'High_pct', 'Low_pct', 'Close_pct', 'Volume_pct', 'RSI']
X = df[features]
y = df['Target']

# Împărțim datele în Train (70%) și Test (30%)
# ATENȚIE: La serii de timp, NU amestecăm datele (shuffle=False), altfel modelul "vede" în viitor!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# --- 5. ANTRENAREA MODELULUI ---
print("Antrenez modelul (Regresie Logistică)...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Cerem modelului să ne dea PROBABILITATEA (între 0 și 1) ca acțiunea să crească mâine
y_pred_prob = model.predict_proba(X_test)[:, 1]

# --- 6. EVALUAREA ȘI GENERAREA CURBEI ROC ---
# Calculăm Rata de Adevărat Pozitiv (TPR) și Fals Pozitiv (FPR)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# --- 7. DESENAREA GRAFICULUI ---
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Modelul Nostru (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Ghicit Aleatoriu (AUC = 0.500)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Rata de Fals Pozitive (Alarme false)')
plt.ylabel('Rata de Adevărat Pozitive (Preziceri corecte)')
plt.title('Curba ROC: Predicția Direcției Pieței (Teoria Piețelor Eficiente)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

print(f"Scorul AUC final este: {roc_auc:.3f}")
if roc_auc < 0.55:
    print("Concluzie: Modelul este la fel de bun ca datul cu banul. Analiza tehnică nu bate piața!")