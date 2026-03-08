import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# --- 1. DESCĂRCAREA DATELOR ---
print("Descarc datele pentru SPY...")
df = yf.download('SPY', start='2010-01-01', end='2023-01-01')

# --- 2. TRANSFORMAREA DATELOR (Randamente zilnice) ---
df['Close_pct'] = df['Close'].pct_change()
df['Volume_pct'] = df['Volume'].pct_change()

# --- 3. CREAREA "MEMORIEI" PIEȚEI (Lagged Variables) ---
# Aici îi dăm modelului istoricul ultimelor 30 zile (o luna de bursă)
df['Lag_1'] = df['Close_pct'].shift(1) # Ce a făcut ieri
df['Lag_2'] = df['Close_pct'].shift(2) # Ce a făcut alaltăieri
df['Lag_3'] = df['Close_pct'].shift(3) # Acum 3 zile
df['Lag_4'] = df['Close_pct'].shift(4) # Acum 4 zile
df['Lag_5'] = df['Close_pct'].shift(5)
df['Lag_6'] = df['Close_pct'].shift(6)
df['Lag_7'] = df['Close_pct'].shift(7)
df['Lag_8'] = df['Close_pct'].shift(8)
df['Lag_9'] = df['Close_pct'].shift(9)
df['Lag_10'] = df['Close_pct'].shift(10)
df['Lag_11'] = df['Close_pct'].shift(11)
df['Lag_12'] = df['Close_pct'].shift(12)
df['Lag_13'] = df['Close_pct'].shift(13)
df['Lag_14'] = df['Close_pct'].shift(14)
df['Lag_15'] = df['Close_pct'].shift(15)
df['Lag_16'] = df['Close_pct'].shift(16)
df['Lag_17'] = df['Close_pct'].shift(17)
df['Lag_18'] = df['Close_pct'].shift(18)
df['Lag_19'] = df['Close_pct'].shift(19)
df['Lag_20'] = df['Close_pct'].shift(20)
df['Lag_21'] = df['Close_pct'].shift(21)

df['Lag_22'] = df['Close_pct'].shift(22)
df['Lag_23'] = df['Close_pct'].shift(23)
df['Lag_24'] = df['Close_pct'].shift(24)
df['Lag_25'] = df['Close_pct'].shift(25)
df['Lag_26'] = df['Close_pct'].shift(26)
df['Lag_27'] = df['Close_pct'].shift(27)
df['Lag_28'] = df['Close_pct'].shift(28)
df['Lag_29'] = df['Close_pct'].shift(29)
df['Lag_30'] = df['Close_pct'].shift(30)


# --- 4. SETAREA ȚINTEI (Prezicem ziua de mâine) ---
# Target = 1 (crește mâine), Target = 0 (scade mâine)
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Curățăm rândurile goale apărute la începutul tabelului din cauza "shift-urilor"
df = df.dropna()

# --- 5. PREGĂTIREA PENTRU MACHINE LEARNING ---
# Acum modelul se va uita la ziua curentă, la volum, plus la tot istoricul pe 5 zile!
features = ['Close_pct', 'Volume_pct', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5','Lag_6','Lag_7','Lag_8',
            'Lag_9','Lag_10','Lag_11','Lag_12','Lag_13','Lag_14','Lag_15','Lag_16','Lag_17','Lag_18','Lag_19',
            'Lag_20','Lag_21','Lag_22','Lag_23','Lag_24','Lag_25','Lag_26','Lag_27','Lag_28','Lag_29', 'Lag_30']

X = df[features]
y = df['Target']

# Împărțim datele cronic (70% antrenare, 30% testare)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- 6. ANTRENAREA MODELULUI ---
print("Antrenez modelul pe istoricul de 30 zile...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Extragem probabilitățile prezise
y_pred_prob = model.predict_proba(X_test)[:, 1]

# --- 7. CURBA ROC & SCORUL AUC ---
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# --- 8. VIZUALIZARE ---
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='purple', lw=2, label=f'Model cu Memorie 5 Zile (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Dat cu banul (AUC = 0.500)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Rata Alarme False (False Positive Rate)')
plt.ylabel('Rata Preziceri Corecte (True Positive Rate)')
plt.title('Curba ROC: Testarea Memoriei Pieței (Lagged Variables)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

print(f"Scorul AUC final este: {roc_auc:.3f}")