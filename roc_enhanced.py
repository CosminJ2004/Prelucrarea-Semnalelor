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

# --- 2. TRANSFORMAREA DATELOR (Procentele relative) ---
df['Open_pct'] = df['Open'].pct_change()
df['High_pct'] = df['High'].pct_change()
df['Low_pct'] = df['Low'].pct_change()
df['Close_pct'] = df['Close'].pct_change()
df['Volume_pct'] = df['Volume'].pct_change()

# --- 3. CALCULAREA CELOR 4 INDICATORI TEHNICI ---

# A. RSI (14 perioade)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# B. MACD Histogram (12, 27, 9) - Setările din imagine
ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
ema_slow = df['Close'].ewm(span=27, adjust=False).mean()
macd_line = ema_fast - ema_slow
signal_line = macd_line.ewm(span=9, adjust=False).mean()
df['MACD_hist'] = macd_line - signal_line

# C. Bollinger Bands (BB) - %B (20 perioade, 2 deviații)
bb_sma = df['Close'].rolling(window=20).mean()
bb_std = df['Close'].rolling(window=20).std()
upper_band = bb_sma + (2 * bb_std)
lower_band = bb_sma - (2 * bb_std)
# Calculăm %B (unde se află prețul relativ la benzi)
df['BB'] = (upper_band - df['Close']) / (upper_band - lower_band)

# D. MFI (Money Flow Index - 14 perioade)
tp = (df['High'] + df['Low'] + df['Close']) / 3
rmf = tp * df['Volume']
# Găsim fluxul de bani pozitiv și negativ
positive_flow = rmf.where(tp > tp.shift(1), 0).rolling(window=14).sum()
negative_flow = rmf.where(tp < tp.shift(1), 0).rolling(window=14).sum()
mfi_ratio = positive_flow / negative_flow
df['MFI'] = 100 - (100 / (1 + mfi_ratio))

# --- 4. SETAREA ȚINTEI ---
# Vrem să prezicem dacă ziua următoare se închide pe verde (1) sau roșu (0)
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Curățăm datele (aruncăm primele ~27 de zile pentru că MACD și BB au nevoie de timp să se calculeze)
df = df.dropna()

# --- 5. MACHINE LEARNING ---
# Am adăugat strategia completă în features
features = ['Open_pct', 'High_pct', 'Low_pct', 'Close_pct', 'Volume_pct', 
            'MACD_hist', 'MFI', 'BB', 'RSI']

X = df[features]
y = df['Target']

# Împărțim datele (Fără shuffle, păstrăm ordinea timpului!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

print("Antrenez modelul pe 4 indicatori și procente zilnice...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# --- 6. CURBA ROC & SCORUL AUC ---
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# --- 7. VIZUALIZARE ---
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='crimson', lw=2, label=f'Modelul Nostru 4 Ind (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Dat cu banul (AUC = 0.500)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Rata Alarme False (False Positive Rate)')
plt.ylabel('Rata Preziceri Corecte (True Positive Rate)')
plt.title('Curba ROC: Eșecul Analizei Tehnice Multiple')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

print(f"Scorul AUC final este: {roc_auc:.3f}")