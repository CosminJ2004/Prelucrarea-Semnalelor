import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler

# --- 1. DATA FETCHING (S&P 500 + VIX) ---
print("Descarcare date S&P 500 si VIX...")
# Luam date din 2021 pana azi (pentru a nu bloca memoria RAM cu GP)
start_date = "2021-01-01"
end_date = "2024-01-01"

sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)['Close']
vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)['Close']

# Curatam datele
if hasattr(sp500, 'squeeze'): sp500 = sp500.squeeze()
if hasattr(vix, 'squeeze'): vix = vix.squeeze()

# Aliniem datele (intersectia datelor disponibile)
df = pd.DataFrame({'SP500': sp500, 'VIX': vix}).dropna()

# --- 2. PREPARARE INPUT 2D ---
# X1: Timpul (zile numerice)
# X2: VIX (frica)
# Y (Target): Pretul S&P 500

df['Days'] = np.arange(len(df))

X = df[['Days', 'VIX']].values  # Input 2D
y = df['SP500'].values          # Target 1D

# SCALARE (Crucial pentru GP)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# --- 3. ANTRENARE PROCES GAUSSIAN 2D ---
print("Antrenare Proces Gaussian 2D (Timp + Frica)...")

# Kernel 2D: 
# - Componenta 1: RBF pe Timp (Trendul pietei)
# - Componenta 2: RBF pe VIX (Relatia inversa pret-frica)
# - WhiteKernel: Zgomotul pietei
kernel = C(1.0) * RBF(length_scale=[1.0, 1.0]) + WhiteKernel(noise_level=0.1)

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=False)
gp.fit(X_scaled, y_scaled)

# --- 4. GENERARE SUPRAFATA DE PREDICTIE (GRID) ---
# Cream o grila fina pentru a desena "Pătura"
res = 30
x1_range = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), res) # Timp
x2_range = np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), res) # VIX

X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
X_grid_flat = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])

# Prezicem suprafata
y_pred_mesh_scaled, sigma_mesh = gp.predict(X_grid_flat, return_std=True)

# Inversam scalarea pentru a avea dolari reali
y_pred_mesh = scaler_y.inverse_transform(y_pred_mesh_scaled.reshape(-1, 1)).reshape(X1_mesh.shape)
# Pentru plotare axe
X1_real = X1_mesh * scaler_X.scale_[0] + scaler_X.mean_[0] # Zile
X2_real = X2_mesh * scaler_X.scale_[1] + scaler_X.mean_[1] # VIX

# --- 5. DETECTIE ANOMALII (Distanta fata de Suprafata) ---
# Prezicem punctele reale antrenate
y_pred_train_scaled = gp.predict(X_scaled)
y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()

# Reziduul: Cat de departe e pretul real fata de ce zice modelul (Timp+Vix)
residuals = np.abs(df['SP500'] - y_pred_train)
threshold = 2.5 * np.std(residuals) # Prag anomalie

anomalies = df[residuals > threshold]
normal = df[residuals <= threshold]

# --- 6. VIZUALIZARE 3D ---
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# A. Deseneaza Suprafata GP (Modelul "Normal")
surf = ax.plot_surface(X1_real, X2_real, y_pred_mesh, cmap=cm.coolwarm, 
                       alpha=0.4, linewidth=0, antialiased=False)

# B. Deseneaza Punctele Reale (Normale)
ax.scatter(normal['Days'], normal['VIX'], normal['SP500'], 
           c='black', s=10, alpha=0.5, label='Date Normale')

# C. Deseneaza Anomaliile (Rosu)
# Aceste puncte nu respecta relatia Timp-VIX-Pret
ax.scatter(anomalies['Days'], anomalies['VIX'], anomalies['SP500'], 
           c='red', s=50, marker='X', label='ANOMALII (Decuplare)')

# Etichete
ax.set_xlabel('Timp (Zile)')
ax.set_ylabel('VIX (Frica)')
ax.set_zlabel('Pret S&P 500')
ax.set_title('Model Gaussian 2D: Prețul în funcție de Timp și Volatilitate')

# Legend hack
import matplotlib.lines as mlines
red_x = mlines.Line2D([], [], color='red', marker='X', linestyle='None', markersize=10, label='Anomalie')
black_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=5, label='Normal')
ax.legend(handles=[red_x, black_dot])

plt.tight_layout()
plt.show()

print(f"Numar anomalii detectate: {len(anomalies)}")