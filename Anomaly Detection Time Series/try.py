import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from arch import arch_model
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")

# --- 1. UTILS (SSA & OU FIT) ---
def get_ssa_trend(series, window=60):
    series_np = series.values
    N = len(series_np)
    if N < window: return series # Fallback
    
    K = N - window + 1
    X = np.column_stack([series_np[i:i+window] for i in range(K)])
    
    try:
        U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
        X_trend = Sigma[0] * np.outer(U[:, 0], VT[0, :])
    except:
        return series
    
    trend = np.zeros(N)
    count = np.zeros(N)
    for r in range(X_trend.shape[0]):
        for c in range(X_trend.shape[1]):
            trend[r+c] += X_trend[r,c]
            count[r+c] += 1
    trend = trend / count
    if len(trend) < N: trend = np.pad(trend, (0, N - len(trend)), 'edge')
    return pd.Series(trend, index=series.index)

def fit_ou_params(residuals):
    # Calibrare OU simplificata pentru a gasi sigma de echilibru
    x_t = residuals[:-1].values.reshape(-1, 1)
    x_t1 = residuals[1:].values.reshape(-1, 1)
    reg = LinearRegression().fit(x_t, x_t1)
    a = reg.coef_[0][0]
    dt = 1
    theta = -np.log(a) / dt
    if theta < 1e-5: theta = 1e-5
    
    pred = reg.predict(x_t)
    epsilon = x_t1 - pred
    sigma_epsilon = np.std(epsilon)
    
    sigma_eq = sigma_epsilon / np.sqrt(2 * theta * (1-np.exp(-2*theta)))
    return sigma_eq

# --- 2. DETECTOARELE (GARCH, OU, PATTERN) ---

def detect_garch(residuals, threshold=2.5):
    """ Metoda 1: GARCH (Volatilitate Dinamica) """
    scale = 100
    res_scaled = residuals / scale
    try:
        model = arch_model(res_scaled, vol='Garch', p=1, q=1, mean='Zero', dist='Normal', rescale=False)
        res_fit = model.fit(disp='off', show_warning=False)
        cond_vol = res_fit.conditional_volatility * scale
    except:
        cond_vol = residuals.rolling(20).std()
    
    # Fix Pandas 2.0
    cond_vol = cond_vol.bfill().fillna(0)
    
    # Returnam semnal binar (1 sau 0)
    flags = (np.abs(residuals) > (cond_vol * threshold)).astype(int)
    return flags

def detect_ou(residuals, threshold=3.0):
    """ Metoda 2: OU (Volatilitate Statica / Echilibru) """
    # Calibram pe tot istoricul disponibil (sau rulant)
    sigma_eq = fit_ou_params(residuals)
    limit = threshold * sigma_eq
    flags = (np.abs(residuals) > limit).astype(int)
    return flags

from scipy.spatial.distance import cdist

def dtw_distance_fast(s1, s2):
    """
    Calculeaza distanta DTW intre doua serii normalizate.
    Foloseste matricea de distante (cdist) pentru viteza.
    """
    # Matricea distantelor locale (Euclidian)
    dist_matrix = cdist(s1.reshape(-1, 1), s2.reshape(-1, 1), metric='euclidean')
    
    n, m = dist_matrix.shape
    
    # Matricea de costuri acumulate (Dynamic Programming)
    dtw = np.zeros((n + 1, m + 1))
    dtw[1:, 0] = np.inf
    dtw[0, 1:] = np.inf
    dtw[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_matrix[i-1, j-1]
            # Luam minimul vecinilor
            dtw[i, j] = cost + min(dtw[i-1, j],    # Insertie
                                   dtw[i, j-1],    # Stergere
                                   dtw[i-1, j-1])  # Match
            
    return dtw[n, m]

def detect_signature_dtw(series, template, threshold=0.75):
    """
    Metoda 3 (Upgrade): Signature Matching folosind DTW.
    Mai lent, dar detecteaza crize care se desfasoara la viteze diferite.
    """
    series_vals = series.values
    temp_vals = template.values
    
    # 1. Normalizare Template (Z-Score) - CRUCIAL PENTRU DTW
    # DTW calculeaza distante absolute, deci scara trebuie sa fie identica (deviatii standard)
    temp_mean = np.mean(temp_vals)
    temp_std = np.std(temp_vals)
    if temp_std == 0: return pd.Series(0, index=series.index)
    
    norm_template = (temp_vals - temp_mean) / temp_std
    window = len(temp_vals)
    
    dtw_scores = []
    
    # Sliding window
    # NOTA: Poate dura 10-30 secunde pe un istoric lung!
    for i in range(len(series_vals) - window + 1):
        win = series_vals[i : i+window]
        
        if np.std(win) == 0:
            dtw_scores.append(0) # Scor 0 (Risc mic)
        else:
            # Normalizam fereastra curenta
            win_norm = (win - np.mean(win)) / np.std(win)
            
            # Calculam Distanta DTW
            dist = dtw_distance_fast(win_norm, norm_template)
            
            # CONVERSIE DISTANTA -> SCOR DE SIMILARITATE (0 la 1)
            # Distanta 0 => Scor 1 (Match perfect)
            # Distanta Mare => Scor 0
            # Formula empirica: 1 / (1 + distanta_normalizata)
            # Normalizam distanta la lungimea ferestrei
            normalized_dist = dist / window 
            similarity = 1 / (1 + normalized_dist)
            
            dtw_scores.append(similarity)
            
    # Padding la inceput
    pad = len(series) - len(dtw_scores)
    dtw_scores = np.array([0.0]*pad + dtw_scores)
    
    # Returnam semnal binar bazat pe prag
    # Atentie: Pragul pentru DTW trebuie recalibrat (poate 0.65 e mai bun decat 0.75)
    flags = pd.Series((dtw_scores > threshold).astype(int), index=series.index)
    return flags
# --- 3. EXECUTIA AGREGATA ---

if __name__ == "__main__":
    print("1. Descarcare date...")
    # Date 2019-2024
    df = yf.download("^GSPC", start="2000-01-01", end="2024-01-01", progress=False)['Close']
    if hasattr(df, 'squeeze'): df = df.squeeze()

    # DEFINIM SEMNATURA (Sablonul): Crash-ul COVID
    signature_template = df["2020-02-20":"2020-03-23"]

    print("2. Procesare Semnale...")
    # A. Extragere Trend si Reziduuri
    trend = get_ssa_trend(df)
    residuals = df - trend

    # B. Rulare Detectoare Individuale
    # 1. GARCH (Reactioneaza la panica de moment)
    signal_garch = detect_garch(residuals, threshold=2.5)
    
    # 2. OU (Reactioneaza la deviatii structurale fata de medie)
    signal_ou = detect_ou(residuals, threshold=3.0)
    
    # 3. Signature (Reactioneaza la forma graficului)
    signal_sign = detect_signature_dtw(df, signature_template, threshold=0.80)

    print("3. Calcul Scor Ponderat (Ensemble)...")
    
    # --- PONDERILE (AICI E MAGIA) ---
    W_GARCH = 1.0  # Pondere mica (e zgomotos)
    W_OU = 1.5     # Pondere medie (e fizica pietei)
    W_SIGN = 2.5   # Pondere MARE (daca seamana cu Covid, e grav!)
    
    # Calculam scorul total
    total_score = (signal_garch * W_GARCH) + \
                  (signal_ou * W_OU) + \
                  (signal_sign * W_SIGN)

    # --- 4. VIZUALIZARE PROFESIONALA ---
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.25)

    # PANEL 1: Pretul si Punctele de Alerta Colorate
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df, color='black', alpha=0.4, label='Pret S&P 500')
    ax1.plot(df.index, trend, color='blue', alpha=0.3, linestyle='--', label='Trend SSA')
    
    # Scatter plot colorat dupa Scor
    mask = total_score > 0
    if mask.any():
        sc = ax1.scatter(df.index[mask], df[mask], 
                         c=total_score[mask], cmap='jet', 
                         s=total_score[mask]*20, # Marimea punctului depinde de scor
                         edgecolors='black', zorder=5, vmin=0, vmax=5)
        plt.colorbar(sc, ax=ax1, label="Scor Risc Total")
    
    ax1.set_title("1. Ensemble Detection: Mărimea și Culoarea punctului indică Gravitatea")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # PANEL 2: Contributia Fiecarei Metode (Stacked Area)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Pregatim datele pentru stackplot
    dates = df.index
    y1 = signal_garch * W_GARCH
    y2 = signal_ou * W_OU
    y3 = signal_sign * W_SIGN
    
    ax2.stackplot(dates, y1, y2, y3, 
                  labels=['GARCH (Vol)', 'OU (Structura)', 'Signature (Forma)'],
                  colors=['#FFA07A', '#20B2AA', '#DC143C'], alpha=0.7)
    
    ax2.set_title("2. Compoziția Riscului: Cine declanșează alarma?")
    ax2.legend(loc='upper left')
    ax2.set_ylabel("Pondere")
    ax2.grid(True, alpha=0.3)
    
    # PANEL 3: Scorul Total vs Prag Critic
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(total_score.index, total_score, color='black', linewidth=1.5)
    
    # Zona Critica (Scor > 3.5 inseamna ca cel putin 2 metode tari au tipat)
    CRITICAL_LEVEL = 3.5
    ax3.axhline(CRITICAL_LEVEL, color='red', linestyle='--', linewidth=2, label='Prag Critic (Actionable)')
    ax3.fill_between(total_score.index, CRITICAL_LEVEL, total_score, 
                     where=(total_score >= CRITICAL_LEVEL), 
                     color='red', alpha=0.5)
    
    ax3.set_title("3. Scor Final de Anomalie (Suma Ponderată)")
    ax3.set_ylabel("Scor")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # --- RAPORT ---
    print("\n--- RAPORT DE RISC ---")
    critical_days = total_score[total_score >= CRITICAL_LEVEL]
    print(f"Număr zile cu risc CRITIC (Scor >= {CRITICAL_LEVEL}): {len(critical_days)}")
    if not critical_days.empty:
        print("Perioade Critice:")
        # Grupam datele consecutive
        idx = critical_days.index
        # Logica simpla de afisare a clusterelor de date
        print(f"Ultima alarmă majoră: {idx[-1].strftime('%Y-%m-%d')}")