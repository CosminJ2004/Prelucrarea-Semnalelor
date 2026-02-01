import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from arch import arch_model
import warnings

# Ignoram warning-urile (Pandas FutureWarnings etc.)
warnings.filterwarnings("ignore")

# --- 1. FUNCTIILE INDIVIDUALE (MARTORII) ---

def detect_ssa(series, window=50, threshold=3.0):
    """
    Metoda 1: SSA - Detecteaza deviatii de nivel fata de trend.
    """
    series_np = series.values
    N = len(series_np)
    if N < window: return pd.Series(0, index=series.index), series
    
    # Matricea Traiectoriei
    K = N - window + 1
    X = np.column_stack([series_np[i:i+window] for i in range(K)])
    
    try:
        U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
        # Trend (Prima componenta)
        X_trend = Sigma[0] * np.outer(U[:, 0], VT[0, :])
    except:
        return pd.Series(0, index=series.index), series # Fallback
    
    # Reconstructie (Diagonal Averaging)
    trend = np.zeros(N)
    count = np.zeros(N)
    for r in range(X_trend.shape[0]):
        for c in range(X_trend.shape[1]):
            trend[r+c] += X_trend[r,c]
            count[r+c] += 1
    trend = trend / count
    
    # Padding simplu la capete
    if len(trend) < N:
        trend = np.pad(trend, (0, N - len(trend)), 'edge')
    
    residuals = series_np - trend
    
    # Evitam impartirea la zero
    std_res = np.std(residuals)
    if std_res == 0: std_res = 1
        
    z_score = (residuals - np.mean(residuals)) / std_res
    
    # Returnam 1 unde z-score e mare, 0 altfel
    anomalies = (np.abs(z_score) > threshold).astype(int)
    return pd.Series(anomalies, index=series.index), pd.Series(trend, index=series.index)

def detect_garch(series, trend, threshold=2.5):
    """
    Metoda 2: GARCH - Detecteaza explozii de volatilitate.
    """
    residuals = series - trend
    scale = 100
    res_scaled = residuals / scale
    
    # Model GARCH(1,1)
    try:
        model = arch_model(res_scaled, vol='Garch', p=1, q=1, mean='Zero', dist='Normal', rescale=False)
        res_fit = model.fit(disp='off', show_warning=False)
        cond_vol = res_fit.conditional_volatility * scale
    except:
        # Fallback simplu daca GARCH nu converge
        cond_vol = residuals.rolling(20).std()
    
    # --- FIX PANDAS 2.0 ---
    # Folosim bfill() fara argumente
    cond_vol = cond_vol.bfill().fillna(0)
    # ----------------------
    
    is_anomaly = (np.abs(residuals) > (cond_vol * threshold)).astype(int)
    return is_anomaly

def detect_pattern(series, template_series, threshold=0.7):
    """
    Metoda 3: Pattern Matching (Corelatie Rulanta - Implementare Numpy).
    """
    series_vals = series.values
    temp_vals = template_series.values 
    
    # Normalizam sablonul (Z-Score)
    temp_mean = np.mean(temp_vals)
    temp_std = np.std(temp_vals)
    if temp_std == 0: return pd.Series(0, index=series.index)
    
    norm_template = (temp_vals - temp_mean) / temp_std
    window = len(temp_vals)
    
    correlations = []
    
    # Sliding window manual (Numpy)
    for i in range(len(series_vals) - window + 1):
        window_slice = series_vals[i : i + window]
        
        w_mean = np.mean(window_slice)
        w_std = np.std(window_slice)
        
        if w_std == 0:
            correlations.append(0)
        else:
            norm_window = (window_slice - w_mean) / w_std
            # Pearson Correlation
            corr = np.mean(norm_window * norm_template)
            correlations.append(corr)
            
    # Padding la inceput
    pad_width = len(series) - len(correlations)
    correlations = np.array([0.0] * pad_width + correlations)
    
    corr_series = pd.Series(correlations, index=series.index)
    
    # Anomalie daca corelatia e mare
    is_anomaly = (corr_series > threshold).astype(int)
    return is_anomaly

# --- 2. EXECUTIA ENSEMBLE ---

if __name__ == "__main__":
    print("1. Descarcare date...")
    # Luam o perioada mai lunga care include COVID
    df = yf.download("^GSPC", start="2019-01-01", end="2024-01-01", progress=False)['Close']
    if hasattr(df, 'squeeze'): df = df.squeeze()

    # Definim 'Amprenta' (Crash-ul COVID ca sablon)
    covid_crash = df["2020-02-20":"2020-03-23"]

    print("2. Rulare detectoare individuale...")
    # A. SSA
    ssa_flags, trend = detect_ssa(df, window=60, threshold=3.0)

    # B. GARCH
    garch_flags = detect_garch(df, trend, threshold=2.5)

    # C. Pattern
    pattern_flags = detect_pattern(df, covid_crash, threshold=0.80)

    print("3. Calculare Scor Ensemble (Votare)...")
    # Ponderi 
    w_ssa = 1.0
    w_garch = 1.0
    w_pattern = 1.5 

    ensemble_score = (ssa_flags * w_ssa) + (garch_flags * w_garch) + (pattern_flags * w_pattern)

    # --- 4. VIZUALIZARE COMPLEXA ---
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1.5], hspace=0.3)

    # --- AXA 1: PRET + ANOMALII AGREGATE ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df, color='k', alpha=0.4, label='Pret S&P 500')
    ax1.plot(df.index, trend, color='blue', alpha=0.3, linestyle='--', label='Trend SSA')

    # Scatter plot cu colormap
    risk_dates = df.index[ensemble_score > 0]
    risk_vals = df[ensemble_score > 0]
    risk_scores = ensemble_score[ensemble_score > 0]

    if len(risk_scores) > 0:
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["yellow", "orange", "red"])
        sc = ax1.scatter(risk_dates, risk_vals, c=risk_scores, cmap=cmap, s=50, 
                         vmin=1, vmax=3.5, edgecolors='black', zorder=5, label='Anomalii Ensemble')
        plt.colorbar(sc, ax=ax1, label="Scor Risc")
    
    ax1.set_title("1. Ensemble Detection: Severitatea Anomaliilor (Culoare = Scor de Risc)")
    ax1.set_ylabel("Pret ($)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- AXA 2: BARCODE (CINE A TIPAT?) ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    matrix = np.vstack([ssa_flags.values, garch_flags.values, pattern_flags.values])

    ax2.imshow(matrix, aspect='auto', cmap='Reds', interpolation='nearest', 
               extent=[df.index[0], df.index[-1], -0.5, 2.5])

    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Pattern (Struct)', 'GARCH (Vol)', 'SSA (Trend)'])
    ax2.set_title("2. Diagnostic ('Barcode'): Ce metodă a declanșat alarma?")
    ax2.grid(False)

    # --- AXA 3: SCORUL DE RISC AGREGAT ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(ensemble_score.index, ensemble_score, color='black', linewidth=1.5, label='Scor Total')
    
    # Fill intre zone
    ax3.fill_between(ensemble_score.index, 0, ensemble_score, where=(ensemble_score>2.5), color='red', alpha=0.5, label='Zona CRITICA')
    ax3.fill_between(ensemble_score.index, 0, ensemble_score, where=((ensemble_score>1) & (ensemble_score<=2.5)), color='orange', alpha=0.5, label='Zona Pericol')

    ax3.set_ylabel("Scor Sumat")
    ax3.set_title("3. Semnalul Final de Risc (Suma Ponderată)")
    ax3.axhline(2.5, color='red', linestyle='--', linewidth=1)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- RAPORT ---
    print("\n--- REZUMAT ANOMALII MAJORE (Scor > 2.5) ---")
    high_risk = ensemble_score[ensemble_score > 2.5]
    if not high_risk.empty:
        # Convertim indexul la date fara timezone pentru afisare curata
        dates_str = high_risk.index.strftime('%Y-%m-%d').unique()
        print(f"Date Critice detectate: {len(dates_str)}")
        print(dates_str[:10]) # Afisam primele 10
    else:
        print("Nu s-au detectat suprapuneri critice majore in perioada selectata.")