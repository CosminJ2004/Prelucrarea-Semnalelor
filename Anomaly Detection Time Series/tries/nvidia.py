import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")

# --- FUNCȚIILE RĂMÂN ACELEAȘI (SSA, GARCH, PATTERN) ---
# (Copiază funcțiile detect_ssa, detect_garch, detect_pattern din scriptul anterior aici)
# Pentru economie de spațiu, nu le mai lipesc, ele sunt identice.
# Asigură-te că ai funcțiile definite mai sus!

def detect_ssa(series, window=50, threshold=3.0):
    series_np = series.values
    N = len(series_np)
    if N < window: return pd.Series(0, index=series.index), series
    K = N - window + 1
    X = np.column_stack([series_np[i:i+window] for i in range(K)])
    try:
        U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
        X_trend = Sigma[0] * np.outer(U[:, 0], VT[0, :])
    except:
        return pd.Series(0, index=series.index), series
    trend = np.zeros(N)
    count = np.zeros(N)
    for r in range(X_trend.shape[0]):
        for c in range(X_trend.shape[1]):
            trend[r+c] += X_trend[r,c]
            count[r+c] += 1
    trend = trend / count
    if len(trend) < N: trend = np.pad(trend, (0, N - len(trend)), 'edge')
    residuals = series_np - trend
    std_res = np.std(residuals)
    if std_res == 0: std_res = 1
    z_score = (residuals - np.mean(residuals)) / std_res
    anomalies = (np.abs(z_score) > threshold).astype(int)
    return pd.Series(anomalies, index=series.index), pd.Series(trend, index=series.index)

def detect_garch(series, trend, threshold=2.5):
    residuals = series - trend
    scale = 100
    res_scaled = residuals / scale
    try:
        model = arch_model(res_scaled, vol='Garch', p=1, q=1, mean='Zero', dist='Normal', rescale=False)
        res_fit = model.fit(disp='off', show_warning=False)
        cond_vol = res_fit.conditional_volatility * scale
    except:
        cond_vol = residuals.rolling(20).std()
    cond_vol = cond_vol.bfill().fillna(0)
    is_anomaly = (np.abs(residuals) > (cond_vol * threshold)).astype(int)
    return is_anomaly

def detect_pattern(series, template_series, threshold=0.7):
    series_vals = series.values
    temp_vals = template_series.values 
    temp_mean = np.mean(temp_vals)
    temp_std = np.std(temp_vals)
    if temp_std == 0: return pd.Series(0, index=series.index)
    norm_template = (temp_vals - temp_mean) / temp_std
    window = len(temp_vals)
    correlations = []
    for i in range(len(series_vals) - window + 1):
        window_slice = series_vals[i : i + window]
        w_mean = np.mean(window_slice)
        w_std = np.std(window_slice)
        if w_std == 0:
            correlations.append(0)
        else:
            norm_window = (window_slice - w_mean) / w_std
            corr = np.mean(norm_window * norm_template)
            correlations.append(corr)
    pad_width = len(series) - len(correlations)
    correlations = np.array([0.0] * pad_width + correlations)
    corr_series = pd.Series(correlations, index=series.index)
    is_anomaly = (corr_series > threshold).astype(int)
    return is_anomaly

# --- EXECUTIA PENTRU NVIDIA ---

if __name__ == "__main__":
    print("1. Descarcare date NVIDIA...")
    # Luam o perioada lunga sa prindem si Crypto Crash 2018 si Covid si 2022
    df = yf.download("NVDA", start="2018-01-01", end="2024-01-01", progress=False)['Close']
    if hasattr(df, 'squeeze'): df = df.squeeze()

    # DEFINIM SABLONUL: Folosim prăbușirea violentă din toamna 2018 (Crypto Winter)
    # NVIDIA a cazut 50% in cateva luni. Vrem sa vedem daca algoritmul gaseste forme similare (ex: 2022).
    # Perioada sablon: Oct 2018 - Dec 2018
    crash_template = df["2018-10-01":"2018-12-25"]
    
    print(f"Sablon definit: {len(crash_template)} zile (Crypto Crash 2018)")

    print("2. Rulare detectoare...")
    # A. SSA (Trend)
    # La actiuni volatile, SSA are nevoie de o fereastra putin mai mica sa nu fie prea "lenes"
    ssa_flags, trend = detect_ssa(df, window=40, threshold=3.0)

    # B. GARCH (Volatilitate)
    # NVIDIA are volatilitate naturala mare, deci GARCH se va adapta bine
    garch_flags = detect_garch(df, trend, threshold=2.5)

    # C. Pattern (Structura)
    # Cautam "Forma lui 2018" in restul istoricului
    pattern_flags = detect_pattern(df, crash_template, threshold=0.75)

    print("3. Calcul Scor Ensemble...")
    # Ponderile raman aceleasi
    w_ssa = 1.0
    w_garch = 1.0
    w_pattern = 1.5 

    ensemble_score = (ssa_flags * w_ssa) + (garch_flags * w_garch) + (pattern_flags * w_pattern)

    # --- VIZUALIZARE ---
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1.5], hspace=0.3)

    # AXA 1: Pret + Anomalii
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df, color='k', alpha=0.4, label='Pret NVIDIA')
    ax1.plot(df.index, trend, color='blue', alpha=0.3, linestyle='--', label='Trend SSA')

    risk_scores = ensemble_score[ensemble_score > 0]
    if len(risk_scores) > 0:
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["yellow", "orange", "red"])
        # Plotam doar punctele cu risc
        sc = ax1.scatter(ensemble_score.index[ensemble_score > 0], 
                         df[ensemble_score > 0], 
                         c=risk_scores, cmap=cmap, s=30, 
                         vmin=1, vmax=3.5, edgecolors='none', zorder=5, label='Anomalii')
        plt.colorbar(sc, ax=ax1, label="Scor Risc")
    
    # Marcam zona sablonului cu gri ca sa stim ce cautam
    ax1.axvspan(pd.to_datetime("2018-10-01"), pd.to_datetime("2018-12-25"), color='gray', alpha=0.2, label="SABLONUL (2018)")
    
    ax1.set_title("1. NVIDIA: Căutarea formei 'Crypto Crash 2018' în istorie")
    ax1.set_yscale('log') # Log scale e obligatoriu la NVIDIA (a crescut de la 40 la 500)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # AXA 2: Barcode
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    matrix = np.vstack([ssa_flags.values, garch_flags.values, pattern_flags.values])
    ax2.imshow(matrix, aspect='auto', cmap='Reds', interpolation='nearest', 
               extent=[df.index[0], df.index[-1], -0.5, 2.5])
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Pattern', 'GARCH', 'SSA'])
    ax2.set_title("2. Diagnostic ('Barcode')")

    # AXA 3: Scor
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(ensemble_score.index, ensemble_score, color='black', linewidth=1)
    ax3.fill_between(ensemble_score.index, 0, ensemble_score, where=(ensemble_score>2.5), color='red', alpha=0.5)
    ax3.fill_between(ensemble_score.index, 0, ensemble_score, where=((ensemble_score>1) & (ensemble_score<=2.5)), color='orange', alpha=0.5)
    ax3.axhline(2.5, color='red', linestyle='--')
    ax3.set_title("3. Scor Final de Risc")
    
    plt.show()
    
    # Raport
    high_risk = ensemble_score[ensemble_score > 2.5]
    if not high_risk.empty:
        print(f"\nPerioade Critice Detectate (similare cu 2018):")
        print(high_risk.index.strftime('%Y-%m-%d').unique()[:10])