import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA

# --- Seria ta (input) ---
N = 400
np.random.seed(42)
t = np.linspace(0, N-1, N)
a, b, c =  0.000001, -0.000002, -0.000003
trend = a * t**2 + b * t + c
period1 = 0.01
period2 = 0.005
seasonal = 0.1 * np.sin(2 * np.pi * t * period1) + 0.2 * np.sin(2 * np.pi * t * period2)
sigma = 0.01
noise = np.random.normal(0, sigma, N)
series = trend + seasonal + noise

# --- Centerare (termenii de eroare vor fi deviațiile față de această medie) ---
mu = series.mean()
series_centered = series - mu

# --- Grid search p,q <= 20 (ARIMA with d=0 => ARMA) ---
max_order = 20
best_aic = np.inf
best_order = None
best_model = None

# Scurtăm mesajele de warning pentru claritate
warnings.filterwarnings("ignore")

# Grid search (note: poate dura ceva timp)
for p in range(0, max_order+1):
    for q in range(0, max_order+1):
        if p == 0 and q == 0:
            continue
        try:
            # Fit ARIMA with d=0 (i.e. ARMA)
            model = ARIMA(series_centered, order=(p, 0, q))
            fitted = model.fit()
            aic = fitted.aic
            if aic < best_aic:
                best_aic = aic
                best_order = (p, q)
                best_model = fitted
                print(f"New best: p={p}, q={q}, AIC={aic:.3f}")
        except Exception as e:
            # skip combinations that don't converge or error out
            # print(f"skip p={p} q={q} -> {e}")
            continue

if best_model is None:
    raise RuntimeError("Nu s-a găsit niciun model valid în grid search (verifică statsmodels).")

print("\n=== Rezultat final ===")
print(f"Parametrii optimi: p={best_order[0]}, q={best_order[1]}")
print(f"AIC (best): {best_aic:.3f}")

# --- Predicții in-sample (fitted values) și reconstrucția cu media inițială ---
fitted_centered = best_model.fittedvalues  # in-sample fitted (centred)
# unele modele pot returna fittedvalues de lungime N (sau N-?); aliniem pe index
# statsmodels returneaza de obicei un array compatibil cu input
fitted_full = fitted_centered + mu  # adăugăm înapoi media

# --- Residuals (epsilon): deviațiile față de medie pentru fiecare orizont de calcul) ---
# here residuals = series_centered - fitted_centered (in-sample)
residuals = best_model.resid
res_mean = residuals.mean()
res_std = residuals.std(ddof=0)

print(f"\nResiduurile (epsilon) -> media: {res_mean:.5e}, deviația standard: {res_std:.5e}")

# --- Plot: original vs fitted ---
plt.figure(figsize=(12,5))
plt.plot(series, label="Seria originală")
plt.plot(fitted_full, linestyle='--', label=f"Predicții ARMA({best_order[0]},{best_order[1]}) in-sample")
plt.legend()
plt.title(f"ARMA({best_order[0]},{best_order[1]}) - original vs in-sample fitted")
plt.xlabel("t")
plt.ylabel("valoare")
plt.grid(True)
plt.show()

# --- Plot residuals ---
plt.figure(figsize=(12,3))
plt.plot(residuals, label="Residuurile (epsilon)")
plt.hlines([res_mean, res_mean+res_std, res_mean-res_std], xmin=0, xmax=len(residuals)-1,
           colors=['k','r','r'], linestyles=['--',':',':'], label="mean ± std")
plt.legend()
plt.title("Residuurile modelului (in-sample)")
plt.grid(True)
plt.show()

# --- Optional: rolling one-step-ahead out-of-sample forecast errors ---
# Vom face o simplă evaluare out-of-sample cu rolling forecast (last 20% ca test)
test_frac = 0.2
train_len = int((1-test_frac) * N)
train_series = series_centered[:train_len]
test_series = series_centered[train_len:]

# Refit model on train with best order
p_opt, q_opt = best_order
model_train = ARIMA(train_series, order=(p_opt, 0, q_opt))
fitted_train = model_train.fit()

# rolling one-step forecast on test
history = train_series.copy()
one_step_preds = []
for t_idx in range(len(test_series)):
    # forecast(1) returns array
    try:
        fc = fitted_train.get_forecast(steps=1, exog=None)
        pred = fc.predicted_mean[0]
    except:
        # fallback: use last fitted value
        pred = history[-1]
    one_step_preds.append(pred)
    # append the actual next value to history and re-fit/update model for the next step
    # here re-fitting each step would be slow; instead append and re-estimate quickly:
    history = np.append(history, test_series[t_idx])
    # re-fit model on updated history for next iteration
    try:
        fitted_train = ARIMA(history, order=(p_opt,0,q_opt)).fit(disp=0)
    except:
        # if fails, keep old fitted_train (approx)
        pass

# reconstrucție cu media
one_step_preds_full = np.array(one_step_preds) + mu
test_full = test_series + mu

mse_one_step = np.mean((test_full - one_step_preds_full)**2)
print(f"\nRolling one-step-ahead out-of-sample MSE: {mse_one_step:.5e}")

plt.figure(figsize=(12,4))
plt.plot(np.arange(train_len, N), test_full, label="Test (real)")
plt.plot(np.arange(train_len, N), one_step_preds_full, '--', label="One-step rolling preds")
plt.legend()
plt.title("Rolling one-step-ahead preds (out-of-sample)")
plt.grid(True)
plt.show()
