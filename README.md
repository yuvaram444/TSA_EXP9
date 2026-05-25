# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 25-05-26

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
"""
AIM:
To create a project on Time Series Analysis for stock price forecasting using
the ARIMA model in Python on the Nifty 50 dataset (HDFCBANK.NS), and compare
its performance against AR and MA models using RMSE and MAE evaluation metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & EXPLORE DATASET
# ═══════════════════════════════════════════════════════════════════════════════
STOCK    = 'HDFCBANK.NS'
CSV_PATH = 'nifty50_2000_2025.csv'   # keep CSV in same folder as notebook

df = pd.read_csv(CSV_PATH)
df['Date'] = pd.to_datetime(df['Date'])

stock_df = df[df['Stock'] == STOCK].sort_values('Date').copy()
stock_df['YearMonth'] = stock_df['Date'].dt.to_period('M')

monthly = stock_df.groupby('YearMonth')['Close'].mean()
monthly.index = monthly.index.to_timestamp()
monthly.index.freq = 'MS'

print("=" * 60)
print("STEP 1 — DATASET EXPLORATION")
print("=" * 60)
print(f"\nStock       : {STOCK}")
print(f"Shape       : {monthly.shape}")
print(f"Date range  : {monthly.index[0].date()} → {monthly.index[-1].date()}")
print(f"\nFirst 5 rows:\n{monthly.head().to_frame()}")
print(f"\nBasic Stats:\n{monthly.describe().round(2)}")

plt.figure(figsize=(14, 4))
plt.plot(monthly, color='steelblue', linewidth=1.5)
plt.title(f'{STOCK} — Monthly Avg Close Price (2000–2024)')
plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHECK STATIONARITY — ADF TEST, ACF, PACF, DIFFERENCING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — STATIONARITY CHECK")
print("=" * 60)

def adf_test(series, label='Series'):
    result = adfuller(series.dropna())
    print(f"\nADF Test — {label}")
    print(f"  ADF Statistic : {result[0]:.4f}")
    print(f"  p-value       : {result[1]:.4f}")
    print(f"  Critical Values: {', '.join([f'{k}: {v:.4f}' for k, v in result[4].items()])}")
    if result[1] > 0.05:
        print(f"  → NON-STATIONARY (p > 0.05)")
    else:
        print(f"  → STATIONARY (p < 0.05)")

adf_test(monthly, 'Original Close Price')

# ACF & PACF of original
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
plot_acf(monthly,  lags=40, ax=axes[0])
axes[0].set_title('ACF — Original Close Price')
axes[0].grid(alpha=0.3)
plot_pacf(monthly, lags=40, ax=axes[1])
axes[1].set_title('PACF — Original Close Price')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Differencing to make stationary
monthly_diff = monthly.diff().dropna()
adf_test(monthly_diff, '1st Order Differenced Close Price')

plt.figure(figsize=(14, 4))
plt.plot(monthly_diff, color='tomato', linewidth=1.2)
plt.title(f'{STOCK} — After 1st Order Differencing')
plt.xlabel('Date')
plt.ylabel('Δ Close Price (₹)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ACF & PACF after differencing
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
plot_acf(monthly_diff,  lags=40, ax=axes[0])
axes[0].set_title('ACF — After Differencing (use to find q for MA)')
axes[0].grid(alpha=0.3)
plot_pacf(monthly_diff, lags=40, ax=axes[1])
axes[1].set_title('PACF — After Differencing (use to find p for AR)')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DETERMINE ARIMA PARAMETERS p, d, q
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — ARIMA PARAMETERS (p, d, q)")
print("=" * 60)
print("  d = 1  (one differencing made series stationary)")
print("  p = AR lags (from PACF: significant lags before cutoff)")
print("  q = MA lags (from ACF:  significant lags before cutoff)")
print("  Manually selected: ARIMA(1, 1, 1)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT & FIT ARIMA(1,1,1)
# ═══════════════════════════════════════════════════════════════════════════════
train = monthly[:-12]
test  = monthly[-12:]

print("\n" + "=" * 60)
print("STEP 4 — FIT ARIMA(1,1,1)")
print("=" * 60)
print(f"\nTrain : {len(train)} months  ({train.index[0].date()} → {train.index[-1].date()})")
print(f"Test  : {len(test)} months   ({test.index[0].date()} → {test.index[-1].date()})")

arima_model = ARIMA(train, order=(1, 1, 1)).fit()
print(f"\n{arima_model.summary()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. ARIMA PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5 — ARIMA PREDICTIONS")
print("=" * 60)

arima_pred = arima_model.forecast(steps=12)
arima_pred.index = test.index

# ═══════════════════════════════════════════════════════════════════════════════
# 6. AUTO-FIT ARIMA (pmdarima)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6 — AUTO ARIMA")
print("=" * 60)

auto_model = auto_arima(
    train,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    information_criterion='aic'
)
print(f"\nAuto ARIMA best order : {auto_model.order}")
print(f"AIC                   : {auto_model.aic():.2f}")

auto_pred = pd.Series(
    auto_model.predict(n_periods=12),
    index=test.index
)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON — AR and MA models
# ═══════════════════════════════════════════════════════════════════════════════

# AR Model
ar_model  = AutoReg(train, lags=13).fit()
ar_pred   = ar_model.predict(start=len(train), end=len(train)+11)
ar_pred.index = test.index

# MA Model — ARIMA(0,0,1)
ma_model  = ARIMA(train, order=(0, 0, 1)).fit()
ma_pred   = ma_model.forecast(steps=12)
ma_pred.index = test.index

# ═══════════════════════════════════════════════════════════════════════════════
# 7. EVALUATE ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7 — MODEL EVALUATION (RMSE & MAE)")
print("=" * 60)

models = {
    'AR(13)'           : ar_pred,
    'MA(1)'            : ma_pred,
    'ARIMA(1,1,1)'     : arima_pred,
    f'Auto ARIMA{auto_model.order}': auto_pred,
}

results = []
for name, pred in models.items():
    rmse = np.sqrt(mean_squared_error(test, pred))
    mae  = mean_absolute_error(test, pred)
    results.append({'Model': name, 'RMSE': round(rmse, 2), 'MAE': round(mae, 2)})
    print(f"  {name:<25} RMSE: {rmse:>8.2f}   MAE: {mae:>8.2f}")

results_df = pd.DataFrame(results).sort_values('RMSE')
print(f"\nBest Model: {results_df.iloc[0]['Model']}  (lowest RMSE = {results_df.iloc[0]['RMSE']})")

# Prediction comparison plot
plt.figure(figsize=(14, 6))
plt.plot(train.iloc[-24:], color='steelblue', linewidth=1.5, label='Train (last 24 months)')
plt.plot(test,             color='black',     linewidth=2,   label='Actual Test')
colors_pred = ['tomato', 'darkorange', 'green', 'purple']
for (name, pred), color in zip(models.items(), colors_pred):
    plt.plot(pred, linestyle='--', linewidth=1.8, color=color, label=name)
plt.title(f'{STOCK} — ARIMA vs AR vs MA — Test Prediction Comparison')
plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL PREDICTION — Auto ARIMA on full data (next 12 months)
# ═══════════════════════════════════════════════════════════════════════════════
final_auto = auto_arima(
    monthly,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)
future_pred  = final_auto.predict(n_periods=12)
future_index = pd.date_range(start=monthly.index[-1], periods=13, freq='MS')[1:]
future_series = pd.Series(future_pred, index=future_index)

print("\n" + "=" * 60)
print("FINAL PREDICTION — Next 12 Months (2025)")
print("=" * 60)
print(future_series.round(2).to_frame(name='Predicted Close (₹)'))

plt.figure(figsize=(14, 5))
plt.plot(monthly,       color='steelblue', linewidth=1.5,               label='Historical Data')
plt.plot(future_series, color='tomato',    linewidth=2,   linestyle='--', label=f'Forecast — Auto ARIMA{final_auto.order}')
plt.axvline(x=monthly.index[-1], color='gray', linestyle=':', linewidth=1, label='Forecast Start')
plt.title(f'{STOCK} — Auto ARIMA Final Forecast (Jan–Dec 2025)')
plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(results_df['Model'], results_df['RMSE'], color=['tomato','darkorange','steelblue','green'])
axes[0].set_title('Model Comparison — RMSE')
axes[0].set_ylabel('RMSE (₹)')
axes[0].tick_params(axis='x', rotation=20)
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(results_df['Model'], results_df['MAE'], color=['tomato','darkorange','steelblue','green'])
axes[1].set_title('Model Comparison — MAE')
axes[1].set_ylabel('MAE (₹)')
axes[1].tick_params(axis='x', rotation=20)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle(f'{STOCK} — AR vs MA vs ARIMA Model Comparison', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nRESULT:")
print("Thus the program ran successfully — ARIMA model implemented and")
print("compared against AR and MA models using RMSE and MAE metrics.")
```

### OUTPUT:
```
============================================================
STEP 1 — DATASET EXPLORATION
============================================================

Stock       : HDFCBANK.NS
Shape       : (299,)
Date range  : 2000-02-01 → 2024-12-01

First 5 rows:
                Close
YearMonth            
2000-02-01  11.551923
2000-03-01  13.150000
2000-04-01  11.367125
2000-05-01  12.426304
2000-06-01  12.466023

Basic Stats:
count    299.00
mean     270.91
std      279.45
min        9.72
25%       37.88
50%      145.32
75%      500.53
max      913.28
Name: Close, dtype: float64
```
<img width="1117" height="321" alt="image" src="https://github.com/user-attachments/assets/830188e5-eaa0-4446-ac04-51a9c7cc5e8f" />

```
============================================================
STEP 2 — STATIONARITY CHECK
============================================================

ADF Test — Original Close Price
  ADF Statistic : 2.4411
  p-value       : 0.9990
  Critical Values: 1%: -3.4537, 5%: -2.8718, 10%: -2.5722
  → NON-STATIONARY (p > 0.05)
```

<img width="1111" height="630" alt="image" src="https://github.com/user-attachments/assets/cf3135c5-fc59-4a78-913a-c45e940686d7" />

```
============================================================
STEP 3 — ARIMA PARAMETERS (p, d, q)
============================================================
  d = 1  (one differencing made series stationary)
  p = AR lags (from PACF: significant lags before cutoff)
  q = MA lags (from ACF:  significant lags before cutoff)
  Manually selected: ARIMA(1, 1, 1)

============================================================
STEP 4 — FIT ARIMA(1,1,1)
============================================================

Train : 287 months  (2000-02-01 → 2023-12-01)
Test  : 12 months   (2024-01-01 → 2024-12-01)

                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  Close   No. Observations:                  287
Model:                 ARIMA(1, 1, 1)   Log Likelihood               -1208.858
Date:                Mon, 25 May 2026   AIC                           2423.716
Time:                        09:49:53   BIC                           2434.684
Sample:                    02-01-2000   HQIC                          2428.113
                         - 12-01-2023                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.3898      0.085     -4.595      0.000      -0.556      -0.224
ma.L1          0.7137      0.056     12.704      0.000       0.604       0.824
sigma2       274.4830      8.984     30.554      0.000     256.876     292.090
===================================================================================
Ljung-Box (L1) (Q):                   0.02   Jarque-Bera (JB):              1816.61
Prob(Q):                              0.89   Prob(JB):                         0.00
Heteroskedasticity (H):             181.85   Skew:                            -0.51
Prob(H) (two-sided):                  0.00   Kurtosis:                        15.30
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).

============================================================
STEP 5 — ARIMA PREDICTIONS
============================================================

============================================================
STEP 6 — AUTO ARIMA
============================================================

Auto ARIMA best order : (1, 1, 1)
AIC                   : 2419.87

============================================================
STEP 7 — MODEL EVALUATION (RMSE & MAE)
============================================================
  AR(13)                    RMSE:    96.17   MAE:    85.91
  MA(1)                     RMSE:   538.16   MAE:   524.07
  ARIMA(1,1,1)              RMSE:    68.73   MAE:    55.83
  Auto ARIMA(1, 1, 1)       RMSE:    72.70   MAE:    59.88

Best Model: ARIMA(1,1,1)  (lowest RMSE = 68.73)
```

<img width="1120" height="478" alt="image" src="https://github.com/user-attachments/assets/209465be-74fa-4eb7-bf4f-07198f75d734" />

```
============================================================
FINAL PREDICTION — Next 12 Months (2025)
============================================================
            Predicted Close (₹)
2025-01-01               920.09
2025-02-01               921.85
2025-03-01               925.13
2025-04-01               928.96
2025-05-01               931.06
2025-06-01               935.53
2025-07-01               937.32
2025-08-01               941.90
2025-09-01               943.69
2025-10-01               948.21
2025-11-01               950.10
2025-12-01               954.49

```


<img width="1118" height="794" alt="image" src="https://github.com/user-attachments/assets/312742da-fc83-4656-b00c-e4bb57cec459" />


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
