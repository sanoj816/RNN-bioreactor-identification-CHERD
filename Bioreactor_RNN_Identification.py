# ============================================================
# MASTER SCRIPT
# Bioreactor Identification (ONE-STEP)
# LSTM / BiLSTM / GRU / BiGRU
# 2 Inputs, 6 Outputs
# ============================================================

# 1. IMPORTS & GLOBAL SETTINGS
# ============================================================
import numpy as np
import pandas as pd
import os
import joblib

from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import Sequential
from keras.layers import (
    LSTM, GRU, Dense, Dropout,
    Activation, Bidirectional
)
from keras.callbacks import Callback

np.random.seed(42)

# ============================================================
# DIRECTORIES
# ============================================================
RESULT_DIR = "results"
DATA_DIR   = os.path.join(RESULT_DIR, "data")
MODEL_DIR  = os.path.join(RESULT_DIR, "models")

for d in [DATA_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# OUTPUT DEFINITIONS
# ============================================================
OUTPUTS = ["cX", "cP", "cS", "cO2", "Tr", "Tag"]
N_OUT = len(OUTPUTS)
Y_MIN, Y_MAX = 0.0, 100.0

# ============================================================
# 2. RANDOM STEP SIGNAL
# ============================================================
def random_step_signal(n, umin, umax, minPW, maxPW, seed=None):
    if seed is not None:
        np.random.seed(seed)
    u, k = np.zeros(n), 0
    while k < n:
        pw = np.random.randint(minPW, maxPW + 1)
        amp = np.random.uniform(umin, umax)
        u[k:k+pw] = amp
        k += pw
    return u[:n]

# ============================================================
# 3. BIOREACTOR ODE
# ============================================================
def bioreactor_ode(t, x, Fi, Fag):
    V, cX, cP, cS, cO2, Tr, Tag = x
    Fe = Fi

    miu_P = 1.79
    Ks, Ks1 = 1.03, 1.68
    Kp, Kp1 = 0.139, 0.07
    Rsx, Rsp = 0.607, 0.435
    YO2, KO2, miu_O2 = 0.97, 8.86, 0.5

    A1, A2 = 9.5e8, 2.55e33
    Ea1, Ea2 = 55000.0, 220000.0
    R = 8.31

    Kla0 = 38.0
    KT = 100 * 3600
    Vm, AT = 50.0, 1.0
    ro, ccal = 1080.0, 4.18
    roag, ccalag = 1000.0, 4.18
    deltaH = 518.0

    T_in, Tiag, cS_in = 25.0, 15.0, 60.0

    c0st = 14.16 - 0.3943*Tr + 0.007714*Tr**2 - 0.0000646*Tr**3
    Kla = Kla0 * (1.024**(Tr-20))

    rO2 = miu_O2 * cO2 * cX / YO2 / (KO2 + cO2) * 1000
    miu_X = A1*np.exp(-Ea1/(R*(Tr+273))) - A2*np.exp(-Ea2/(R*(Tr+273)))

    dV = Fi - Fe
    dcX = miu_X*cX*cS/(Ks+cS)*np.exp(-Kp*cP) - (Fe/V)*cX
    dcP = miu_P*cX*cS/(Ks1+cS)*np.exp(-Kp1*cP) - (Fe/V)*cP
    dcS = (-miu_X*cX*cS/(Ks+cS)*np.exp(-Kp*cP)/Rsx
           -miu_P*cX*cS/(Ks1+cS)*np.exp(-Kp1*cP)/Rsp
           + (Fi/V)*cS_in - (Fe/V)*cS)
    dcO2 = Kla*(c0st-cO2) - rO2 - (Fe/V)*cO2
    dTr = (1/32*V*rO2*deltaH - KT*AT*(Tr-Tag)
           + Fi*ro*ccal*(T_in+273)
           - Fe*ro*ccal*(Tr+273))/(ro*ccal*V)
    dTag = (Fag*ccalag*roag*(Tiag-Tag)
            + KT*AT*(Tr-Tag))/(Vm*roag*ccalag)

    return [dV, dcX, dcP, dcS, dcO2, dTr, dTag]


# ============================================================
# 4. DATA GENERATION
# ============================================================
Ts, n = 1, 15000
t = np.arange(n+1) * Ts

Fi  = random_step_signal(n,0,100,160,200,seed=1)
Fag = random_step_signal(n,0,100,150,200,seed=2)

x0 = [1000,0.9,12.5,29.7,3.1,29.6,27.0]
X = np.zeros((n+1,7))
X[0] = x0

for k in range(n):
    sol = solve_ivp(bioreactor_ode,[t[k],t[k+1]],X[k],
                    args=(Fi[k],Fag[k]),method="BDF")
    X[k+1] = sol.y[:,-1]

df_sim = pd.DataFrame({
    "Fi": np.r_[Fi,Fi[-1]],
    "Fag": np.r_[Fag,Fag[-1]],
    "cX": X[:,1], "cP": X[:,2], "cS": X[:,3],
    "cO2": X[:,4], "Tr": X[:,5], "Tag": X[:,6]
})

# ============================================================
# 5. DATASET PREPARATION
# ============================================================
WINDOW = 15

Xraw = df_sim[["Fi","Fag"] + OUTPUTS].values
yraw = df_sim[OUTPUTS].values

sx, sy = MinMaxScaler(), MinMaxScaler()
Xs, ys = sx.fit_transform(Xraw), sy.fit_transform(yraw)

X_seq, y_seq = [], []
for i in range(WINDOW, len(Xs)):
    X_seq.append(Xs[i-WINDOW:i])
    y_seq.append(ys[i])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

Xtr, Xte, ytr, yte = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)


# ============================================================
# 6. MODEL FACTORY
# ============================================================
def build_model(model_type):
    model = Sequential()

    if model_type == "LSTM":
        model.add(LSTM(50, return_sequences=True,
                       input_shape=(WINDOW, X_seq.shape[2])))
        model.add(Dropout(0.1))
        model.add(LSTM(50))

    elif model_type == "BiLSTM":
        model.add(Bidirectional(
            LSTM(50, return_sequences=True),
            input_shape=(WINDOW, X_seq.shape[2])
        ))
        model.add(Dropout(0.1))
        model.add(Bidirectional(LSTM(50)))

    elif model_type == "GRU":
        model.add(GRU(50, return_sequences=True,
                      input_shape=(WINDOW, X_seq.shape[2])))
        model.add(Dropout(0.1))
        model.add(GRU(50))

    elif model_type == "BiGRU":
        model.add(Bidirectional(
            GRU(50, return_sequences=True),
            input_shape=(WINDOW, X_seq.shape[2])
        ))
        model.add(Dropout(0.1))
        model.add(Bidirectional(GRU(50)))

    model.add(Dropout(0.1))
    model.add(Dense(N_OUT))
    model.add(Activation("sigmoid"))
    model.compile(optimizer="adam", loss="mse")
    return model


# ============================================================
# 7. MODEL TRAINING AND EVALUATION
# ============================================================
import tensorflow as tf

MODEL_LIST = ["LSTM", "BiLSTM", "GRU", "BiGRU"]
metrics_rows = []

for MODEL_NAME in MODEL_LIST:
    tf.keras.backend.clear_session()
    model = build_model(MODEL_NAME)

    # Train model silently
    history = model.fit(
        Xtr, ytr,
        validation_split=0.2,
        epochs=300,
        batch_size=100,
        verbose=0
    )

    # Predictions
    yp_train = sy.inverse_transform(model.predict(Xtr, verbose=0))
    yp_test  = sy.inverse_transform(model.predict(Xte, verbose=0))

    ytrain_u = sy.inverse_transform(ytr)
    ytest_u  = sy.inverse_transform(yte)

    # Calculate performance metrics
    for phase, yt, yp in [
        ("TRAIN", ytrain_u, yp_train),
        ("TEST",  ytest_u,  yp_test)
    ]:
        for i, name in enumerate(OUTPUTS):
            rmse = np.sqrt(mean_squared_error(yt[:, i], yp[:, i]))
            mae  = mean_absolute_error(yt[:, i], yp[:, i])
            r2   = r2_score(yt[:, i], yp[:, i])

            metrics_rows.append({
                "model": MODEL_NAME,
                "phase": phase,
                "output": name,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            })

    # Save model and scalers
    model.save(os.path.join(
        MODEL_DIR, f"{MODEL_NAME.lower()}_bioreactor_6outputs.h5"
    ))
    joblib.dump(
        sx, os.path.join(MODEL_DIR, f"sx_{MODEL_NAME.lower()}_6outputs.pkl")
    )
    joblib.dump(
        sy, os.path.join(MODEL_DIR, f"sy_{MODEL_NAME.lower()}_6outputs.pkl")
    )


# ============================================================
# 9. FREE-RUN MULTI-STEP FORECASTING
# ============================================================
def free_run_forecast(model, X0, Fi, Fag, sx, sy):
    Xw = X0.copy()
    Yp = []

    for k in range(len(Fi)):
        y = sy.inverse_transform(
            model.predict(Xw[np.newaxis], verbose=0)
        )[0]
        Yp.append(y)
        nxt = sx.transform(
            np.hstack([Fi[k], Fag[k], y]).reshape(1, -1)
        )
        Xw = np.vstack([Xw[1:], nxt])

    return np.array(Yp)

# Generate forecasting dataset
Nf = 4500
Fi_f  = random_step_signal(Nf, 0, 100, 140, 220, seed=101)
Fag_f = random_step_signal(Nf, 0, 100, 130, 210, seed=202)

Xf = np.zeros((Nf + 1, 7))
Xf[0] = x0

for k in range(Nf):
    sol = solve_ivp(
        bioreactor_ode,
        [k, k + 1],
        Xf[k],
        args=(Fi_f[k], Fag_f[k]),
        method="BDF"
    )
    Xf[k + 1] = sol.y[:, -1]

Y_true = Xf[WINDOW + 1:, 1:]
Xt = np.column_stack([Fi_f, Fag_f, Xf[:-1, 1:]])
Xt_s = sx.transform(Xt)
X0 = Xt_s[:WINDOW]

Fi_aligned  = Fi_f[WINDOW:]
Fag_aligned = Fag_f[WINDOW:]

free_run_metrics = []

for MODEL_NAME in MODEL_LIST:
    model = tf.keras.models.load_model(
        os.path.join(
            MODEL_DIR,
            f"{MODEL_NAME.lower()}_bioreactor_6outputs.h5"
        ),
        compile=False
    )

    Yp = free_run_forecast(
        model,
        X0,
        Fi_f[WINDOW:],
        Fag_f[WINDOW:],
        sx,
        sy
    )

    for i, name in enumerate(OUTPUTS):
        rmse = np.sqrt(mean_squared_error(Y_true[:, i], Yp[:, i]))
        mae  = mean_absolute_error(Y_true[:, i], Yp[:, i])
        r2   = r2_score(Y_true[:, i], Yp[:, i])

        free_run_metrics.append({
            "model": MODEL_NAME,
            "output": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })


# ============================================================
# 10. NOISE ROBUSTNESS ANALYSIS
# ============================================================
noise_levels = [0.01, 0.03, 0.05, 0.06]
n_mc = 5
noise_rows = []

for MODEL_NAME in MODEL_LIST:
    # Forecast internally
    model = tf.keras.models.load_model(
        os.path.join(
            MODEL_DIR,
            f"{MODEL_NAME.lower()}_bioreactor_6outputs.h5"
        ),
        compile=False
    )
    Yp = free_run_forecast(
        model,
        X0,
        Fi_f[WINDOW:],
        Fag_f[WINDOW:],
        sx,
        sy
    )

    for nl in noise_levels:
        rmse_mc = []
        mae_mc  = []
        drift_mc = []

        for run in range(n_mc):
            std_signal = np.std(Y_true, axis=0)
            noise_std = nl * std_signal
            noise = np.random.randn(*Y_true.shape) * noise_std
            Y_noisy = Y_true + noise
            e = np.sqrt(np.sum((Y_noisy - Yp)**2, axis=1))

            rmse = np.sqrt(np.mean(e**2))
            mae  = mean_absolute_error(Y_noisy, Yp)
            final_drift = e[-1]

            rmse_mc.append(rmse)
            mae_mc.append(mae)
            drift_mc.append(final_drift)

        noise_rows.append({
            "model": MODEL_NAME,
            "noise_level": nl,
            "RMSE": np.mean(rmse_mc),
            "MAE": np.mean(mae_mc),
            "Final_Drift": np.mean(drift_mc)
        })


# ============================================================
# 11. RECURSIVE NOISY FREE-RUN FORECASTING
# ============================================================
def free_run_forecast_noisy(model, X0, Fi, Fag, sx, sy, noise_percent):
    Xw = X0.copy()
    Yp = []
    std_signal = np.std(Y_true, axis=0)
    noise_std = noise_percent * std_signal

    for k in range(len(Fi)):
        y = sy.inverse_transform(
            model.predict(Xw[np.newaxis], verbose=0)
        )[0]

        noise = np.random.randn(len(y)) * noise_std
        y_noisy = y + noise
        Yp.append(y_noisy)

        nxt = sx.transform(
            np.hstack([Fi[k], Fag[k], y_noisy]).reshape(1, -1)
        )
        Xw = np.vstack([Xw[1:], nxt])

    return np.array(Yp)

noise_levels = [0.01, 0.03, 0.05, 0.06]
n_mc = 5
recursive_noise_results = []

for MODEL_NAME in MODEL_LIST:
    model = tf.keras.models.load_model(
        os.path.join(
            MODEL_DIR,
            f"{MODEL_NAME.lower()}_bioreactor_6outputs.h5"
        ),
        compile=False
    )

    for nl in noise_levels:
        rmse_mc = []
        mae_mc = []
        drift_mc = []

        for run in range(n_mc):
            Yp_noisy = free_run_forecast_noisy(
                model,
                X0,
                Fi_f[WINDOW:],
                Fag_f[WINDOW:],
                sx,
                sy,
                nl
            )
            e = np.sqrt(np.sum((Y_true - Yp_noisy)**2, axis=1))

            rmse_mc.append(np.sqrt(np.mean(e**2)))
            mae_mc.append(np.mean(e))
            drift_mc.append(e[-1])

        recursive_noise_results.append({
            "model": MODEL_NAME,
            "noise_level": nl,
            "RMSE": np.mean(rmse_mc),
            "MAE": np.mean(mae_mc),
            "Final_Drift": np.mean(drift_mc)
        })
