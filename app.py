"""
Full Surrogate Website (Single Script)
Binary Distillation — Ethanol/Water
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# ===============================
# 1. Load best model + scaler
# ===============================
model_files = glob.glob("best_model_*.pkl")
if not model_files:
    st.error("No trained model found. Please run training first.")
    st.stop()

best_model_file = model_files[0]
st.sidebar.success(f"Loaded model: {best_model_file}")

model = joblib.load(best_model_file)
scaler = joblib.load("scaler.pkl")

try:
    poly_features = joblib.load("poly_features.pkl")
    use_poly = True
except:
    poly_features = None
    use_poly = False

# ===============================
# 2. Distillation Simulator (Truth Generator)
# ===============================
class DistillationDataGenerator:
    def _init_(self):
        self.pressure = 101.325

    def antoine_ethanol(self, T):
        A, B, C = 8.20417, 1642.89, 230.300
        return 10 ** (A - B / (T + C))

    def antoine_water(self, T):
        A, B, C = 8.07131, 1730.63, 233.426
        return 10 ** (A - B / (T + C))

    def bubble_point(self, x_eth, P=101.325):
        T = 80.0
        for _ in range(50):
            P_eth = self.antoine_ethanol(T)
            P_water = self.antoine_water(T)
            f = x_eth * P_eth + (1 - x_eth) * P_water - P
            dP_eth_dT = P_eth * 1642.89 * np.log(10) / (T + 230.300) ** 2
            dP_water_dT = P_water * 1730.63 * np.log(10) / (T + 233.426) ** 2
            df_dT = x_eth * dP_eth_dT + (1 - x_eth) * dP_water_dT
            T_new = T - f / df_dT
            if abs(T_new - T) < 0.001:
                break
            T = T_new
        return T

    def relative_volatility(self, x_eth):
        T = self.bubble_point(x_eth)
        P_eth = self.antoine_ethanol(T)
        P_water = self.antoine_water(T)
        return P_eth / P_water

    def simulate_column(self, R, B, x_F, F, N, q):
        if R < 0.1 or B < 0.1 or x_F <= 0 or x_F >= 1:
            return None, None
        recovery = min(0.95, 0.5 + 0.3 * R / (R + 1))
        D = F * x_F * recovery / max(0.01, x_F)
        W = F - D
        if D <= 0 or W <= 0:
            return None, None
        alpha_avg = self.relative_volatility((x_F + 0.95) / 2)
        x_W = max(0.001, (F * x_F - D * 0.95) / W)
        N_min = np.log((0.95 / (1 - 0.95)) * ((1 - x_W) / x_W)) / np.log(alpha_avg)
        N_eff = N * 0.7
        separation_factor = min(1.0, (R * N_eff) / (R + 1) / max(N_min, 1))
        x_D = x_F + (0.95 - x_F) * separation_factor
        x_D = max(x_F, min(0.999, x_D))
        noise_factor = 1 + 0.02 * (np.random.random() - 0.5)
        x_D *= noise_factor
        x_D = max(x_F, min(0.999, x_D))
        lambda_mix = 40000 - 5000 * x_F
        V_bottom = W * (1 + B)
        Q_R = V_bottom * lambda_mix / 3600
        Q_R *= (1 + R * 0.1) * (1 + 0.02 * (N - 20))
        Q_R = max(Q_R * 0.5, Q_R)
        return x_D, Q_R

generator = DistillationDataGenerator()

# ===============================
# 3. UI
# ===============================
st.title("🌡 Binary Distillation Surrogate Model")
st.write("Predict *Distillate Purity (x_D)* and *Reboiler Duty (Q_R)*")

# Input form
st.sidebar.header("Input Parameters")
R = st.sidebar.slider("Reflux Ratio (R)", 0.8, 5.0, 2.5, step=0.1)
B = st.sidebar.slider("Boilup Ratio (B)", 0.5, 3.0, 1.5, step=0.1)
x_F = st.sidebar.slider("Feed Mole Fraction (x_F)", 0.2, 0.95, 0.5, step=0.01)
F = st.sidebar.slider("Feed Flowrate (F)", 70, 130, 100, step=1)
N = st.sidebar.selectbox("Number of Stages (N)", [15, 20, 25])
q = st.sidebar.selectbox("Feed Condition (q)", [0.8, 1.0, 1.2])

input_df = pd.DataFrame([{"R": R, "B": B, "x_F": x_F, "F": F, "N": N, "q": q}])

# Preprocess
X = pd.get_dummies(input_df, columns=["N", "q"], prefix=["N", "q"])
expected_features = scaler.feature_names_in_
for col in expected_features:
    if col not in X.columns:
        X[col] = 0
X = X[expected_features]
X_scaled = scaler.transform(X)
if use_poly:
    X_scaled = poly_features.transform(X_scaled)

# Prediction
pred = model.predict(X_scaled)
pred_xD, pred_QR = pred[0]

# True values
true_xD, true_QR = generator.simulate_column(R, B, x_F, F, N, q)

# Display
st.subheader("🔮 Results")
st.write(f"*Predicted Distillate Purity (x_D):* {pred_xD:.4f}")
st.write(f"*Predicted Reboiler Duty (Q_R):* {pred_QR:.2f} kW")
if true_xD and true_QR:
    st.write(f"*True x_D (simulated):* {true_xD:.4f}")
    st.write(f"*True Q_R (simulated):* {true_QR:.2f} kW")

# Metrics (for single point, not meaningful, but useful if extended to batch)
if true_xD and true_QR:
    mae_xD = mean_absolute_error([true_xD], [pred_xD])
    mae_QR = mean_absolute_error([true_QR], [pred_QR])
    st.write(f"Errors → x_D MAE={mae_xD:.4f}, Q_R MAE={mae_QR:.2f}")

# ===============================
# 4. Batch CSV upload
# ===============================
st.header("📂 Batch Predictions (Upload CSV)")
uploaded_file = st.file_uploader("Upload CSV with columns: R,B,x_F,F,N,q", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    X = pd.get_dummies(df[["R", "B", "x_F", "F", "N", "q"]], columns=["N", "q"], prefix=["N", "q"])
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_features]
    X_scaled = scaler.transform(X)
    if use_poly:
        X_scaled = poly_features.transform(X_scaled)
    preds = model.predict(X_scaled)
    df["Pred_x_D"] = preds[:,0]
    df["Pred_Q_R"] = preds[:,1]
    st.dataframe(df.head())
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")