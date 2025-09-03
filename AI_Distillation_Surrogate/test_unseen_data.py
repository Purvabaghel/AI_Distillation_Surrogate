"""
Test surrogate model on NEW unseen distillation data
and compare predictions vs true (simulated) values
Author: FOSSEE AI/ML Intern
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import glob
from main_distillation_surrogate import DistillationDataGenerator  # reuse simulator
from sklearn.metrics import mean_absolute_error, r2_score

# ===================================================
# 1. Load best saved model and scaler automatically
# ===================================================
print("Searching for best saved model...")

model_files = glob.glob("best_model_*.pkl")
if not model_files:
    raise FileNotFoundError("No best_model_*.pkl file found. Run training first.")

best_model_file = model_files[0]  # pick the first (only one should exist)
print(f"Loading model from {best_model_file}")

model = joblib.load(best_model_file)
scaler = joblib.load("scaler.pkl")

# Try to load polynomial features if they exist (for polynomial regression models)
try:
    poly_features = joblib.load("poly_features.pkl")
    use_poly = True
    print("Polynomial features loaded.")
except:
    poly_features = None
    use_poly = False

# ===================================================
# 2. Create NEW unseen operating conditions
# ===================================================
print("Generating unseen test cases...")

unseen_cases = pd.DataFrame([
    {"R": 4.8, "B": 2.5, "x_F": 0.30, "F": 110, "N": 25, "q": 1.2},
    {"R": 1.2, "B": 0.7, "x_F": 0.85, "F": 95,  "N": 15, "q": 0.8},
    {"R": 3.6, "B": 1.8, "x_F": 0.50, "F": 105, "N": 20, "q": 1.0},
    {"R": 5.0, "B": 3.0, "x_F": 0.95, "F": 120, "N": 25, "q": 1.0},
    {"R": 0.9, "B": 0.6, "x_F": 0.25, "F": 85,  "N": 15, "q": 0.8},
])

print("\nUnseen test inputs:")
print(unseen_cases)

# ===================================================
# 3. Get TRUE outputs from simulator
# ===================================================
generator = DistillationDataGenerator()
true_xD, true_QR = [], []

for _, row in unseen_cases.iterrows():
    x_D, Q_R = generator.simulate_column(
        row["R"], row["B"], row["x_F"], row["F"], row["N"], row["q"]
    )
    true_xD.append(x_D)
    true_QR.append(Q_R)

unseen_cases["True_x_D"] = true_xD
unseen_cases["True_Q_R"] = true_QR

# ===================================================
# 4. Prepare inputs for ML model
# ===================================================
X_unseen = pd.get_dummies(unseen_cases[["R", "B", "x_F", "F", "N", "q"]],
                          columns=['N', 'q'], prefix=['N', 'q'])

# Ensure columns match training
expected_features = scaler.feature_names_in_
for col in expected_features:
    if col not in X_unseen.columns:
        X_unseen[col] = 0
X_unseen = X_unseen[expected_features]

# Scale features
X_unseen_scaled = scaler.transform(X_unseen)

# If polynomial model was used, transform features
if use_poly:
    X_unseen_scaled = poly_features.transform(X_unseen_scaled)

# ===================================================
# 5. Predict with surrogate model
# ===================================================
predictions = model.predict(X_unseen_scaled)

unseen_cases["Pred_x_D"] = predictions[:, 0]
unseen_cases["Pred_Q_R"] = predictions[:, 1]

print("\nPredictions vs True values:")
print(unseen_cases[["R", "B", "x_F", "F", "N", "q",
                    "True_x_D", "Pred_x_D", "True_Q_R", "Pred_Q_R"]])

# ===================================================
# 6. Evaluate metrics
# ===================================================
mae_xD = mean_absolute_error(unseen_cases["True_x_D"], unseen_cases["Pred_x_D"])
mae_QR = mean_absolute_error(unseen_cases["True_Q_R"], unseen_cases["Pred_Q_R"])
r2_xD = r2_score(unseen_cases["True_x_D"], unseen_cases["Pred_x_D"])
r2_QR = r2_score(unseen_cases["True_Q_R"], unseen_cases["Pred_Q_R"])

print(f"\nUnseen Test Metrics:")
print(f"x_D: MAE={mae_xD:.4f}, R²={r2_xD:.4f}")
print(f"Q_R: MAE={mae_QR:.2f}, R²={r2_QR:.4f}")

# ===================================================
# 7. Visualization
# ===================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Parity plot for x_D
axes[0].scatter(unseen_cases["True_x_D"], unseen_cases["Pred_x_D"], color="blue", alpha=0.7)
axes[0].plot([0, 1], [0, 1], "r--")
axes[0].set_xlabel("True x_D")
axes[0].set_ylabel("Predicted x_D")
axes[0].set_title(f"x_D Predictions vs True (R²={r2_xD:.3f})")
axes[0].grid(True, alpha=0.3)

# Parity plot for Q_R
axes[1].scatter(unseen_cases["True_Q_R"], unseen_cases["Pred_Q_R"], color="green", alpha=0.7)
min_qr, max_qr = unseen_cases[["True_Q_R", "Pred_Q_R"]].min().min(), unseen_cases[["True_Q_R", "Pred_Q_R"]].max().max()
axes[1].plot([min_qr, max_qr], [min_qr, max_qr], "r--")
axes[1].set_xlabel("True Q_R (kW)")
axes[1].set_ylabel("Predicted Q_R (kW)")
axes[1].set_title(f"Q_R Predictions vs True (R²={r2_QR:.3f})")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("unseen_vs_true.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nComparison plot saved as 'unseen_vs_true.png'")