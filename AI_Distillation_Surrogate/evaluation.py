"""
Evaluate saved surrogate model for Binary Distillation
Author: FOSSEE AI/ML Intern (Evaluation Only Script)
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===================================================
# 1. Load dataset and saved model/scaler
# ===================================================
print("Loading dataset and model...")

# Load dataset generated earlier
df = pd.read_csv("distill_data.csv")

# Load saved model and scaler (from main script)
model = joblib.load("best_model_random_forest.pkl")   # Change filename if best model was different
scaler = joblib.load("scaler.pkl")

# ===================================================
# 2. Prepare features/targets
# ===================================================
feature_cols = ['R', 'B', 'x_F', 'F', 'N', 'q']
target_cols = ['x_D', 'Q_R']

X = df[feature_cols].copy()
y = df[target_cols].copy()

# One-hot encode categorical variables (N, q)
X = pd.get_dummies(X, columns=['N', 'q'], prefix=['N', 'q'])

# Ensure columns match training features
# (in case some categories were missing in this dataset slice)
expected_features = scaler.feature_names_in_
for col in expected_features:
    if col not in X.columns:
        X[col] = 0
X = X[expected_features]

# Scale features
X_scaled = scaler.transform(X)

# ===================================================
# 3. Predictions & Metrics
# ===================================================
print("\nEvaluating model...")
y_pred = model.predict(X_scaled)

metrics = {}
for i, target in enumerate(target_cols):
    mae = mean_absolute_error(y.iloc[:, i], y_pred[:, i])
    rmse = mean_squared_error(y.iloc[:, i], y_pred[:, i], squared=False)
    r2 = r2_score(y.iloc[:, i], y_pred[:, i])
    metrics[target] = {"MAE": mae, "RMSE": rmse, "R²": r2}
    print(f"{target}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

# ===================================================
# 4. Parity Plots
# ===================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, target in enumerate(target_cols):
    axes[i].scatter(y.iloc[:, i], y_pred[:, i], alpha=0.6)
    axes[i].plot([y.iloc[:, i].min(), y.iloc[:, i].max()],
                 [y.iloc[:, i].min(), y.iloc[:, i].max()], 'r--')
    axes[i].set_xlabel(f"True {target}")
    axes[i].set_ylabel(f"Predicted {target}")
    axes[i].set_title(f"Parity Plot - {target}")
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("evaluation_parity_plots.png", dpi=300, bbox_inches='tight')
plt.show()
print("\nEvaluation plots saved as 'evaluation_parity_plots.png'")