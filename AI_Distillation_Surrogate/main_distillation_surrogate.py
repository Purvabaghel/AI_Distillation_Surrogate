
"""
AI/ML Surrogate Modeling for Binary Distillation Column
Ethanol-Water System at 1 atm

This script implements multiple ML models to predict:
- Distillate purity (x_D): mole fraction of ethanol in distillate
- Reboiler duty (Q_R): energy requirement in kW

Author: PURVA BAGHEL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DistillationDataGenerator:
    """
    Simulates binary distillation column data for Ethanol-Water system
    Based on McCabe-Thiele method and Riedel equation for vapor pressure
    """
    
    def __init__(self):
        self.system = "Ethanol-Water"
        self.pressure = 101.325  # kPa (1 atm)
        
    def antoine_ethanol(self, T):
        """Antoine equation for ethanol vapor pressure (kPa)"""
        # Constants for ethanol (T in Celsius)
        A, B, C = 8.20417, 1642.89, 230.300
        return 10**(A - B/(T + C))
    
    def antoine_water(self, T):
        """Antoine equation for water vapor pressure (kPa)"""
        # Constants for water (T in Celsius)
        A, B, C = 8.07131, 1730.63, 233.426
        return 10**(A - B/(T + C))
    
    def bubble_point(self, x_eth, P=101.325):
        """Calculate bubble point temperature for ethanol-water mixture"""
        # Initial guess
        T = 80.0  # Celsius
        for _ in range(50):  # Newton-Raphson iterations
            P_eth = self.antoine_ethanol(T)
            P_water = self.antoine_water(T)
            
            f = x_eth * P_eth + (1 - x_eth) * P_water - P
            
            # Derivatives for Newton-Raphson
            dP_eth_dT = P_eth * 1642.89 * np.log(10) / (T + 230.300)**2
            dP_water_dT = P_water * 1730.63 * np.log(10) / (T + 233.426)**2
            
            df_dT = x_eth * dP_eth_dT + (1 - x_eth) * dP_water_dT
            
            T_new = T - f / df_dT
            if abs(T_new - T) < 0.001:
                break
            T = T_new
            
        return T
    
    def relative_volatility(self, x_eth):
        """Calculate relative volatility at bubble point"""
        T = self.bubble_point(x_eth)
        P_eth = self.antoine_ethanol(T)
        P_water = self.antoine_water(T)
        return P_eth / P_water
    
    def simulate_column(self, R, B, x_F, F, N, q):
        """
        Simulate binary distillation column using simplified correlations
        
        Parameters:
        R: Reflux ratio
        B: Boilup ratio  
        x_F: Feed mole fraction of ethanol
        F: Feed flowrate (kmol/h)
        N: Number of stages
        q: Feed thermal condition (1.0 for saturated liquid)
        
        Returns:
        x_D: Distillate purity (mole fraction ethanol)
        Q_R: Reboiler duty (kW)
        """
        
        # Check for invalid inputs
        if R < 0.1 or B < 0.1 or x_F <= 0 or x_F >= 1:
            return None, None
            
        # Estimate distillate flowrate (D) and bottoms flowrate (W)
        # Using overall material balance and assuming reasonable split
        recovery = min(0.95, 0.5 + 0.3 * R / (R + 1))  # Higher R -> higher recovery
        D = F * x_F * recovery / max(0.01, x_F)  # Prevent division by zero
        W = F - D
        
        if D <= 0 or W <= 0:
            return None, None
        
        # Estimate distillate composition using Fenske equation for minimum stages
        alpha_avg = self.relative_volatility((x_F + 0.95) / 2)  # Average volatility
        
        # Modified Fenske equation accounting for finite stages and reflux
        x_W = max(0.001, (F * x_F - D * 0.95) / W)  # Estimated bottoms composition
        
        N_min = np.log((0.95 / (1 - 0.95)) * ((1 - x_W) / x_W)) / np.log(alpha_avg)
        N_eff = N * 0.7  # Stage efficiency
        
        # Actual distillate purity based on stages and reflux
        separation_factor = min(1.0, (R * N_eff) / (R + 1) / max(N_min, 1))
        
        x_D = x_F + (0.95 - x_F) * separation_factor
        x_D = max(x_F, min(0.999, x_D))  # Physical bounds
        
        # Add some realistic noise and non-linearity
        noise_factor = 1 + 0.02 * (np.random.random() - 0.5)
        x_D *= noise_factor
        x_D = max(x_F, min(0.999, x_D))
        
        # Reboiler duty calculation
        # Latent heat of vaporization (kJ/kmol) - approximate for ethanol-water
        lambda_mix = 40000 - 5000 * x_F  # Decreases with ethanol content
        
        V_bottom = W * (1 + B)  # Vapor from reboiler
        Q_R = V_bottom * lambda_mix / 3600  # Convert to kW
        
        # Add dependency on reflux and stages
        Q_R *= (1 + R * 0.1) * (1 + 0.02 * (N - 20))
        Q_R = max(Q_R * 0.5, Q_R)  # Ensure positive
        
        return x_D, Q_R
    
    def generate_dataset(self, n_samples=500):
        """Generate synthetic distillation data"""
        print("Generating synthetic distillation data...")
        
        data = []
        failed_runs = 0
        
        # Define parameter ranges
        R_range = (0.8, 5.0)
        B_range = (0.5, 3.0)
        xF_range = (0.2, 0.95)
        F_range = (80, 120)  # ±30% around 100 kmol/h
        N_options = [15, 20, 25]
        q_options = [0.8, 1.0, 1.2]  # subcooled, saturated, superheated
        
        target_samples = n_samples
        attempts = 0
        max_attempts = target_samples * 3
        
        while len(data) < target_samples and attempts < max_attempts:
            attempts += 1
            
            # Random sampling of parameters
            R = np.random.uniform(*R_range)
            B = np.random.uniform(*B_range)
            x_F = np.random.uniform(*xF_range)
            F = np.random.uniform(*F_range)
            N = np.random.choice(N_options)
            q = np.random.choice(q_options)
            
            # Simulate column
            x_D, Q_R = self.simulate_column(R, B, x_F, F, N, q)
            
            if x_D is not None and Q_R is not None:
                data.append({
                    'R': R,
                    'B': B, 
                    'x_F': x_F,
                    'F': F,
                    'N': N,
                    'q': q,
                    'x_D': x_D,
                    'Q_R': Q_R
                })
            else:
                failed_runs += 1
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} valid samples")
        print(f"Failed runs: {failed_runs}")
        
        # Add some additional structured data points for better coverage
        self._add_structured_points(df, R_range, B_range, xF_range, F_range, N_options, q_options)
        
        return df
    
    def _add_structured_points(self, df, R_range, B_range, xF_range, F_range, N_options, q_options):
        """Add structured grid points for better parameter space coverage"""
        print("Adding structured grid points...")
        
        # Create a smaller grid for systematic coverage
        R_grid = np.linspace(R_range[0], R_range[1], 5)
        xF_grid = np.linspace(xF_range[0], xF_range[1], 5)
        
        structured_data = []
        for R in R_grid:
            for x_F in xF_grid:
                B = np.random.uniform(*B_range)
                F = np.random.uniform(*F_range)
                N = np.random.choice(N_options)
                q = np.random.choice(q_options)
                
                x_D, Q_R = self.simulate_column(R, B, x_F, F, N, q)
                if x_D is not None and Q_R is not None:
                    structured_data.append({
                        'R': R, 'B': B, 'x_F': x_F, 'F': F, 'N': N, 'q': q,
                        'x_D': x_D, 'Q_R': Q_R
                    })
        
        # Append to existing dataframe
        structured_df = pd.DataFrame(structured_data)
        df = pd.concat([df, structured_df], ignore_index=True)
        print(f"Added {len(structured_df)} structured points. Total: {len(df)}")
        
        return df


class DistillationSurrogateModels:
    """
    Implements and compares multiple ML models for distillation surrogate modeling
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.poly_features = None
        self.results = {}
        
    def prepare_data(self, df, test_size=0.2, val_size=0.2):
        """Prepare data for ML modeling"""
        print("Preparing data for ML modeling...")
        
        # Features and targets
        feature_cols = ['R', 'B', 'x_F', 'F', 'N', 'q']
        target_cols = ['x_D', 'Q_R']
        
        X = df[feature_cols].copy()
        y = df[target_cols].copy()
        
        # One-hot encode discrete variables
        X = pd.get_dummies(X, columns=['N', 'q'], prefix=['N', 'q'])
        
        # Split data - use stratified approach for better generalization testing
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_names = X.columns.tolist()
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples") 
        print(f"Test set: {X_test.shape[0]} samples")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train.values, y_val.values, y_test.values,
                X_train, X_val, X_test)
    
    def train_polynomial_regression(self, X_train, y_train, X_val, y_val):
        """Train polynomial regression model"""
        print("Training Polynomial Regression...")
        
        # Try different polynomial degrees
        best_score = -np.inf
        best_degree = 2
        
        for degree in [2, 3]:
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly_features.fit_transform(X_train)
            X_val_poly = poly_features.transform(X_val)
            
            model = MultiOutputRegressor(LinearRegression())
            model.fit(X_train_poly, y_train)
            
            score = model.score(X_val_poly, y_val)
            if score > best_score:
                best_score = score
                best_degree = degree
                best_model = model
                best_poly_features = poly_features
        
        self.models['polynomial'] = best_model
        self.poly_features = best_poly_features
        print(f"Best polynomial degree: {best_degree}, R²: {best_score:.4f}")
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        param_grid = {
            'estimator__n_estimators': [100, 200],
            'estimator__max_depth': [10, 15, None],
            'estimator__min_samples_split': [2, 5],
            'estimator__min_samples_leaf': [1, 2]
        }
        
        rf_base = RandomForestRegressor(random_state=42)
        rf_multi = MultiOutputRegressor(rf_base)
        
        # Use a smaller parameter grid for faster training
        simplified_grid = {
            'estimator__n_estimators': [100, 200],
            'estimator__max_depth': [10, None]
        }
        
        rf_grid = GridSearchCV(rf_multi, simplified_grid, cv=3, scoring='r2', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        
        self.models['random_forest'] = rf_grid.best_estimator_
        print(f"Best RF params: {rf_grid.best_params_}")
        print(f"Best RF CV score: {rf_grid.best_score_:.4f}")
    
    def train_svm(self, X_train, y_train, X_val, y_val):
        """Train Support Vector Machine model"""
        print("Training SVM...")
        
        param_grid = {
            'estimator__C': [0.1, 1, 10],
            'estimator__gamma': ['scale', 'auto'],
            'estimator__kernel': ['rbf', 'poly']
        }
        
        # Use subset for SVM training (it's slow on large datasets)
        if X_train.shape[0] > 1000:
            idx = np.random.choice(X_train.shape[0], 1000, replace=False)
            X_train_svm = X_train[idx]
            y_train_svm = y_train[idx]
        else:
            X_train_svm = X_train
            y_train_svm = y_train
        
        svm_base = SVR()
        svm_multi = MultiOutputRegressor(svm_base)
        
        # Simplified grid for faster training
        simplified_grid = {
            'estimator__C': [1, 10],
            'estimator__kernel': ['rbf']
        }
        
        svm_grid = GridSearchCV(svm_multi, simplified_grid, cv=3, scoring='r2')
        svm_grid.fit(X_train_svm, y_train_svm)
        
        self.models['svm'] = svm_grid.best_estimator_
        print(f"Best SVM params: {svm_grid.best_params_}")
        print(f"Best SVM CV score: {svm_grid.best_score_:.4f}")
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train Neural Network model"""
        print("Training Neural Network...")
        
        param_grid = {
            'hidden_layer_sizes': [(64, 32), (128, 64), (100, 50, 25)],
            'alpha': [0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01]
        }
        
        # Simplified grid for faster training
        simplified_grid = {
            'hidden_layer_sizes': [(64, 32), (128, 64)],
            'alpha': [0.001, 0.01]
        }
        
        mlp = MLPRegressor(max_iter=500, random_state=42, early_stopping=True)
        mlp_grid = GridSearchCV(mlp, simplified_grid, cv=3, scoring='r2')
        
        # Multi-output wrapper
        mlp_multi = MultiOutputRegressor(mlp_grid)
        mlp_multi.fit(X_train, y_train)
        
        self.models['neural_network'] = mlp_multi
        print("Neural Network training completed")
    
    def evaluate_models(self, X_test, y_test, X_val, y_val):
        """Evaluate all trained models"""
        print("\nEvaluating models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Prepare test data based on model type
            if name == 'polynomial':
                X_test_processed = self.poly_features.transform(X_test)
                X_val_processed = self.poly_features.transform(X_val)
            else:
                X_test_processed = X_test
                X_val_processed = X_val
            
            # Predictions
            y_test_pred = model.predict(X_test_processed)
            y_val_pred = model.predict(X_val_processed)
            
            # Metrics for validation set
            val_metrics = {}
            for i, target in enumerate(['x_D', 'Q_R']):
                val_metrics[f'{target}_mae'] = mean_absolute_error(y_val[:, i], y_val_pred[:, i])
                val_metrics[f'{target}_rmse'] = np.sqrt(mean_squared_error(y_val[:, i], y_val_pred[:, i]))
                val_metrics[f'{target}_r2'] = r2_score(y_val[:, i], y_val_pred[:, i])
            
            # Metrics for test set
            test_metrics = {}
            for i, target in enumerate(['x_D', 'Q_R']):
                test_metrics[f'{target}_mae'] = mean_absolute_error(y_test[:, i], y_test_pred[:, i])
                test_metrics[f'{target}_rmse'] = np.sqrt(mean_squared_error(y_test[:, i], y_test_pred[:, i]))
                test_metrics[f'{target}_r2'] = r2_score(y_test[:, i], y_test_pred[:, i])
            
            results[name] = {
                'validation': val_metrics,
                'test': test_metrics,
                'val_predictions': y_val_pred,
                'test_predictions': y_test_pred
            }
            
            # Print key metrics
            print(f"Validation R² - x_D: {val_metrics['x_D_r2']:.4f}, Q_R: {val_metrics['Q_R_r2']:.4f}")
            print(f"Test R² - x_D: {test_metrics['x_D_r2']:.4f}, Q_R: {test_metrics['Q_R_r2']:.4f}")
        
        self.results = results
        return results
    
    def create_evaluation_plots(self, X_test, y_test, save_plots=True):
        """Create comprehensive evaluation plots"""
        print("Creating evaluation plots...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model Comparison - R² scores
        ax1 = plt.subplot(3, 4, 1)
        models = list(self.results.keys())
        x_D_r2 = [self.results[m]['test']['x_D_r2'] for m in models]
        Q_R_r2 = [self.results[m]['test']['Q_R_r2'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, x_D_r2, width, label='x_D', alpha=0.8)
        ax1.bar(x + width/2, Q_R_r2, width, label='Q_R', alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Comparison - R² Scores')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2-5. Parity Plots for best model (highest average R²)
        best_model_name = max(self.results.keys(), 
                             key=lambda x: (self.results[x]['test']['x_D_r2'] + 
                                          self.results[x]['test']['Q_R_r2']) / 2)
        
        best_results = self.results[best_model_name]
        
        # Parity plot for x_D
        ax2 = plt.subplot(3, 4, 2)
        y_true_xD = y_test[:, 0]
        y_pred_xD = best_results['test_predictions'][:, 0]
        
        ax2.scatter(y_true_xD, y_pred_xD, alpha=0.6, s=30)
        ax2.plot([y_true_xD.min(), y_true_xD.max()], [y_true_xD.min(), y_true_xD.max()], 'r--', lw=2)
        ax2.set_xlabel('Simulated x_D')
        ax2.set_ylabel('Predicted x_D')
        ax2.set_title(f'Parity Plot x_D - {best_model_name}\nR² = {best_results["test"]["x_D_r2"]:.4f}')
        ax2.grid(True, alpha=0.3)
        
        # Parity plot for Q_R
        ax3 = plt.subplot(3, 4, 3)
        y_true_QR = y_test[:, 1]
        y_pred_QR = best_results['test_predictions'][:, 1]
        
        ax3.scatter(y_true_QR, y_pred_QR, alpha=0.6, s=30, color='green')
        ax3.plot([y_true_QR.min(), y_true_QR.max()], [y_true_QR.min(), y_true_QR.max()], 'r--', lw=2)
        ax3.set_xlabel('Simulated Q_R (kW)')
        ax3.set_ylabel('Predicted Q_R (kW)')
        ax3.set_title(f'Parity Plot Q_R - {best_model_name}\nR² = {best_results["test"]["Q_R_r2"]:.4f}')
        ax3.grid(True, alpha=0.3)
        
        # 6. Residuals vs Predicted
        ax4 = plt.subplot(3, 4, 4)
        residuals_xD = y_true_xD - y_pred_xD
        ax4.scatter(y_pred_xD, residuals_xD, alpha=0.6, s=30)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Predicted x_D')
        ax4.set_ylabel('Residuals x_D')
        ax4.set_title('Residuals vs Predicted x_D')
        ax4.grid(True, alpha=0.3)
        
        # 7-10. Error distribution and metrics summary
        ax5 = plt.subplot(3, 4, 5)
        ax5.hist(residuals_xD, bins=30, alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Residuals x_D')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Distribution of Residuals x_D')
        ax5.grid(True, alpha=0.3)
        
        # Metrics comparison table as text
        ax6 = plt.subplot(3, 4, 6)
        ax6.axis('off')
        
        metrics_text = "Model Performance Summary\n" + "="*30 + "\n\n"
        for model_name in models:
            metrics = self.results[model_name]['test']
            metrics_text += f"{model_name.upper()}\n"
            metrics_text += f"x_D: MAE={metrics['x_D_mae']:.4f}, RMSE={metrics['x_D_rmse']:.4f}, R²={metrics['x_D_r2']:.4f}\n"
            metrics_text += f"Q_R: MAE={metrics['Q_R_mae']:.1f}, RMSE={metrics['Q_R_rmse']:.1f}, R²={metrics['Q_R_r2']:.4f}\n\n"
        
        ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
        
        # High purity region analysis
        high_purity_mask = y_true_xD >= 0.95
        if np.sum(high_purity_mask) > 10:
            ax7 = plt.subplot(3, 4, 7)
            ax7.scatter(y_true_xD[high_purity_mask], y_pred_xD[high_purity_mask], 
                       alpha=0.8, s=50, color='red')
            ax7.plot([0.95, 1.0], [0.95, 1.0], 'k--', lw=2)
            ax7.set_xlabel('Simulated x_D (High Purity)')
            ax7.set_ylabel('Predicted x_D (High Purity)')
            ax7.set_title('High Purity Region (x_D ≥ 0.95)')
            ax7.grid(True, alpha=0.3)
            
            # Calculate high purity metrics
            hp_r2 = r2_score(y_true_xD[high_purity_mask], y_pred_xD[high_purity_mask])
            hp_mae = mean_absolute_error(y_true_xD[high_purity_mask], y_pred_xD[high_purity_mask])
            ax7.text(0.05, 0.95, f'R² = {hp_r2:.4f}\nMAE = {hp_mae:.4f}', 
                    transform=ax7.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
            print("Evaluation plots saved as 'model_evaluation_plots.png'")
        
        plt.show()
    
    def physical_consistency_check(self, df_original, X_test_original, y_test):
        """Check physical consistency of the best model"""
        print("\nPerforming physical consistency checks...")
        
        # Get best model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: (self.results[x]['test']['x_D_r2'] + 
                                          self.results[x]['test']['Q_R_r2']) / 2)
        
        best_model = self.models[best_model_name]
        print(f"Using best model: {best_model_name}")
        
        # 1. Check bounds (0 ≤ x_D ≤ 1)
        if best_model_name == 'polynomial':
            X_test_processed = self.poly_features.transform(self.scaler.transform(X_test_original))
        else:
            X_test_processed = self.scaler.transform(X_test_original)
        
        predictions = best_model.predict(X_test_processed)
        x_D_pred = predictions[:, 0]
        
        violations = np.sum((x_D_pred < 0) | (x_D_pred > 1))
        print(f"Bound violations (x_D): {violations}/{len(x_D_pred)} ({100*violations/len(x_D_pred):.2f}%)")
        
        # 2. Monotonicity check: For fixed conditions, increasing R should increase x_D
        print("Checking monotonicity: x_D vs Reflux Ratio...")
        
        # Create test cases with varying R, fixed other parameters
        test_cases = []
        R_range = np.linspace(1.0, 4.0, 10)
        
        # Fixed conditions
        base_conditions = {
            'B': 1.5,
            'x_F': 0.5,
            'F': 100,
            'N': 20,
            'q': 1.0
        }
        
        for R in R_range:
            conditions = base_conditions.copy()
            conditions['R'] = R
            test_cases.append(conditions)
        
        test_df = pd.DataFrame(test_cases)
        test_df_encoded = pd.get_dummies(test_df, columns=['N', 'q'], prefix=['N', 'q'])
        
        # Ensure all columns match training data
        for col in self.feature_names:
            if col not in test_df_encoded.columns:
                test_df_encoded[col] = 0
        test_df_encoded = test_df_encoded[self.feature_names]
        
        if best_model_name == 'polynomial':
            X_mono_test = self.poly_features.transform(self.scaler.transform(test_df_encoded))
        else:
            X_mono_test = self.scaler.transform(test_df_encoded)
        
        mono_predictions = best_model.predict(X_mono_test)
        x_D_mono = mono_predictions[:, 0]
        
        # Check for non-monotonic behavior
        monotonic_violations = 0
        for i in range(1, len(x_D_mono)):
            if x_D_mono[i] < x_D_mono[i-1]:
                monotonic_violations += 1
        
        print(f"Monotonicity violations: {monotonic_violations}/{len(x_D_mono)-1}")
        
        # 3. Create sensitivity plots
        self._create_sensitivity_plots(best_model, best_model_name)
        
        return {
            'bound_violations': violations,
            'total_predictions': len(x_D_pred),
            'monotonic_violations': monotonic_violations,
            'monotonic_tests': len(x_D_mono)-1
        }
    
    def _create_sensitivity_plots(self, model, model_name):
        """Create partial dependence plots"""
        print("Creating sensitivity plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Base case for sensitivity analysis
        base_case = {
            'R': 2.0, 'B': 1.5, 'x_F': 0.5, 'F': 100, 'N': 20, 'q': 1.0
        }
        
        # 1. x_D vs R
        R_range = np.linspace(0.8, 5.0, 50)
        x_D_sensitivity_R = []
        
        for R in R_range:
            case = base_case.copy()
            case['R'] = R
            case_df = pd.DataFrame([case])
            case_encoded = pd.get_dummies(case_df, columns=['N', 'q'], prefix=['N', 'q'])
            
            for col in self.feature_names:
                if col not in case_encoded.columns:
                    case_encoded[col] = 0
            case_encoded = case_encoded[self.feature_names]
            
            if model_name == 'polynomial':
                X_case = self.poly_features.transform(self.scaler.transform(case_encoded))
            else:
                X_case = self.scaler.transform(case_encoded)
            
            pred = model.predict(X_case)
            x_D_sensitivity_R.append(pred[0, 0])
        
        axes[0,0].plot(R_range, x_D_sensitivity_R, 'b-', linewidth=2)
        axes[0,0].set_xlabel('Reflux Ratio (R)')
        axes[0,0].set_ylabel('Distillate Purity (x_D)')
        axes[0,0].set_title('Sensitivity: x_D vs Reflux Ratio')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. x_D vs x_F
        xF_range = np.linspace(0.2, 0.95, 50)
        x_D_sensitivity_xF = []
        
        for x_F in xF_range:
            case = base_case.copy()
            case['x_F'] = x_F
            case_df = pd.DataFrame([case])
            case_encoded = pd.get_dummies(case_df, columns=['N', 'q'], prefix=['N', 'q'])
            
            for col in self.feature_names:
                if col not in case_encoded.columns:
                    case_encoded[col] = 0
            case_encoded = case_encoded[self.feature_names]
            
            if model_name == 'polynomial':
                X_case = self.poly_features.transform(self.scaler.transform(case_encoded))
            else:
                X_case = self.scaler.transform(case_encoded)
            
            pred = model.predict(X_case)
            x_D_sensitivity_xF.append(pred[0, 0])
        
        axes[0,1].plot(xF_range, x_D_sensitivity_xF, 'g-', linewidth=2)
        axes[0,1].set_xlabel('Feed Composition (x_F)')
        axes[0,1].set_ylabel('Distillate Purity (x_D)')
        axes[0,1].set_title('Sensitivity: x_D vs Feed Composition')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Q_R vs R
        Q_R_sensitivity_R = []
        for R in R_range:
            case = base_case.copy()
            case['R'] = R
            case_df = pd.DataFrame([case])
            case_encoded = pd.get_dummies(case_df, columns=['N', 'q'], prefix=['N', 'q'])
            
            for col in self.feature_names:
                if col not in case_encoded.columns:
                    case_encoded[col] = 0
            case_encoded = case_encoded[self.feature_names]
            
            if model_name == 'polynomial':
                X_case = self.poly_features.transform(self.scaler.transform(case_encoded))
            else:
                X_case = self.scaler.transform(case_encoded)
            
            pred = model.predict(X_case)
            Q_R_sensitivity_R.append(pred[0, 1])
        
        axes[1,0].plot(R_range, Q_R_sensitivity_R, 'r-', linewidth=2)
        axes[1,0].set_xlabel('Reflux Ratio (R)')
        axes[1,0].set_ylabel('Reboiler Duty (Q_R)')
        axes[1,0].set_title('Sensitivity: Q_R vs Reflux Ratio')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Q_R vs B
        B_range = np.linspace(0.5, 3.0, 50)
        Q_R_sensitivity_B = []
        
        for B in B_range:
            case = base_case.copy()
            case['B'] = B
            case_df = pd.DataFrame([case])
            case_encoded = pd.get_dummies(case_df, columns=['N', 'q'], prefix=['N', 'q'])
            
            for col in self.feature_names:
                if col not in case_encoded.columns:
                    case_encoded[col] = 0
            case_encoded = case_encoded[self.feature_names]
            
            if model_name == 'polynomial':
                X_case = self.poly_features.transform(self.scaler.transform(case_encoded))
            else:
                X_case = self.scaler.transform(case_encoded)
            
            pred = model.predict(X_case)
            Q_R_sensitivity_B.append(pred[0, 1])
        
        axes[1,1].plot(B_range, Q_R_sensitivity_B, 'm-', linewidth=2)
        axes[1,1].set_xlabel('Boilup Ratio (B)')
        axes[1,1].set_ylabel('Reboiler Duty (Q_R)')
        axes[1,1].set_title('Sensitivity: Q_R vs Boilup Ratio')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sensitivity_plots.png', dpi=300, bbox_inches='tight')
        print("Sensitivity plots saved as 'sensitivity_plots.png'")
        plt.show()
    
    def generalization_test(self, df_original):
        """Test model generalization by training without a specific region"""
        print("\nPerforming generalization test...")
        
        # Hold out reflux ratio range [3.5, 4.5] for testing
        holdout_mask = (df_original['R'] >= 3.5) & (df_original['R'] <= 4.5)
        train_mask = ~holdout_mask
        
        print(f"Training on {np.sum(train_mask)} samples")
        print(f"Testing on {np.sum(holdout_mask)} samples (R ∈ [3.5, 4.5])")
        
        if np.sum(holdout_mask) < 10:
            print("Insufficient holdout samples for generalization test")
            return
        
        # Prepare data
        feature_cols = ['R', 'B', 'x_F', 'F', 'N', 'q']
        target_cols = ['x_D', 'Q_R']
        
        X_train_gen = df_original[train_mask][feature_cols].copy()
        y_train_gen = df_original[train_mask][target_cols].copy()
        X_test_gen = df_original[holdout_mask][feature_cols].copy()
        y_test_gen = df_original[holdout_mask][target_cols].copy()
        
        # Encode and scale
        X_train_gen = pd.get_dummies(X_train_gen, columns=['N', 'q'], prefix=['N', 'q'])
        X_test_gen = pd.get_dummies(X_test_gen, columns=['N', 'q'], prefix=['N', 'q'])
        
        # Ensure consistent columns
        for col in X_train_gen.columns:
            if col not in X_test_gen.columns:
                X_test_gen[col] = 0
        for col in X_test_gen.columns:
            if col not in X_train_gen.columns:
                X_train_gen[col] = 0
        
        X_test_gen = X_test_gen[X_train_gen.columns]
        
        scaler_gen = StandardScaler()
        X_train_gen_scaled = scaler_gen.fit_transform(X_train_gen)
        X_test_gen_scaled = scaler_gen.transform(X_test_gen)
        
        # Train a quick Random Forest model for generalization test
        rf_gen = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
        rf_gen.fit(X_train_gen_scaled, y_train_gen.values)
        
        # Predict and evaluate
        y_pred_gen = rf_gen.predict(X_test_gen_scaled)
        
        # Calculate metrics
        r2_xD = r2_score(y_test_gen.values[:, 0], y_pred_gen[:, 0])
        r2_QR = r2_score(y_test_gen.values[:, 1], y_pred_gen[:, 1])
        mae_xD = mean_absolute_error(y_test_gen.values[:, 0], y_pred_gen[:, 0])
        mae_QR = mean_absolute_error(y_test_gen.values[:, 1], y_pred_gen[:, 1])
        
        print(f"Generalization Test Results:")
        print(f"x_D: R² = {r2_xD:.4f}, MAE = {mae_xD:.4f}")
        print(f"Q_R: R² = {r2_QR:.4f}, MAE = {mae_QR:.1f}")
        
        # Plot generalization results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].scatter(y_test_gen.values[:, 0], y_pred_gen[:, 0], alpha=0.7)
        axes[0].plot([y_test_gen.values[:, 0].min(), y_test_gen.values[:, 0].max()], 
                    [y_test_gen.values[:, 0].min(), y_test_gen.values[:, 0].max()], 'r--')
        axes[0].set_xlabel('True x_D')
        axes[0].set_ylabel('Predicted x_D')
        axes[0].set_title(f'Generalization Test - x_D\nR² = {r2_xD:.4f}')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(y_test_gen.values[:, 1], y_pred_gen[:, 1], alpha=0.7, color='green')
        axes[1].plot([y_test_gen.values[:, 1].min(), y_test_gen.values[:, 1].max()], 
                    [y_test_gen.values[:, 1].min(), y_test_gen.values[:, 1].max()], 'r--')
        axes[1].set_xlabel('True Q_R (kW)')
        axes[1].set_ylabel('Predicted Q_R (kW)')
        axes[1].set_title(f'Generalization Test - Q_R\nR² = {r2_QR:.4f}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('generalization_test.png', dpi=300, bbox_inches='tight')
        plt.show()


class DistillationOptimizer:
    """
    Uses the best surrogate model for optimization
    """
    
    def __init__(self, model, scaler, feature_names, poly_features=None, model_type='random_forest'):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.poly_features = poly_features
        self.model_type = model_type
    
    def predict(self, R, B, x_F, F, N, q):
        """Make prediction using the surrogate model"""
        # Create input dataframe
        input_data = pd.DataFrame({
            'R': [R], 'B': [B], 'x_F': [x_F], 'F': [F], 'N': [N], 'q': [q]
        })
        
        # Encode categorical variables
        input_encoded = pd.get_dummies(input_data, columns=['N', 'q'], prefix=['N', 'q'])
        
        # Ensure all feature columns exist
        for col in self.feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[self.feature_names]
        
        # Scale and predict
        if self.model_type == 'polynomial':
            X_processed = self.poly_features.transform(self.scaler.transform(input_encoded))
        else:
            X_processed = self.scaler.transform(input_encoded)
        
        prediction = self.model.predict(X_processed)
        return prediction[0, 0], prediction[0, 1]  # x_D, Q_R
    
    def optimize_energy(self, target_purity=0.95, x_F=0.5, F=100, N=20, q=1.0):
        """
        Optimize operating conditions to minimize energy for target purity
        """
        print(f"\nOptimizing for target purity x_D = {target_purity:.3f}")
        
        from scipy.optimize import minimize
        
        def objective(x):
            """Objective function: minimize Q_R subject to x_D >= target_purity"""
            R, B = x
            
            # Bounds checking
            if R < 0.8 or R > 5.0 or B < 0.5 or B > 3.0:
                return 1e6
            
            try:
                x_D, Q_R = self.predict(R, B, x_F, F, N, q)
                
                # Penalty for not meeting purity constraint
                purity_penalty = max(0, target_purity - x_D) * 1e6
                
                return Q_R + purity_penalty
            
            except:
                return 1e6
        
        # Initial guess
        x0 = [2.0, 1.5]
        
        # Bounds
        bounds = [(0.8, 5.0), (0.5, 3.0)]
        
        # Optimize
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            R_opt, B_opt = result.x
            x_D_opt, Q_R_opt = self.predict(R_opt, B_opt, x_F, F, N, q)
            
            print(f"Optimization Results:")
            print(f"Optimal R = {R_opt:.3f}")
            print(f"Optimal B = {B_opt:.3f}")
            print(f"Achieved x_D = {x_D_opt:.4f}")
            print(f"Reboiler Duty = {Q_R_opt:.1f} kW")
            
            return {
                'R_optimal': R_opt,
                'B_optimal': B_opt,
                'x_D_achieved': x_D_opt,
                'Q_R_minimized': Q_R_opt,
                'success': True
            }
        else:
            print("Optimization failed")
            return {'success': False}


def main():
    """Main execution function"""
    print("="*60)
    print("AI/ML Surrogate Modeling for Binary Distillation Column")
    print("System: Ethanol-Water at 1 atm")
    print("="*60)
    
    # Step 1: Generate Data
    print("\n1. Data Generation")
    data_generator = DistillationDataGenerator()
    df = data_generator.generate_dataset(n_samples=500)
    
    # Save dataset
    df.to_csv('distill_data.csv', index=False)
    print(f"Dataset saved as 'distill_data.csv'")
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(df.describe())
    
    # Step 2: Data visualization
    print("\n2. Data Visualization")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Feature distributions
    features = ['R', 'B', 'x_F', 'F']
    for i, feature in enumerate(features):
        axes[0, i if i < 3 else i-1].hist(df[feature], bins=30, alpha=0.7, edgecolor='black')
        axes[0, i if i < 3 else i-1].set_xlabel(feature)
        axes[0, i if i < 3 else i-1].set_ylabel('Frequency')
        axes[0, i if i < 3 else i-1].set_title(f'Distribution of {feature}')
        axes[0, i if i < 3 else i-1].grid(True, alpha=0.3)
    
    # Target distributions
    axes[0, 2].hist(df['x_D'], bins=30, alpha=0.7, edgecolor='black', color='green')
    axes[0, 2].set_xlabel('x_D')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of x_D')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].hist(df['Q_R'], bins=30, alpha=0.7, edgecolor='black', color='red')
    axes[1, 0].set_xlabel('Q_R (kW)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Q_R')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation plots
    axes[1, 1].scatter(df['R'], df['x_D'], alpha=0.6)
    axes[1, 1].set_xlabel('Reflux Ratio (R)')
    axes[1, 1].set_ylabel('Distillate Purity (x_D)')
    axes[1, 1].set_title('x_D vs Reflux Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].scatter(df['x_F'], df['x_D'], alpha=0.6, color='purple')
    axes[1, 2].set_xlabel('Feed Composition (x_F)')
    axes[1, 2].set_ylabel('Distillate Purity (x_D)')
    axes[1, 2].set_title('x_D vs Feed Composition')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 3: Train ML Models
    print("\n3. Machine Learning Model Training")
    surrogate_models = DistillationSurrogateModels()
    
    # Prepare data
    data_splits = surrogate_models.prepare_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test, X_train_orig, X_val_orig, X_test_orig = data_splits
    
    # Train models
    surrogate_models.train_polynomial_regression(X_train, y_train, X_val, y_val)
    surrogate_models.train_random_forest(X_train, y_train, X_val, y_val)
    surrogate_models.train_svm(X_train, y_train, X_val, y_val)
    surrogate_models.train_neural_network(X_train, y_train, X_val, y_val)
    
    # Step 4: Model Evaluation
    print("\n4. Model Evaluation")
    results = surrogate_models.evaluate_models(X_test, y_test, X_val, y_val)
    
    # Create evaluation plots
    surrogate_models.create_evaluation_plots(X_test, y_test)
    
    # Step 5: Physical Consistency Analysis
    print("\n5. Physical Consistency Analysis")
    consistency_results = surrogate_models.physical_consistency_check(df, X_test_orig, y_test)
    
    # Step 6: Generalization Test
    print("\n6. Generalization Test")
    surrogate_models.generalization_test(df)
    
    # Step 7: Optimization
    print("\n7. Optimization Using Best Model")
    
    # Get best model
    best_model_name = max(results.keys(), 
                         key=lambda x: (results[x]['test']['x_D_r2'] + 
                                      results[x]['test']['Q_R_r2']) / 2)
    
    best_model = surrogate_models.models[best_model_name]
    poly_features = surrogate_models.poly_features if best_model_name == 'polynomial' else None
    
    optimizer = DistillationOptimizer(
        model=best_model,
        scaler=surrogate_models.scaler,
        feature_names=surrogate_models.feature_names,
        poly_features=poly_features,
        model_type=best_model_name
    )
    
    # Optimize for different target purities
    target_purities = [0.90, 0.95, 0.98]
    optimization_results = []
    
    for target in target_purities:
        opt_result = optimizer.optimize_energy(target_purity=target)
        if opt_result['success']:
            optimization_results.append(opt_result)
    
    # Save models
    print("\n8. Saving Models and Results")
    joblib.dump(best_model, f'best_model_{best_model_name}.pkl')
    joblib.dump(surrogate_models.scaler, 'scaler.pkl')
    if poly_features:
        joblib.dump(poly_features, 'poly_features.pkl')
    
    print(f"Best model ({best_model_name}) saved successfully")
    
    # Final Summary
    print("\n" + "="*60)
    print("PROJECT SUMMARY")
    print("="*60)
    print(f"Dataset size: {len(df)} samples")
    print(f"Best model: {best_model_name}")
    print(f"Best model performance:")
    best_results = results[best_model_name]['test']
    print(f"  x_D: R² = {best_results['x_D_r2']:.4f}, MAE = {best_results['x_D_mae']:.4f}")
    print(f"  Q_R: R² = {best_results['Q_R_r2']:.4f}, MAE = {best_results['Q_R_mae']:.1f}")
    
    print(f"\nPhysical Consistency:")
    print(f"  Bound violations: {consistency_results['bound_violations']}/{consistency_results['total_predictions']}")
    print(f"  Monotonicity violations: {consistency_results['monotonic_violations']}/{consistency_results['monotonic_tests']}")
    
    if optimization_results:
        print(f"\nOptimization Results:")
        for i, result in enumerate(optimization_results):
            print(f"  Target x_D = {target_purities[i]:.2f}: R = {result['R_optimal']:.3f}, "
                  f"Q_R = {result['Q_R_minimized']:.1f} kW")


if __name__ == "__main__":
    main()