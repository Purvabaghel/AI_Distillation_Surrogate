"""
Quick Evaluation Script for Distillation Surrogate Models
This script demonstrates how to use the trained models for predictions and optimization
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from main_distillation_surrogate import DistillationOptimizer

def load_trained_model():
    """Load the pre-trained model and preprocessing objects"""
    try:
        # Load the best model (Random Forest)
        model = joblib.load('best_model_random_forest.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Feature names (should match training data)
        feature_names = ['R', 'B', 'x_F', 'F', 'N_15', 'N_20', 'N_25', 'q_0.8', 'q_1.0', 'q_1.2']
        
        print("✅ Model loaded successfully!")
        return model, scaler, feature_names
    
    except FileNotFoundError as e:
        print(f"❌ Error loading model: {e}")
        print("Please run the main script first to train and save the models.")
        return None, None, None

def predict_single_case(model, scaler, feature_names, R, B, x_F, F, N, q):
    """Make a single prediction using the trained model"""
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'R': [R], 'B': [B], 'x_F': [x_F], 'F': [F], 'N': [N], 'q': [q]
    })
    
    # One-hot encode discrete variables
    input_encoded = pd.get_dummies(input_data, columns=['N', 'q'], prefix=['N', 'q'])
    
    # Ensure all feature columns exist
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_names]
    
    # Scale features
    X_scaled = scaler.transform(input_encoded)
    
    # Make prediction
    prediction = model.predict(X_scaled)
    return prediction[0, 0], prediction[0, 1]  # x_D, Q_R

def demonstrate_predictions():
    """Demonstrate model predictions for various operating conditions"""
    
    # Load model
    model, scaler, feature_names = load_trained_model()
    if model is None:
        return
    
    print("\n" + "="*60)
    print("DISTILLATION COLUMN PERFORMANCE PREDICTIONS")
    print("="*60)
    
    # Test cases
    test_cases = [
        {"R": 1.5, "B": 1.2, "x_F": 0.4, "F": 100, "N": 20, "q": 1.0, "description": "Low reflux, moderate feed"},
        {"R": 3.0, "B": 1.8, "x_F": 0.6, "F": 100, "N": 20, "q": 1.0, "description": "High reflux, rich feed"},
        {"R": 2.0, "B": 1.5, "x_F": 0.3, "F": 120, "F": 100, "N": 25, "q": 0.8, "description": "High stages, subcooled feed"},
        {"R": 4.0, "B": 2.2, "x_F": 0.8, "F": 80, "N": 15, "q": 1.2, "description": "Very high reflux, very rich feed"}
    ]
    
    print(f"{'Case':<5} {'Description':<25} {'R':<5} {'B':<5} {'x_F':<6} {'x_D':<8} {'Q_R (kW)':<10} {'Efficiency':<12}")
    print("-" * 85)
    
    for i, case in enumerate(test_cases, 1):
        x_D, Q_R = predict_single_case(
            model, scaler, feature_names,
            case["R"], case["B"], case["x_F"], case["F"], case["N"], case["q"]
        )
        
        # Calculate separation efficiency
        efficiency = (x_D - case["x_F"]) / (0.99 - case["x_F"]) * 100
        
        print(f"{i:<5} {case['description']:<25} {case['R']:<5} {case['B']:<5} {case['x_F']:<6} "
              f"{x_D:<8.4f} {Q_R:<10.1f} {efficiency:<12.1f}%")

def sensitivity_analysis():
    """Perform sensitivity analysis on key parameters"""
    
    # Load model
    model, scaler, feature_names = load_trained_model()
    if model is None:
        return
    
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Base case
    base_case = {"B": 1.5, "x_F": 0.5, "F": 100, "N": 20, "q": 1.0}
    
    # Reflux ratio sensitivity
    R_values = np.linspace(1.0, 4.0, 10)
    x_D_values = []
    Q_R_values = []
    
    for R in R_values:
        x_D, Q_R = predict_single_case(model, scaler, feature_names, R, **base_case)
        x_D_values.append(x_D)
        Q_R_values.append(Q_R)
    
    # Create sensitivity plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(R_values, x_D_values, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Reflux Ratio (R)')
    ax1.set_ylabel('Distillate Purity (x_D)')
    ax1.set_title('Effect of Reflux Ratio on Purity')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(R_values, Q_R_values, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Reflux Ratio (R)')
    ax2.set_ylabel('Reboiler Duty (kW)')
    ax2.set_title('Effect of Reflux Ratio on Energy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Sensitivity analysis plot saved as 'quick_sensitivity_analysis.png'")
    plt.show()
    
    # Print some key insights
    print(f"\nKey Insights:")
    print(f"- Increasing R from {R_values[0]:.1f} to {R_values[-1]:.1f}:")
    print(f"  • Purity increases from {x_D_values[0]:.4f} to {x_D_values[-1]:.4f}")
    print(f"  • Energy increases from {Q_R_values[0]:.1f} to {Q_R_values[-1]:.1f} kW")
    print(f"  • Energy penalty: {(Q_R_values[-1]/Q_R_values[0]-1)*100:.1f}% for purity gain")

def optimization_demo():
    """Demonstrate optimization capabilities"""
    
    # Load model
    model, scaler, feature_names = load_trained_model()
    if model is None:
        return
    
    print("\n" + "="*60)
    print("OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Create optimizer
    optimizer = DistillationOptimizer(model, scaler, feature_names, model_type='random_forest')
    
    # Test different scenarios
    scenarios = [
        {"x_F": 0.3, "target": 0.90, "description": "Lean feed, moderate purity"},
        {"x_F": 0.5, "target": 0.95, "description": "Typical feed, high purity"},
        {"x_F": 0.7, "target": 0.98, "description": "Rich feed, ultra-high purity"}
    ]
    
    print(f"{'Scenario':<25} {'Feed x_F':<10} {'Target x_D':<12} {'Opt. R':<8} {'Opt. B':<8} {'Energy (kW)':<12}")
    print("-" * 85)
    
    for scenario in scenarios:
        result = optimizer.optimize_energy(
            target_purity=scenario["target"],
            x_F=scenario["x_F"],
            F=100, N=20, q=1.0
        )
        
        if result['success']:
            print(f"{scenario['description']:<25} {scenario['x_F']:<10.2f} {scenario['target']:<12.2f} "
                  f"{result['R_optimal']:<8.2f} {result['B_optimal']:<8.2f} {result['Q_R_minimized']:<12.1f}")
        else:
            print(f"{scenario['description']:<25} {scenario['x_F']:<10.2f} {scenario['target']:<12.2f} {'FAILED':<30}")

def model_diagnostics():
    """Run basic model diagnostics"""
    
    # Load model
    model, scaler, feature_names = load_trained_model()
    if model is None:
        return
    
    print("\n" + "="*60)
    print("MODEL DIAGNOSTICS")
    print("="*60)
    
    # Test physical bounds
    print("Testing physical bounds...")
    test_points = 100
    violations = 0
    
    np.random.seed(42)
    for _ in range(test_points):
        R = np.random.uniform(0.8, 5.0)
        B = np.random.uniform(0.5, 3.0)
        x_F = np.random.uniform(0.2, 0.95)
        F = np.random.uniform(80, 120)
        N = np.random.choice([15, 20, 25])
        q = np.random.choice([0.8, 1.0, 1.2])
        
        x_D, Q_R = predict_single_case(model, scaler, feature_names, R, B, x_F, F, N, q)
        
        if x_D < 0 or x_D > 1 or Q_R < 0:
            violations += 1
    
    print(f"Physical bound violations: {violations}/{test_points} ({100*violations/test_points:.1f}%)")
    
    # Feature importance (if Random Forest)
    if hasattr(model, 'estimators_'):
        print("\nFeature Importance Analysis:")
        # Get feature importances from the Random Forest
        if hasattr(model.estimators_[0], 'feature_importances_'):
            importance_x_D = model.estimators_[0].feature_importances_
            importance_Q_R = model.estimators_[1].feature_importances_
            
            print(f"{'Feature':<15} {'x_D Importance':<15} {'Q_R Importance':<15}")
            print("-" * 45)
            for i, feature in enumerate(feature_names):
                print(f"{feature:<15} {importance_x_D[i]:<15.4f} {importance_Q_R[i]:<15.4f}")
    
    print("\n✅ Model diagnostics completed!")

def interactive_calculator():
    """Interactive calculator for column performance"""
    
    # Load model
    model, scaler, feature_names = load_trained_model()
    if model is None:
        return
    
    print("\n" + "="*60)
    print("INTERACTIVE DISTILLATION CALCULATOR")
    print("="*60)
    print("Enter operating conditions to predict column performance")
    print("(Press Enter for default values)")
    
    while True:
        try:
            print("\n" + "-"*40)
            
            # Get user inputs with defaults
            R_input = input("Reflux Ratio (R) [default: 2.0]: ").strip()
            R = float(R_input) if R_input else 2.0
            
            B_input = input("Boilup Ratio (B) [default: 1.5]: ").strip()
            B = float(B_input) if B_input else 1.5
            
            x_F_input = input("Feed Composition (x_F) [default: 0.5]: ").strip()
            x_F = float(x_F_input) if x_F_input else 0.5
            
            F_input = input("Feed Flowrate (F, kmol/h) [default: 100]: ").strip()
            F = float(F_input) if F_input else 100
            
            N_input = input("Number of Stages (N) [15, 20, 25, default: 20]: ").strip()
            N = int(N_input) if N_input else 20
            
            q_input = input("Feed Condition (q) [0.8, 1.0, 1.2, default: 1.0]: ").strip()
            q = float(q_input) if q_input else 1.0
            
            # Validate inputs
            if not (0.8 <= R <= 5.0):
                print("⚠️ Warning: R outside typical range [0.8, 5.0]")
            if not (0.5 <= B <= 3.0):
                print("⚠️ Warning: B outside typical range [0.5, 3.0]")
            if not (0.2 <= x_F <= 0.95):
                print("⚠️ Warning: x_F outside typical range [0.2, 0.95]")
            
            # Make prediction
            x_D, Q_R = predict_single_case(model, scaler, feature_names, R, B, x_F, F, N, q)
            
            # Display results
            print("\n🎯 PREDICTION RESULTS:")
            print(f"   Distillate Purity (x_D): {x_D:.4f} ({x_D*100:.2f}% ethanol)")
            print(f"   Reboiler Duty (Q_R): {Q_R:.1f} kW")
            
            # Additional calculations
            recovery = (x_D - x_F) / (0.99 - x_F) * 100 if x_F < 0.99 else 100
            specific_energy = Q_R / F if F > 0 else 0
            
            print(f"   Separation Efficiency: {recovery:.1f}%")
            print(f"   Specific Energy: {specific_energy:.1f} kW/(kmol/h)")
            
            # Physical checks
            if x_D <= x_F:
                print("⚠️  Warning: Predicted distillate purity is not higher than feed!")
            if x_D > 0.999:
                print("⚠️  Warning: Very high purity may require validation")
            if Q_R < 0:
                print("❌ Error: Negative energy prediction - check inputs")
            
            # Continue or exit
            continue_input = input("\nMake another prediction? (y/n) [y]: ").strip().lower()
            if continue_input in ['n', 'no']:
                break
                
        except ValueError:
            print("❌ Invalid input. Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def benchmark_comparison():
    """Compare model predictions with simple correlations"""
    
    # Load model
    model, scaler, feature_names = load_trained_model()
    if model is None:
        return
    
    print("\n" + "="*60)
    print("BENCHMARK COMPARISON")
    print("="*60)
    
    # Simple correlation for comparison (Gilliland correlation approximation)
    def simple_correlation(R, x_F):
        """Simplified distillate purity correlation"""
        # Very approximate - for comparison only
        x_D_simple = x_F + (1 - x_F) * (R - 1) / (R + 1) * 0.8
        return min(0.999, max(x_F, x_D_simple))
    
    # Test cases
    test_cases = [
        (1.5, 1.2, 0.3, 100, 20, 1.0),
        (2.0, 1.5, 0.5, 100, 20, 1.0),
        (3.0, 2.0, 0.7, 100, 20, 1.0),
        (4.0, 2.5, 0.8, 100, 20, 1.0),
    ]
    
    print(f"{'R':<5} {'B':<5} {'x_F':<6} {'ML x_D':<8} {'Simple x_D':<12} {'Difference':<12}")
    print("-" * 55)
    
    differences = []
    for R, B, x_F, F, N, q in test_cases:
        x_D_ml, _ = predict_single_case(model, scaler, feature_names, R, B, x_F, F, N, q)
        x_D_simple = simple_correlation(R, x_F)
        difference = abs(x_D_ml - x_D_simple)
        differences.append(difference)
        
        print(f"{R:<5} {B:<5} {x_F:<6} {x_D_ml:<8.4f} {x_D_simple:<12.4f} {difference:<12.4f}")
    
    avg_difference = np.mean(differences)
    print(f"\nAverage difference: {avg_difference:.4f}")
    print("Note: Simple correlation is very approximate - ML model should be more accurate")

def main():
    """Main function to run all demonstrations"""
    
    print("🚀 DISTILLATION SURROGATE MODEL EVALUATION")
    print("="*60)
    print("This script demonstrates the capabilities of the trained ML models")
    print("for predicting binary distillation column performance.")
    
    # Menu system
    while True:
        print("\n📋 AVAILABLE DEMONSTRATIONS:")
        print("1. Model Predictions for Test Cases")
        print("2. Sensitivity Analysis")
        print("3. Optimization Demo")
        print("4. Model Diagnostics")
        print("5. Interactive Calculator")
        print("6. Benchmark Comparison")
        print("7. Run All Demos")
        print("0. Exit")
        
        choice = input("\nSelect option (0-7): ").strip()
        
        if choice == '1':
            demonstrate_predictions()
        elif choice == '2':
            sensitivity_analysis()
        elif choice == '3':
            optimization_demo()
        elif choice == '4':
            model_diagnostics()
        elif choice == '5':
            interactive_calculator()
        elif choice == '6':
            benchmark_comparison()
        elif choice == '7':
            print("\n🎯 Running all demonstrations...")
            demonstrate_predictions()
            sensitivity_analysis()
            optimization_demo()
            model_diagnostics()
            benchmark_comparison()
            print("\n✅ All demonstrations completed!")
        elif choice == '0':
            print("\n👋 Thank you for using the Distillation Surrogate Model!")
            break
        else:
            print("❌ Invalid option. Please select 0-7.")

if __name__ == "__main__":
    main()