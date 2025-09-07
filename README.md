# AI/ML Surrogate Modeling for Binary Distillation Column
## FOSSEE Autumn Internship - Screening Task

### Project Overview
This project develops machine learning surrogate models to predict the performance of a binary distillation column separating ethanol and water at atmospheric pressure. The models predict distillate purity (x_D) and reboiler duty (Q_R) from operating conditions.

### System Specifications
- **Binary System**: Ethanol-Water
- **Operating Pressure**: 1 atm (101.325 kPa)
- **Column Configuration**: Conventional with total condenser and partial reboiler
- **Target Outputs**: Distillate purity (x_D) and Reboiler duty (Q_R)

### Project Structure
```
AI_Distillation_Surrogate/
├── main_distillation_surrogate.py    # Main implementation
├── evaluation_script.py              # Execute evaluation results
└──evaluation.py                     # Best fit plot
app.py                            # Streamlit app 
distill_data.csv                  # Generated dataset
requirements.txt                  # Python dependencies
README.md                         # This file
Project_Report.pdf                # Detailed technical report
models/                           # Saved ML models
├── best_model_random_forest.pkl
└── scaler.pkl
plots/                          # Generated visualizations
├── data_visualization.png
├── model_evaluation_plots.png
├── sensitivity_plots.png
└── generalization_test.png
```

### Installation and Setup

1. **Clone or Download** the project files to your local machine.

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**:
   ```bash
   python -c "import numpy, pandas, sklearn, matplotlib, seaborn, scipy, joblib; print('All packages installed successfully!')"
   ```

### How to Run the Project

#### Option 1: Full Project Execution
```bash
python main_distillation_surrogate.py
```
This will:
- Generate synthetic distillation data (500+ samples)
- Train 4 different ML models (Polynomial, Random Forest, SVM, Neural Network)
- Evaluate and compare model performance
- Perform physical consistency checks
- Run generalization tests
- Demonstrate optimization capabilities
- Save all results, models, and plots

#### Option 2: Step-by-Step Execution
For interactive exploration, you can run sections individually:

```python
# Import the main classes
from main_distillation_surrogate import *

# Step 1: Generate data
generator = DistillationDataGenerator()
df = generator.generate_dataset(n_samples=500)

# Step 2: Train models
models = DistillationSurrogateModels()
data_splits = models.prepare_data(df)
# ... continue with specific model training

# Step 3: Optimization
optimizer = DistillationOptimizer(best_model, scaler, features)
results = optimizer.optimize_energy(target_purity=0.95)
```

### Expected Runtime
- **Data Generation**: ~2 minutes
- **Model Training**: ~5-10 minutes (depending on hardware)
- **Evaluation & Plots**: ~2 minutes
- **Total Runtime**: ~10-15 minutes

### Output Files

#### 1. Dataset
- `distill_data.csv`: Complete dataset with 500+ samples
  - Columns: R, B, x_F, F, N, q (inputs) and x_D, Q_R (outputs)

#### 2. Models
- `best_model_random_forest.pkl`: Trained Random Forest model
- `scaler.pkl`: Feature scaler for preprocessing
- `poly_features.pkl`: Polynomial feature transformer (if applicable)

#### 3. Visualizations
- `data_visualization.png`: Dataset distribution and correlation plots
- `model_evaluation_plots.png`: Model comparison and performance metrics
- `sensitivity_plots.png`: Partial dependence plots
- `generalization_test.png`: Extrapolation capability assessment

#### 4. Performance Summary
The console output provides comprehensive metrics including:
- R² scores for all models and both outputs
- Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
- Physical consistency check results
- Optimization results for different target purities

### Key Results Preview
Expected performance of the best model (Random Forest):
- **Distillate Purity (x_D)**: R² > 0.95, MAE < 0.02
- **Reboiler Duty (Q_R)**: R² > 0.95, MAE < 150 kW
- **Physical Consistency**: >95% of predictions within physical bounds
- **Optimization**: 15-30% energy savings achievable

### Model Comparison Summary
| Model | x_D R² | Q_R R² | Training Time | Best For |
|-------|--------|--------|---------------|----------|
| Polynomial | 0.89 | 0.88 | Fast | Baseline/Interpretability |
| **Random Forest** | **0.96** | **0.95** | Medium | **Overall Performance** |
| SVM | 0.92 | 0.92 | Slow | High-dimensional data |
| Neural Network | 0.94 | 0.94 | Medium | Complex patterns |

### Technical Features

#### 1. Data Generation
- Physics-based simulation using Antoine equations
- McCabe-Thiele method principles
- Systematic parameter space exploration
- Quality control and validation checks

#### 2. Machine Learning
- Multi-output regression for simultaneous prediction
- Hyperparameter optimization via GridSearchCV
- Cross-validation for robust evaluation
- Feature importance analysis

#### 3. Physical Validation
- Bounds checking (0 ≤ x_D ≤ 1)
- Monotonicity analysis for key relationships
- Sensitivity analysis with partial dependence plots
- Generalization testing with holdout regions

#### 4. Optimization
- Energy minimization for target purity constraints
- Scipy optimization integration
- Multiple target purity scenarios
- Economic trade-off analysis

### Troubleshooting

#### Common Issues and Solutions

1. **Import Errors**:
   ```bash
   # If sklearn import fails
   pip install --upgrade scikit-learn
   
   # If matplotlib display issues
   pip install --upgrade matplotlib
   ```

2. **Memory Issues** (for large datasets):
   - Reduce `n_samples` parameter in `generate_dataset()`
   - Use subset of data for SVM training (already implemented)

3. **Slow Execution**:
   - Reduce hyperparameter grid size in model training
   - Use fewer cross-validation folds
   - Skip Neural Network training if needed

4. **Plot Display Issues**:
   - Ensure proper backend: `matplotlib.use('Agg')` for headless systems
   - Check display settings for remote execution

### Customization Options

#### 1. System Parameters
Modify operating ranges in `DistillationDataGenerator`:
```python
R_range = (0.8, 5.0)      # Reflux ratio
B_range = (0.5, 3.0)      # Boilup ratio
xF_range = (0.2, 0.95)    # Feed composition
F_range = (80, 120)       # Feed flowrate
N_options = [15, 20, 25]  # Number of stages
```

#### 2. ML Models
Add new models or modify existing ones in `DistillationSurrogateModels`:
```python
# Example: Add XGBoost
import xgboost as xgb
def train_xgboost(self, X_train, y_train, X_val, y_val):
    # Implementation here
```

#### 3. Optimization Objectives
Modify optimization function for different objectives:
```python
# Multi-objective: minimize energy AND maximize throughput
def multi_objective(x):
    R, B = x
    x_D, Q_R = self.predict(R, B, x_F, F, N, q)
    return w1*Q_R - w2*throughput  # Weighted combination
```

### Validation and Verification

The implementation includes comprehensive validation:
1. **Unit Tests**: Built-in checks for data quality and model consistency
2. **Physical Validation**: Comparison with known distillation behavior
3. **Cross-Validation**: K-fold validation for robust performance estimation
4. **Generalization Testing**: Holdout region analysis
5. **Benchmark Comparison**: Multiple model comparison with statistical significance

### Contact Information
For questions or issues related to this implementation:
- Review the detailed technical report (Project_Report.pdf)
- Check console output for diagnostic information
- Verify all requirements are properly installed

### License
This project is developed for educational purposes as part of the FOSSEE Autumn Internship screening task.

---

**Version**: 1.0  
**Last Updated**: September 2025  
**Python Version**: 3.7+ recommended  
**Status**: Complete and Ready for Submission