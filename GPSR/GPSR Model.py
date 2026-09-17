import pandas as pd
import numpy as np
import joblib

from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# ==================== Data preprocessing ====================
# Read data
dataPath = 'Mydata_gpsr.csv'
data = pd.read_csv(dataPath, header=0, index_col=None)

feature_cols = [
    "TMO_AN",  # Atomic number of metals in TMO
    "TMO_VS",  # Valence state of metals in TMO
    "TMO_EN",  # Electronegativity of metals in TMO
    "TMO_BG",  # Band gap of TMO
    "TMO_CB",  # Conduction band position of TMO
    "TMO_VB",  # Valence band position of TMO
    "Dop_AN",  # Atomic number of doping metal
    "Dop_EN",  # Electronegativity of doping metal
    "Dop_AR",  # Atomic radius of doping metal
    "Dop_MF",  # Mass fraction of doping metal
]

target_col = "FE"

X = data[feature_cols].values
y = data[target_col].values

if y.min() < 0 or y.max() > 1:
    raise ValueError("FE should be between 0 and 1.")

# Split the training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "minmax_scaler.pkl")
print("MinMaxScaler has been saved: minmax_scaler.pkl")

# Custom R2
def r2_fitness(y_true, y_pred, sample_weight):
    return r2_score(y_true, y_pred, sample_weight=sample_weight)

r2_metric = make_fitness(
    function=r2_fitness,
    greater_is_better=True
)


# ==================== Build a  symbolic regression model ====================
function_set = (
    "add",
    "sub", 
    "mul",
    "div",
)

model = SymbolicRegressor(
    population_size=5000,  # Population size in every generation
    generations=300,
    tournament_size=20,

    function_set=function_set,
    metric=r2_metric,

    parsimony_coefficient=0.003,
    init_depth=(6, 10),

    p_crossover=0.5,  # probability of crossover 
    p_subtree_mutation=0.1,  # probability of subtree mutation
    p_hoist_mutation=0.1,  # probability of hoist mutation 
    p_point_mutation=0.3,  # probability of point mutation

    max_samples=0.9,
    stopping_criteria=0.900,

    verbose=1,
    random_state=42,
    n_jobs=-1
)


# ==================== Train the model ====================
model.fit(X_train, y_train)

# Output formula
program_str = str(model._program)

print("\nDiscovered formula:")
print(program_str)

print("\nVariable correspondences:")
for i, name in enumerate(feature_cols):
    print(f"X{i} = {name}, normalized variable")

# ==================== Model performance evaluation ====================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\nTraining set performance:")
print(f"R2   = {train_r2:.4f}")
print(f"RMSE = {train_rmse:.4f}")
print(f"MAE  = {train_mae:.4f}")

print("\nTest set performance:")
print(f"R2   = {test_r2:.4f}")
print(f"RMSE = {test_rmse:.4f}")
print(f"MAE  = {test_mae:.4f}")

# ==================== Save model and results ====================
joblib.dump(model, "gpsr_model.pkl")

result_df = pd.DataFrame({
    "y_true_train": y_train,
    "y_pred_train": y_train_pred
})

test_result_df = pd.DataFrame({
    "y_true_test": y_test,
    "y_pred_test": y_test_pred
})

summary_df = pd.DataFrame([{
    "formula": program_str,
    "train_r2": train_r2,
    "test_r2": test_r2,
    "train_rmse": train_rmse,
    "test_rmse": test_rmse,
    "train_mae": train_mae,
    "test_mae": test_mae
}])

print("\nResults saved:")
print("minmax_scaler.pkl")
print("gpsr_model.pkl")

