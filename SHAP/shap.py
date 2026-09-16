import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Increase the reproducibility of the model
def set_deterministic(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_deterministic(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# ==================== Data preprocessing ====================
dataPath = 'Mydata_NN.csv'
df = pd.read_csv(dataPath, header=0, index_col=None)
x_data = df.iloc[:, :-1].values.astype(np.float32)
y_data = df.iloc[:, -1].values.astype(np.float32)
raw_feature_names = df.columns[:-1].tolist()

def create_preprocessor():
    return Pipeline([
        ('preprocess', ColumnTransformer([
            ('num', Pipeline([
                ('yeojohnson', PowerTransformer(method='yeo-johnson')), 
                ('minmax', MinMaxScaler()) 
            ]), slice(0, 8)),
            ('passthrough', 'passthrough', slice(8, None)) 
        ]))
    ])


# ==================== Model framework contruction ====================
class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 214), nn.ReLU(), 
            nn.Linear(214, 248), nn.ReLU(),
            nn.Linear(248, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# ==================== Cross-validation loop ====================
def final_validation():
    kf = KFold(n_splits=10, shuffle=False)
    all_metrics = {'MAE': [], 'RMSE': [], 'R2': []}
    all_shap_values = []
    all_X_processed = []

    for fold in range(10):
        print(f"\n {fold+1} fold validation:")
  
        #load the preprocessor
        preprocessor = joblib.load(f'My_saved_pipelines_NN\\preprocessor_fold{fold}.pkl')
        train_idx, val_idx = list(kf.split(x_data))[fold]
        X_val, y_val = x_data[val_idx], y_data[val_idx]

       
        X_val_processed = preprocessor.transform(X_val)
        val_xt = torch.from_numpy(X_val_processed).float().to(device)
        val_yt = torch.from_numpy(y_val).float().unsqueeze(1).to(device)
        val_loader = Data.DataLoader(Data.TensorDataset(val_xt, val_yt), batch_size=9)

        #load the model
        model = Model(x_data.shape[1]).to(device)
        model.load_state_dict(torch.load(f'My_saved_pipelines_NN\\model_fold{fold}.pth',weights_only=True))
        model.eval()

        # Evaluate the model
        y_pred, y_true = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                y_pred.append(outputs.cpu().numpy())
                y_true.append(targets.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
  
        all_metrics['MAE'].append(mean_absolute_error(y_true, y_pred))
        all_metrics['RMSE'].append(np.sqrt(mean_squared_error(y_true, y_pred)))
        all_metrics['R2'].append(r2_score(y_true, y_pred))

        # ==================== SHAP analysis ====================
        X_train = x_data[train_idx]
        X_train_processed = preprocessor.transform(X_train)
        X_train_tensor = torch.from_numpy(X_train_processed).float().to(device)
        background = X_train_tensor[:].clone().requires_grad_(True)
  
        model.eval()
        explainer = shap.DeepExplainer(model, background)
  
        with torch.enable_grad():
            val_xt_requires_grad = val_xt.clone().requires_grad_(True)
            shap_values = explainer.shap_values(val_xt_requires_grad)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
          
        model.eval()
        all_shap_values.append(shap_values)
        all_X_processed.append(X_val_processed)

    print("\nCross-Validation Results:")
    print(f"MAE: {np.mean(all_metrics['MAE']):.4f} ± {np.std(all_metrics['MAE']):.4f}")
    print(f"RMSE: {np.mean(all_metrics['RMSE']):.4f} ± {np.std(all_metrics['RMSE']):.4f}")
    print(f"R2: {np.mean(all_metrics['R2']):.4f} ± {np.std(all_metrics['R2']):.4f}")

    # ==================== SHAP visualization ====================
    all_shap_values = np.concatenate(all_shap_values, axis=0)
    all_X_processed = np.concatenate(all_X_processed, axis=0)

    def merge_shap_features(shap_array, X_processed):
        merged_shap = []
        for i in range(shap_array.shape[0]):
            part1 = shap_array[i, :9]

            # support shape features
            support_idx = np.argmax(X_processed[i, 9:13])
            support_shap = shap_array[i, 9 + support_idx]

            # catalyst morphology features
            morphology_idx = np.argmax(X_processed[i, 13:20])
            morphology_shap = shap_array[i, 13 + morphology_idx]

            merged_row = np.concatenate([part1, [support_shap], [morphology_shap]])
            merged_shap.append(merged_row)
        return np.array(merged_shap)

    def merge_original_features(feature_array):
        """convert one-hot encoding to numerical features"""
        merged_features = []
        for row in feature_array:
            part1 = row[:9]
            
            support = np.argmax(row[9:13])
            morphology = np.argmax(row[13:20])
            
            merged_row = np.concatenate([part1, [support], [morphology]])
            merged_features.append(merged_row)
        return np.array(merged_features)

    # merge SHAP values and original features
    merged_shap = merge_shap_features(all_shap_values.squeeze(axis=2), all_X_processed)
    merged_features = merge_original_features(all_X_processed)

    # feature names
    merged_feature_names = (
        raw_feature_names[:9] +  
        ['Support shape', 'Catalyst morphology']   
    )

    # SHAP scatter plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        merged_shap,
        features=merged_features,
        feature_names=merged_feature_names,
        plot_type="dot",
        show=False,
        plot_size=(10, 8)
    )
    
    # SHAP bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        merged_shap,
        features=None,
        feature_names=merged_feature_names,
        plot_type="bar",
        show=False)
    plt.show()

if __name__ == "__main__":
    final_validation()
