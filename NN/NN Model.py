import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import os
import joblib


# Increase the reproducibility of the model
def set_deterministic(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    import random
    random.seed(seed)
set_deterministic(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# ==================== Data preprocessing ====================
# Read data
dataPath = 'Mydata.csv'
df = pd.read_csv(dataPath, header=0, index_col=None)

# Divide input and out features
x_data = df.iloc[:, :-1].values.astype(np.float32)
y_data = df.iloc[:, -1].values.astype(np.float32)

# Create a save directory for pipelines
os.makedirs('My_saved_pipelines', exist_ok=True) 

def create_preprocessor():
    '''Preprocessing pipeline'''
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
    ''' NN model'''
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 214), nn.ReLU(), 
            nn.Linear(214, 248), nn.ReLU(),
            nn.Linear(248, 1), nn.Sigmoid()  
        )
    
    def forward(self, x):
        '''forward propagation'''
        y = self.net(x)
        return y

# Hyperparameters configuration
k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=False) 
batch_size = 29
learning_rate = 0.000865
num_epochs = 1500
Loss_Fn = nn.MSELoss()
Optimizer = torch.optim.Adam

# Results storage 
cv_results = {
    'mae': [], 'rmse': [], 'r2': [],
    'all_true': np.array([]),  
    'all_pred': np.array([])   
}
history_loss = {'train': [[] for _ in range(k_folds)], 'val': [[] for _ in range(k_folds)]}

# ==================== Cross-validation loop ====================
for fold, (train_idx, val_idx) in enumerate(kf.split(x_data)):

    # Divide data into training and validation sets
    x_train_fold, x_val_fold = x_data[train_idx], x_data[val_idx]
    y_train_fold, y_val_fold = y_data[train_idx], y_data[val_idx]

    # Preprocess the data
    preprocessor = create_preprocessor()
    x_train_processed = preprocessor.fit_transform(x_train_fold)
    x_val_processed = preprocessor.transform(x_val_fold)
    
    # Save the preprocessing pipeline
    joblib.dump(preprocessor, f'My_saved_pipelines/preprocessor_fold{fold}.pkl')

    # Convert to PyTorch tensors 
    train_xt = torch.from_numpy(x_train_processed).to(device)
    train_yt = torch.from_numpy(y_train_fold).to(device)
    val_xv = torch.from_numpy(x_val_processed).to(device)
    val_yv = torch.from_numpy(y_val_fold).to(device)

    # Create DataLoader
    train_data = Data.TensorDataset(train_xt, train_yt)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    val_data = Data.TensorDataset(val_xv, val_yv)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = Model(input_size=x_data.shape[1]).to(device)
    loss_fn = Loss_Fn     
    optimizer = Optimizer(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_num = 0
        for x, y in train_loader:
            output = model(x)
            y = y.unsqueeze(1)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            train_num += x.size(0)
        
        # Calculate average training loss
        train_loss_avg = train_loss / train_num
        history_loss['train'][fold].append(train_loss_avg)

        # Validation phase
        model.eval()
        val_loss = 0
        val_num = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                output = model(x_val)
                y_val = y_val.unsqueeze(1)
                loss = loss_fn(output, y_val)

                val_loss += loss.item() * x_val.size(0)
                val_num += x_val.size(0)

        val_loss_avg = val_loss / val_num 
        history_loss['val'][fold].append(val_loss_avg)
      
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")

    # Save the model 
    torch.save(model.state_dict(), f'My_saved_pipelines/model_fold{fold}.pth')

    #  ==================== Evaluate the model on the validation set ====================
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            output = model(x_val)
            y_val = y_val.unsqueeze(1)
            y_pred.append(output.cpu().numpy())
            y_true.append(y_val.cpu().numpy())
    
    y_pred = np.concatenate(y_pred).flatten()
    y_true = np.concatenate(y_true).flatten()
    
    cv_results['all_true'] = np.concatenate([cv_results['all_true'], y_true])
    cv_results['all_pred'] = np.concatenate([cv_results['all_pred'], y_pred])

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    cv_results['mae'].append(mae)
    cv_results['rmse'].append(rmse)
    cv_results['r2'].append(r2)
   
    print(f"Fold {fold + 1} Final | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

# Cross-validation results 
print("\nCross-Validation Results:")
print(f"Mean MAE: {np.mean(cv_results['mae']):.4f} ± {np.std(cv_results['mae']):.4f}")
print(f"Mean RMSE: {np.mean(cv_results['rmse']):.4f} ± {np.std(cv_results['rmse']):.4f}")
print(f"Mean R2: {np.mean(cv_results['r2']):.4f} ± {np.std(cv_results['r2']):.4f}")
