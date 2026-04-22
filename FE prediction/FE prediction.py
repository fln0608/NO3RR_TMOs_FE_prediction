import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    
class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 214), nn.ReLU(), 
            nn.Linear(214, 248), nn.ReLU(),
            nn.Linear(248, 1), nn.Sigmoid()
        )
    
    
    def forward(self, x):
        y = self.net(x)
        return y

preprocessors = []
models = []

for fold in range(10):
    preprocessor = joblib.load(f'My_saved_pipelines\\preprocessor_fold{fold}.pkl')
    preprocessors.append(preprocessor)
    
    model = Model(20).to(device)
    model.load_state_dict(torch.load(f'My_saved_pipelines\\model_fold{fold}.pth', weights_only=True))
    model.eval()
    models.append(model)

# ==================== FE value prediction ====================

input_features = np.array([
    29,  # 0.Atomic number of the metal in metal oxides
    2,    # 1.Valence state of metal ions
    29,   # 2.Atomic number of modified element
    0,  # 3.Mass fraction of modified element
    29,  # 4.Atomic number of the supporter atom
    0.1,  # 5.Nitrate concentration
    14,   # 6.Electrolyte pH
    -0.3, # 7.Applied potential
    1,    # 8.Modification method
    # Shape of the supporter (9-12)：
    0, 0, 0, 1,  # porous foam, mesh, fiber, plate/foil
    # Catalyst morphology (13-19)：
    0, 0, 1, 0, 0, 0, 0  # nanoparticle, nanosphere, nanowire, nanorod, nanotube, nanobelt, nanosheet
], dtype=np.float32)


predictions = []
for fold in range(10):
    processed = preprocessors[fold].transform(input_features.reshape(1, -1))    
    tensor_data = torch.from_numpy(processed).float().to(device)
    
    with torch.no_grad():
        pred = models[fold](tensor_data).cpu().numpy()
        predictions.append(pred.item())


mean_pred = np.mean(predictions)
std_pred = np.std(predictions)
print(f"Predicted FE for each fold: {predictions}")
print(f"Predicted FE: {mean_pred:.4f} ± {std_pred:.4f}")
