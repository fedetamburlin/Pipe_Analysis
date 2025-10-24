# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# FILE: train_models.py
# OBIETTIVO: Addestrare i modelli GPR (DR & AR) e MLP (AP) e salvarli in ./Models_APP/
# ESEGUIRE QUESTO SCRIPT UNA SOLA VOLTA.
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# Libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
import warnings
from sklearn.exceptions import ConvergenceWarning
import copy
import joblib
import os
import random

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


#-----------------------------------------------------------------------------------------------------

print("Avvio script di addestramento e salvataggio modelli...")

# Ignora le avvertenze di convergenza del GPR
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Number of curvatures
n_c = 4

# LETTURA DATI COMUNI
try:
    data = pd.read_csv('AR_DR_results.csv')
except FileNotFoundError:
    print("ERRORE: File 'AR_DR_results.csv' non trovato.")
    exit()

targets = [col for col in data.columns if '(%)' in col] # ['AR (%)', 'DR (%)']
features = [col for col in data.columns if col not in targets]

X_data = data[features].to_numpy()
Y_data = data[targets].to_numpy() # Colonna 0: AR, Colonna 1: DR
n_run = len(X_data[:,1])

min_curv = np.min(X_data[:, -1])
max_curv = np.max(X_data[:, -1])
bounds_curv_tuple = (round(min_curv, 3), round(max_curv, 3))


# ==============================================================================
# Diameter Reduction - DR - SETUP E ADDESTRAMENTO GPR 
# ==============================================================================

print("Inizio Setup GPR (Modello 1 - DR)...")
num_samples_gpr = X_data.shape[0] // n_c
params_gpr_plot_axis = np.zeros((num_samples_gpr, n_c+1))
dr = np.zeros((num_samples_gpr, n_c+1))
ar = np.zeros((num_samples_gpr, n_c+1)) 

for ii in range(num_samples_gpr):
    dr[ii, 1:] = Y_data[ii * n_c:(ii+1) * n_c, 1] 
    ar[ii, 1:] = Y_data[ii * n_c:(ii+1) * n_c, 0] 
    params_gpr_plot_axis[ii, 1:] = X_data[ii * n_c:(ii+1) * n_c, -1] 

# Scaler comuni per input scalari GPR (DR e AR)
scaler_scalari_gpr = StandardScaler()
train_scalari_gpr = X_data[::n_c, :-1] # Rimuove solo la curvatura
scaler_scalari_gpr.fit(train_scalari_gpr)
train_scalari_gpr_norm = scaler_scalari_gpr.transform(train_scalari_gpr)

# Scaling e Training GPR per DR
scaler_output_dr = StandardScaler()
train_output_dr = dr

# Trasformazione logaritmica 
#train_dr_log = np.log(train_output_dr + 1)
train_dr_log = np.log1p(train_output_dr)

scaler_output_dr.fit(train_dr_log)
train_dr_final = scaler_output_dr.transform(train_dr_log)

# Kernel e modello GPR per DR 
kernel_dr = ConstantKernel(1.0) * (RBF(length_scale_bounds=(1e-05, 1e6)) + Matern(length_scale_bounds=(1e-05, 1e6), nu=1.5))
gaussian_process_dr = GaussianProcessRegressor(kernel=kernel_dr, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=80)

# Training
print("Addestramento del GPR (Modello 1 - DR) in corso...")
gaussian_process_dr.fit(train_scalari_gpr_norm, train_dr_final)
print("Modello GPR per DR addestrato.")


# ==============================================================================
# Area Reduction - AR - SETUP E ADDESTRAMENTO GPR 
# ==============================================================================

print("Inizio Setup GPR (Modello 1bis - AR)...")

# Scaling GPR per AR
scaler_output_ar = StandardScaler() 
train_output_ar = ar               

# Applica trasformazione logaritmica (analoga a DR)
train_ar_log = np.log(train_output_ar + 1) 
scaler_output_ar.fit(train_ar_log)         
train_ar_final = scaler_output_ar.transform(train_ar_log) 

# Kernel e modello GPR per AR
kernel_ar = ConstantKernel(1.0) * (RBF(length_scale_bounds=(1e-05, 1e6)) + Matern(length_scale_bounds=(1e-05, 1e6), nu=1.5))
gaussian_process_ar = GaussianProcessRegressor(kernel=kernel_ar, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=80) 

# Training
print("Addestramento del GPR (Modello 1bis - AR) in corso...")
# Usa lo stesso input normalizzato di DR, ma target AR finali
gaussian_process_ar.fit(train_scalari_gpr_norm, train_ar_final) 
print("Modello GPR per AR addestrato.")


# ==============================================================================
# Area Profile - AP - SETUP E ADDESTRAMENTO MLP
# ==============================================================================
print("Inizio Setup MLP (Modello 2 - AP)...")
def_A = []
pipe_abs = None

for ii in range(1, n_run+1):
    try:
        mypathfile = f'postProc_out/detailed_tables/detailed_run_{str(ii).zfill(4)}_results.csv'
        datafrm = pd.read_csv(mypathfile, sep=",", decimal='.')
        dati = datafrm.values
        if pipe_abs is None:
             pipe_abs = dati[:,0]
        def_A.append(dati[:,2])
    except FileNotFoundError:
        if pipe_abs is not None:
             def_A.append(np.zeros_like(pipe_abs))
        else:
             pass # Se non trovo neanche un file dettagliato e pipe_abs è ancora None

if pipe_abs is None:
    print("ERRORE: Nessun file 'detailed_run_*.csv' trovato. Impossibile addestrare MLP.")
    exit()

a_profiles = np.array(def_A)
num_abs = pipe_abs.shape[0]
ap_curves_flat = a_profiles
X_scalar_flat = X_data # Usiamo X_data originale (include la curvatura)

num_total_curves = min(X_scalar_flat.shape[0], a_profiles.shape[0])
if num_total_curves < X_scalar_flat.shape[0]:
    print(f"Attenzione: Trovati {num_total_curves} profili, ma {X_scalar_flat.shape[0]} run scalari. Uso {num_total_curves} run.")
    X_scalar_flat = X_scalar_flat[:num_total_curves]
    ap_curves_flat = ap_curves_flat[:num_total_curves]

num_total_points = num_total_curves * num_abs
input_features_count = X_scalar_flat.shape[1] + 1 # Features scalari + ascissa
X_long = np.zeros((num_total_points, input_features_count))
y_long = np.zeros(num_total_points)

current_row = 0
for i in range(num_total_curves):
    scalar_features = X_scalar_flat[i, :]
    features_matrix = np.hstack([
        np.tile(scalar_features, (num_abs, 1)),
        pipe_abs.reshape(-1, 1) # Aggiunge l'ascissa come feature
    ])
    X_long[current_row : current_row + num_abs] = features_matrix
    y_long[current_row : current_row + num_abs] = ap_curves_flat[i, :]
    current_row += num_abs

# Split Dati MLP
num_scenarios_in_test = 0  # non ci servono test in questa fase
num_curves_in_test = num_scenarios_in_test * n_c
num_curves_in_test = min(num_curves_in_test, num_total_curves - n_c)
if num_curves_in_test <= 0: num_curves_in_test = n_c
if num_total_curves <= num_curves_in_test:
    print("ERRORE: Dati insufficienti per addestrare l'MLP.")
    exit()

test_curve_indices = np.arange(num_curves_in_test)
train_val_curve_indices = np.arange(num_curves_in_test, num_total_curves)
train_val_mask = np.isin(np.floor(np.arange(num_total_points) / num_abs), train_val_curve_indices)
X_train_val, y_train_val = X_long[train_val_mask], y_long[train_val_mask]
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Scaling e Preparazione Dati MLP
scaler_mlp = StandardScaler()
X_train_scaled = scaler_mlp.fit_transform(X_train)
X_val_scaled = scaler_mlp.transform(X_val)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

# Definizione Classe MLP
class ProfileMLP(nn.Module):
    def __init__(self, input_features):
        super(ProfileMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

# Addestramento MLP 
model_mlp = ProfileMLP(input_features=input_features_count)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_mlp.parameters(), lr=0.004)
epochs = 150
best_val_loss = float('inf')
best_model_state_mlp = None

print(f"Addestramento dell'MLP (Modello 2 - AP) per {epochs} epoche...")
for epoch in range(epochs):
    model_mlp.train()
    epoch_train_loss = 0.0
    for inputs, labels in train_loader:
        outputs = model_mlp(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    model_mlp.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for inputs_val, labels_val in val_loader:
            outputs_val = model_mlp(inputs_val)
            epoch_val_loss += criterion(outputs_val, labels_val).item()

    current_val_loss = epoch_val_loss / len(val_loader)

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_model_state_mlp = copy.deepcopy(model_mlp.state_dict())

    if (epoch + 1) % 50 == 0:
        print(f'MLP Epoca [{epoch+1}/{epochs}], Val Loss: {current_val_loss:.4f}')

print("Modello MLP addestrato.")


# ==============================================================================
# SALVATAGGIO DEI MODELLI
# ==============================================================================

save_dir = "Models_APP"
os.makedirs(save_dir, exist_ok=True)
print(f"\nSalvataggio degli artefatti su disco in '{save_dir}'...")

# GPR (DR)
joblib.dump(gaussian_process_dr, os.path.join(save_dir, 'gpr_model_dr.joblib')) 
joblib.dump(scaler_scalari_gpr, os.path.join(save_dir, 'scaler_gpr_input.joblib')) 
joblib.dump(scaler_output_dr, os.path.join(save_dir, 'scaler_gpr_output_dr.joblib')) 
joblib.dump(params_gpr_plot_axis, os.path.join(save_dir, 'gpr_plot_params.joblib')) 

# GPR (AR) 
joblib.dump(gaussian_process_ar, os.path.join(save_dir, 'gpr_model_ar.joblib')) 
joblib.dump(scaler_output_ar, os.path.join(save_dir, 'scaler_gpr_output_ar.joblib')) 

# MLP (AP)
torch.save(best_model_state_mlp, os.path.join(save_dir, 'mlp_model_ap.pth'))
joblib.dump(scaler_mlp, os.path.join(save_dir, 'scaler_mlp_ap.joblib'))     
joblib.dump(pipe_abs, os.path.join(save_dir, 'pipe_abs_ap.joblib'))         

# Salvataggi Comuni
joblib.dump(bounds_curv_tuple, os.path.join(save_dir, 'bounds_curv.joblib'))

# Print fine addestramento
print("="*30)
print("OPERAZIONE COMPLETATA.")
print(f"Modelli e scaler salvati in '{save_dir}'.")
print("Ora puoi eseguire l'app per lanciare la dashboard.")
print("="*30)