# ----------------------------------------------------
# Mooney Prediction Pipeline V2.0 Path Bootstrap
# ----------------------------------------------------
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(PARENT_DIR)
sys.path.extend([
    PARENT_DIR,
    os.path.join(PARENT_DIR, 'data_processing'),
    os.path.join(PARENT_DIR, 'model_training'),
    os.path.join(PARENT_DIR, 'model_analysis'),
])
# ----------------------------------------------------

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Directories
DATASET_PATH = "scratch/neural_network_dataset.joblib"
OUTPUT_DIR = os.path.join(PARENT_DIR, "models", "results_nn_ae")

# Configure plotting style
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ================== PyTorch Datasets ==================

class MooneyRecipeDataset(Dataset):
    def __init__(self, X_curves, X_recipe, y):
        self.X_curves = torch.tensor(X_curves, dtype=torch.float32)
        self.X_recipe = torch.tensor(X_recipe, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X_curves[idx], self.X_recipe[idx], self.y[idx]

# ================== PyTorch Models ==================

# 1. Autoencoder (trained on curves only)
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# 2. Direct MLP (takes curves + recipe weights concatenated)
class DirectMLP(nn.Module):
    def __init__(self, input_dim, recipe_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + recipe_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, recipe):
        combined = torch.cat([x, recipe], dim=1)
        return self.net(combined)

# 3. 1D-CNN (CNN for curves + Concat recipe weights in FC)
class CNN1D(nn.Module):
    def __init__(self, in_channels=5, length=120, recipe_dim=9):
        super().__init__()
        self.length = length
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # FC layer takes curves pooled features (64) + static recipe features (recipe_dim)
        self.fc = nn.Sequential(
            nn.Linear(64 + recipe_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, recipe):
        x = x.view(x.size(0), 5, self.length)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.avg_pool(out).squeeze(-1)
        
        # Concatenate static recipe features
        combined = torch.cat([out, recipe], dim=1)
        return self.fc(combined)

# 4. ResNet1D Block & Network
class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.1)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, length=120, recipe_dim=9):
        super().__init__()
        self.length = length
        self.in_conv = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        
        # ResNet Blocks
        self.block1 = ResNetBlock1D(32, 32, stride=1)
        self.block2 = ResNetBlock1D(32, 64, stride=2)
        self.block3 = ResNetBlock1D(64, 128, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # FC layer takes curves pooled features (128) + static recipe features (recipe_dim)
        self.fc = nn.Linear(128 + recipe_dim, 64)
        self.fc_out = nn.Linear(64, out_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, recipe):
        x = x.view(x.size(0), 5, self.length)
        out = self.relu(self.bn(self.in_conv(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.avg_pool(out).squeeze(-1)
        
        # Concatenate static recipe features
        combined = torch.cat([out, recipe], dim=1)
        out = self.relu(self.fc(combined))
        out = self.dropout(out)
        out = self.fc_out(out)
        return out

# 5. MLP on Autoencoder Bottleneck + Recipe Features
class BottleneckMLP(nn.Module):
    def __init__(self, latent_dim=32, recipe_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + recipe_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, recipe):
        combined = torch.cat([x, recipe], dim=1)
        return self.net(combined)


# ================== Training Functions ==================

def train_autoencoder(model, train_loader, val_loader, epochs=50, lr=1e-3, save_path=None):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_curves_batch, _, _ in train_loader:
            X_curves_batch = X_curves_batch.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(X_curves_batch)
            loss = criterion(reconstructed, X_curves_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_curves_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_curves_val, _, _ in val_loader:
                X_curves_val = X_curves_val.to(device)
                reconstructed, _ = model(X_curves_val)
                loss = criterion(reconstructed, X_curves_val)
                val_loss += loss.item() * X_curves_val.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
                
    return train_losses, val_losses

def train_regressor(model, train_loader, val_loader, encoder=None, epochs=100, lr=1e-3, save_path=None):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    criterion = nn.HuberLoss(delta=1.5)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_curves_batch, X_recipe_batch, y_batch in train_loader:
            X_curves_batch = X_curves_batch.to(device)
            X_recipe_batch = X_recipe_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            if encoder is not None:
                encoder.eval()
                with torch.no_grad():
                    _, X_curves_batch = encoder(X_curves_batch)
                    
            pred = model(X_curves_batch, X_recipe_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y_batch.size(0)
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_curves_val, X_recipe_val, y_val in val_loader:
                X_curves_val = X_curves_val.to(device)
                X_recipe_val = X_recipe_val.to(device)
                y_val = y_val.to(device)
                
                if encoder is not None:
                    _, X_curves_val = encoder(X_curves_val)
                pred = model(X_curves_val, X_recipe_val)
                loss = criterion(pred, y_val)
                val_loss += loss.item() * y_val.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
                
    return best_loss


def evaluate_model(model, X_curves_test, X_recipe_test, y_test, encoder=None):
    model.eval()
    dataset = MooneyRecipeDataset(X_curves_test, X_recipe_test, y_test)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    preds = []
    with torch.no_grad():
        for X_curves_batch, X_recipe_batch, _ in loader:
            X_curves_batch = X_curves_batch.to(device)
            X_recipe_batch = X_recipe_batch.to(device)
            if encoder is not None:
                _, X_curves_batch = encoder(X_curves_batch)
            pred = model(X_curves_batch, X_recipe_batch)
            preds.extend(pred.cpu().squeeze(-1).numpy())
            
    preds = np.array(preds)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    return mae, rmse, r2, preds, y_test

def compute_permutation_importance(model, X_curves, X_recipe, y, mae_baseline, encoder=None, num_stages=6):
    """Compute permutation feature importance for the multi-modal neural network model"""
    model.eval()
    
    feature_groups = {
        'weight_pct_solid_elastomer': ('recipe', 0),
        'weight_pct_natural_rubber': ('recipe', 1),
        'weight_pct_silica': ('recipe', 2),
        'weight_pct_oil': ('recipe', 3),
        'weight_pct_silian': ('recipe', 4),
        'weight_pct_carbon_black': ('recipe', 5),
        'supplier_rubber_viscosity_avg': ('recipe', 6),
        'supplier_silica_moisture_avg': ('recipe', 7),
        'supplier_silica_surface_area_avg': ('recipe', 8),
        'supplier_carbon_black_structure_avg': ('recipe', 9),
        'supplier_carbon_black_surface_area_avg': ('recipe', 10),
        'supplier_carbon_black_moisture_avg': ('recipe', 11),
        'Top_Fill_Factor': ('recipe', 12),
        'Bot_Fill_Factor': ('recipe', 13),
        'Mixing Curve: Temperature': ('curve', 0),
        'Mixing Curve: Power': ('curve', 1),
        'Mixing Curve: Torque': ('curve', 2),
        'Mixing Curve: RotorSpeed': ('curve', 3),
        'Mixing Curve: WayofRam': ('curve', 4)
    }
    
    importances = {}
    
    for name, (type_val, col_idx) in feature_groups.items():
        # Copy original test matrices
        X_curves_shuffled = X_curves.copy()
        X_recipe_shuffled = X_recipe.copy()
        
        if type_val == 'recipe':
            # Shuffle the single static recipe column
            np.random.shuffle(X_recipe_shuffled[:, col_idx])
        elif type_val == 'curve':
            # Shuffle the curve points of a specific variable across all stages together
            # Reshape curves to (Batch, Stages, Variables, K_POINTS) to shuffle easily
            batch_size = len(X_curves)
            curves_reshaped = X_curves_shuffled.reshape(batch_size, num_stages, 5, 20)
            
            # Extract indices to shuffle
            shuffled_indices = np.random.permutation(batch_size)
            curves_reshaped[:, :, col_idx, :] = curves_reshaped[shuffled_indices, :, col_idx, :]
            
            X_curves_shuffled = curves_reshaped.reshape(batch_size, -1)
            
        # Evaluate model with shuffled features
        mae_shuffled, _, _, _, _ = evaluate_model(model, X_curves_shuffled, X_recipe_shuffled, y, encoder=encoder)
        importance_score = mae_shuffled - mae_baseline
        importances[name] = max(0.0, importance_score)
        
    return importances


# ================== Main Process ==================

def run_track(track_name, batches, input_dim, length):
    print(f"\n==================== TRAINING FOR TRACK (M1 ONLY): {track_name} ====================")
    
    # 1. Create directory
    track_dir = os.path.join(OUTPUT_DIR, track_name.lower().replace('-', '_'))
    os.makedirs(track_dir, exist_ok=True)
    
    # 2. Extract features, recipe weights, and absolute labels (MNY)
    X_curves = np.array([b['features'] for b in batches])
    X_recipe = np.array([b['recipe_features'] for b in batches])
    y = np.array([b['MNY'] for b in batches])
    recipe_dim = X_recipe.shape[1]
    
    # 3. Train-Test Split (80% / 20%)
    indices = np.arange(len(batches))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_curves_train, X_curves_test = X_curves[train_idx], X_curves[test_idx]
    X_recipe_train, X_recipe_test = X_recipe[train_idx], X_recipe[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 4. Standardize curve and recipe features separately
    curves_scaler = StandardScaler()
    X_curves_train_scaled = curves_scaler.fit_transform(X_curves_train)
    X_curves_test_scaled = curves_scaler.transform(X_curves_test)
    
    recipe_scaler = StandardScaler()
    X_recipe_train_scaled = recipe_scaler.fit_transform(X_recipe_train)
    X_recipe_test_scaled = recipe_scaler.transform(X_recipe_test)
    
    # Save the scalers
    joblib.dump(curves_scaler, os.path.join(track_dir, 'curves_scaler.joblib'))
    joblib.dump(recipe_scaler, os.path.join(track_dir, 'recipe_scaler.joblib'))
    
    # Split train further into train_train and val (10% validation) for PyTorch early stopping
    tt_idx, val_idx = train_test_split(np.arange(len(X_curves_train_scaled)), test_size=0.1, random_state=42)
    
    X_curves_train_t = X_curves_train_scaled[tt_idx]
    X_curves_val = X_curves_train_scaled[val_idx]
    
    X_recipe_train_t = X_recipe_train_scaled[tt_idx]
    X_recipe_val = X_recipe_train_scaled[val_idx]
    
    y_train_t = y_train[tt_idx]
    y_val = y_train[val_idx]
    
    # 5. Create Data Loaders
    train_dataset = MooneyRecipeDataset(X_curves_train_t, X_recipe_train_t, y_train_t)
    val_dataset = MooneyRecipeDataset(X_curves_val, X_recipe_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # A. Train Autoencoder (trained on curves only)
    ae_model = Autoencoder(input_dim=input_dim, latent_dim=32).to(device)
    ae_save_path = os.path.join(track_dir, 'autoencoder.pth')
    
    print(f"Training Autoencoder for {track_name}...")
    ae_train_loss, ae_val_loss = train_autoencoder(
        ae_model, train_loader, val_loader, epochs=60, lr=1e-3, save_path=ae_save_path
    )
    # Load best Autoencoder weights
    ae_model.load_state_dict(torch.load(ae_save_path))
    
    # Plot AE training history
    plt.figure(figsize=(8, 5))
    plt.plot(ae_train_loss, label='Train MSE')
    plt.plot(ae_val_loss, label='Val MSE')
    plt.title(f'Autoencoder Reconstruction Loss ({track_name})')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(track_dir, 'autoencoder_loss.png'))
    plt.close()
    
    # B. Train Direct MLP
    mlp_model = DirectMLP(input_dim=input_dim, recipe_dim=recipe_dim).to(device)
    mlp_save_path = os.path.join(track_dir, 'mlp.pth')
    print(f"Training Direct MLP for {track_name}...")
    train_regressor(mlp_model, train_loader, val_loader, epochs=100, lr=1e-3, save_path=mlp_save_path)
    mlp_model.load_state_dict(torch.load(mlp_save_path))
    
    # C. Train 1D-CNN (multi-modal: curves convolution + recipe concat)
    cnn_model = CNN1D(in_channels=5, length=length, recipe_dim=recipe_dim).to(device)
    cnn_save_path = os.path.join(track_dir, 'cnn.pth')
    print(f"Training 1D-CNN for {track_name}...")
    train_regressor(cnn_model, train_loader, val_loader, epochs=100, lr=1e-3, save_path=cnn_save_path)
    cnn_model.load_state_dict(torch.load(cnn_save_path))
    
    # D. Train ResNet1D (multi-modal: curves ResNet + recipe concat)
    resnet_model = ResNet1D(in_channels=5, length=length, recipe_dim=recipe_dim).to(device)
    resnet_save_path = os.path.join(track_dir, 'resnet.pth')
    print(f"Training ResNet1D for {track_name}...")
    train_regressor(resnet_model, train_loader, val_loader, epochs=100, lr=1e-3, save_path=resnet_save_path)
    resnet_model.load_state_dict(torch.load(resnet_save_path))
    
    # E. Train Autoencoder + MLP Regressor (latent 32 + recipe 14 = 46 features)
    ae_mlp_model = BottleneckMLP(latent_dim=32, recipe_dim=recipe_dim).to(device)
    ae_mlp_save_path = os.path.join(track_dir, 'ae_mlp.pth')
    print(f"Training Autoencoder + MLP for {track_name}...")
    train_regressor(ae_mlp_model, train_loader, val_loader, encoder=ae_model, epochs=100, lr=1e-3, save_path=ae_mlp_save_path)
    ae_mlp_model.load_state_dict(torch.load(ae_mlp_save_path))
    
    # 6. Evaluate on Test Set
    results = {}
    
    models = {
        'Direct MLP': (mlp_model, None),
        '1D-CNN': (cnn_model, None),
        'ResNet1D': (resnet_model, None),
        'Autoencoder + MLP': (ae_mlp_model, ae_model)
    }
    
    # Plot parity graphs for direct MNY comparison
    plt.figure(figsize=(12, 10))
    
    plot_idx = 1
    best_model_name = 'Direct MLP'
    best_mae = float('inf')
    
    for name, (model, enc) in models.items():
        mae, rmse, r2, preds, actuals = evaluate_model(
            model, X_curves_test_scaled, X_recipe_test_scaled, y_test, encoder=enc
        )
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"[{name}] Test Set (Absolute MNY): MAE = {mae:.3f} MNY, RMSE = {rmse:.3f} MNY, R2 = {r2:.3f}")
        
        if mae < best_mae:
            best_mae = mae
            best_model_name = name
        
        # Subplot for Parity
        plt.subplot(2, 2, plot_idx)
        plt.scatter(actuals, preds, alpha=0.5, color='forestgreen', edgecolors='k')
        lims = [
            np.min([plt.xlim()[0], plt.ylim()[0], actuals.min(), preds.min()]),
            np.max([plt.xlim()[1], plt.ylim()[1], actuals.max(), preds.max()])
        ]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=3)
        plt.title(f"{name} (MAE={mae:.2f}, R2={r2:.2f})")
        plt.xlabel("Actual MNY")
        plt.ylabel("Predicted MNY")
        plt.grid(True)
        plot_idx += 1
        
    plt.suptitle(f"Absolute MNY Neural Model Comparison (M1 Track: {track_name})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(track_dir, 'model_comparison_parity.png'))
    plt.close()
    
    # 7. Compute Permutation Feature Importance for the best MLP/joint model
    print(f"\nComputing Permutation Feature Importance using best model: {best_model_name}")
    best_m, best_enc = models[best_model_name]
    num_stages = 6 if track_name == "With-Oil" else 4
    
    importances = compute_permutation_importance(
        best_m, X_curves_test_scaled, X_recipe_test_scaled, y_test, best_mae, encoder=best_enc, num_stages=num_stages
    )
    
    # Sort and plot importance
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_importances]
    scores = [x[1] for x in sorted_importances]
    
    plt.figure(figsize=(10, 6))
    plt.barh(names[::-1], scores[::-1], color='darkorange', edgecolor='k', alpha=0.8)
    plt.title(f"Permutation Feature Importance ({best_model_name} on {track_name})")
    plt.xlabel("MAE Increase (MNY)")
    plt.grid(True, axis='x', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(track_dir, 'feature_importance.png'))
    plt.close()
    
    # Save importance to CSV
    imp_df = pd.DataFrame(sorted_importances, columns=['Feature', 'MAE_Increase'])
    imp_df.to_csv(os.path.join(track_dir, 'feature_importance.csv'), index=False)
    print(f"Feature importance saved to {track_dir}/feature_importance.csv")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(os.path.join(track_dir, 'test_metrics.csv'))
    print(f"Metrics saved to {track_dir}/test_metrics.csv")
    
    # Save predictions vs actuals for the test set
    test_preds_df = pd.DataFrame({'actual_MNY': y_test})
    for name, (model, enc) in models.items():
        _, _, _, preds, _ = evaluate_model(
            model, X_curves_test_scaled, X_recipe_test_scaled, y_test, encoder=enc
        )
        test_preds_df[f'predicted_MNY_{name.lower().replace(" ", "_").replace("+", "_")}'] = preds
    
    test_preds_path = os.path.join(track_dir, 'test_predictions_nn.csv')
    test_preds_df.to_csv(test_preds_path, index=False)
    print(f"Saved test predictions to {test_preds_path}")

    # Save a training set summary CSV
    train_summary = {
        'metric': ['count', 'mean', 'std', 'min', '50%', 'max'],
        'MNY': [len(y_train), np.mean(y_train), np.std(y_train), np.min(y_train), np.median(y_train), np.max(y_train)]
    }
    # Add recipe features statistics
    recipe_cols_list = [
        'weight_pct_solid_elastomer', 'weight_pct_natural_rubber', 'weight_pct_silica',
        'weight_pct_oil', 'weight_pct_silian', 'weight_pct_carbon_black',
        'supplier_rubber_viscosity_avg',
        'supplier_silica_moisture_avg', 'supplier_silica_surface_area_avg',
        'supplier_carbon_black_structure_avg', 'supplier_carbon_black_surface_area_avg',
        'supplier_carbon_black_moisture_avg',
        'Top_Fill_Factor', 'Bot_Fill_Factor'
    ]
    for idx_c, col_c in enumerate(recipe_cols_list):
        vals = X_recipe_train[:, idx_c]
        train_summary[col_c] = [
            len(vals), np.mean(vals), np.std(vals), np.min(vals), np.median(vals), np.max(vals)
        ]
    train_summary_df = pd.DataFrame(train_summary)
    train_summary_path = os.path.join(track_dir, 'train_dataset_summary.csv')
    train_summary_df.to_csv(train_summary_path, index=False)
    print(f"Saved training dataset summary to {train_summary_path}")

    return results

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: dataset file '{DATASET_PATH}' does not exist! Please run data processing/preprocess_raw_curves.py first.")
        return
        
    dataset = joblib.load(DATASET_PATH)
    print(f"Loaded {len(dataset)} preprocessed M1 batches.")
    
    with_oil_batches = [b for b in dataset if b['is_oil_loading_present'] == 1]
    without_oil_batches = [b for b in dataset if b['is_oil_loading_present'] == 0]
    
    print(f"With-Oil M1 batches: {len(with_oil_batches)}")
    print(f"Without-Oil M1 batches: {len(without_oil_batches)}")
    
    all_metrics = {}
    
    # Run With-Oil
    if len(with_oil_batches) > 0:
        metrics_with = run_track(
            track_name="With-Oil",
            batches=with_oil_batches,
            input_dim=600,
            length=120
        )
        all_metrics['With-Oil'] = metrics_with
        
    # Run Without-Oil
    if len(without_oil_batches) > 0:
        metrics_without = run_track(
            track_name="Without-Oil",
            batches=without_oil_batches,
            input_dim=400,
            length=80
        )
        all_metrics['Without-Oil'] = metrics_without
        
    # Print overall comparison summary
    print("\n==================== OVERALL NN & AUTOENCODER COMPARISON SUMMARY (M1 ONLY) ====================")
    for track, metrics in all_metrics.items():
        print(f"\nTrack: {track}")
        for model_name, m_dict in metrics.items():
            print(f"  - {model_name:18s}: MAE = {m_dict['MAE']:.3f} MNY, RMSE = {m_dict['RMSE']:.3f} MNY, R2 = {m_dict['R2']:.3f}")

if __name__ == '__main__':
    main()
