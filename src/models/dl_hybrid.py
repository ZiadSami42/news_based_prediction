import duckdb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

DB_PATH = 'data/prediction.db'
PLOTS_DIR = 'plots'
ASSET_ORDER = ['NVDA', 'USO', 'EGX']
SEQ_LEN = 30
EPOCHS = 30
BATCH_SIZE = 32

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class VolatilityLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(VolatilityLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

def create_sequences(data, target, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model(model, train_loader, val_loader, epochs, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.squeeze())
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
    return train_losses, val_losses

def run_dl_models():
    con = duckdb.connect(DB_PATH)
    df = con.execute("SELECT * FROM model_features ORDER BY trading_session").df()
    
    target_col = 'realized_vol_22d'
    
    features_base = ['log_return_1d', 'log_return_3d', 'log_return_5d', 'vol_roc_1d']
    features_hybrid = features_base + ['garch_residual', 'gemma_dispersion', 'sentiment_persistence_3d', 'article_impact_index']
    
    for col in features_hybrid:
        if col not in df.columns:
            df[col] = 0
            
    all_predictions = []
    results = []
    
    for asset in ASSET_ORDER:
        print(f"\n--- Training Deep Learning Models for {asset} ---")
        asset_df = df[df['asset_tag'] == asset].copy()
        asset_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        asset_df = asset_df.dropna(subset=[target_col] + features_hybrid)
        
        if len(asset_df) < SEQ_LEN + 10:
            print(f"Not enough data for {asset}")
            continue
            
        scaler_X_base = StandardScaler()
        scaler_X_hybrid = StandardScaler()
        scaler_y = StandardScaler()
        
        X_base_scaled = scaler_X_base.fit_transform(asset_df[features_base])
        X_hyb_scaled = scaler_X_hybrid.fit_transform(asset_df[features_hybrid])
        y_scaled = scaler_y.fit_transform(asset_df[[target_col]])
        
        # Split chronological (80% / 20%)
        # Note: sequences take SEQ_LEN from the start, so the test set logic is:
        # test set starts at train_size index.
        train_size = int(len(asset_df) * 0.8)
        
        X_seq_base, y_seq = create_sequences(X_base_scaled, y_scaled, SEQ_LEN)
        
        X_train_base = X_seq_base[:train_size-SEQ_LEN]
        y_train = y_seq[:train_size-SEQ_LEN]
        X_test_base = X_seq_base[train_size-SEQ_LEN:]
        y_test = y_seq[train_size-SEQ_LEN:]
        
        train_dataset_base = TimeSeriesDataset(X_train_base, y_train)
        test_dataset_base = TimeSeriesDataset(X_test_base, y_test)
        
        train_loader_base = DataLoader(train_dataset_base, batch_size=BATCH_SIZE, shuffle=False)
        test_loader_base = DataLoader(test_dataset_base, batch_size=BATCH_SIZE, shuffle=False)
        
        lstm_base = VolatilityLSTM(input_dim=len(features_base))
        train_losses_b, val_losses_b = train_model(lstm_base, train_loader_base, test_loader_base, EPOCHS)
        
        lstm_base.eval()
        with torch.no_grad():
            preds_base = lstm_base(torch.tensor(X_test_base, dtype=torch.float32)).numpy()
            
        preds_base_unscaled = scaler_y.inverse_transform(preds_base.reshape(-1, 1)).flatten()
        y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        rmse_b = np.sqrt(mean_squared_error(y_test_unscaled, preds_base_unscaled))
        mae_b = mean_absolute_error(y_test_unscaled, preds_base_unscaled)
        results.append({'Asset': asset, 'Model': 'Standalone LSTM', 'RMSE': rmse_b, 'MAE': mae_b})
        
        # Model 2: GARCH-LSTM Hybrid
        X_seq_hyb, _ = create_sequences(X_hyb_scaled, y_scaled, SEQ_LEN)
        
        X_train_hyb = X_seq_hyb[:train_size-SEQ_LEN]
        X_test_hyb = X_seq_hyb[train_size-SEQ_LEN:]
        
        train_dataset_hyb = TimeSeriesDataset(X_train_hyb, y_train)
        test_dataset_hyb = TimeSeriesDataset(X_test_hyb, y_test)
        
        train_loader_hyb = DataLoader(train_dataset_hyb, batch_size=BATCH_SIZE, shuffle=False)
        test_loader_hyb = DataLoader(test_dataset_hyb, batch_size=BATCH_SIZE, shuffle=False)
        
        lstm_hyb = VolatilityLSTM(input_dim=len(features_hybrid))
        train_losses_h, val_losses_h = train_model(lstm_hyb, train_loader_hyb, test_loader_hyb, EPOCHS)
        
        lstm_hyb.eval()
        with torch.no_grad():
            preds_hyb = lstm_hyb(torch.tensor(X_test_hyb, dtype=torch.float32)).numpy()
            
        preds_hyb_unscaled = scaler_y.inverse_transform(preds_hyb.reshape(-1, 1)).flatten()
        
        rmse_h = np.sqrt(mean_squared_error(y_test_unscaled, preds_hyb_unscaled))
        mae_h = mean_absolute_error(y_test_unscaled, preds_hyb_unscaled)
        results.append({'Asset': asset, 'Model': 'GARCH-LSTM Hybrid', 'RMSE': rmse_h, 'MAE': mae_h})
        
        # Plot Training Curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses_b, label='Train Loss')
        plt.plot(val_losses_b, label='Val Loss')
        plt.title(f'Standalone LSTM Loss ({asset})')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_losses_h, label='Train Loss')
        plt.plot(val_losses_h, label='Val Loss')
        plt.title(f'GARCH-LSTM Hybrid Loss ({asset})')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/dl_loss_curves_{asset.lower()}.png')
        plt.close()
        
        # Store predictions
        asset_test_df = asset_df.iloc[train_size:].copy()
        # Truncate to align with predictions if len(preds) < len(asset_test_df)
        asset_test_df = asset_test_df.iloc[:len(preds_base_unscaled)]
        asset_test_df['lstm_base_pred'] = preds_base_unscaled
        asset_test_df['lstm_hybrid_pred'] = preds_hyb_unscaled
        all_predictions.append(asset_test_df)
        
    res_df = pd.DataFrame(results)
    print("\n--- Deep Learning Models Evaluation Summary ---")
    print(res_df.to_string(index=False))
    
    os.makedirs('artifacts', exist_ok=True)
    res_df.to_csv('artifacts/dl_results.csv', index=False)
    
    final_preds = pd.concat(all_predictions, ignore_index=True)
    pred_cols = ['trading_session', 'asset_tag', 'close', target_col, 'lstm_base_pred', 'lstm_hybrid_pred']
    eval_df = final_preds[pred_cols]
    
    con.execute("DROP TABLE IF EXISTS dl_predictions")
    con.execute("CREATE TABLE dl_predictions AS SELECT * FROM eval_df")
    print(f"\nSaved {len(eval_df)} predictions to dl_predictions table.")
    con.close()

if __name__ == '__main__':
    os.makedirs(PLOTS_DIR, exist_ok=True)
    run_dl_models()
