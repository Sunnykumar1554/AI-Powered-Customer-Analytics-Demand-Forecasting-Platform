"""
RetailPulse – Day 6: LSTM Forecasting
=======================================
Goal: PyTorch LSTM model for demand forecasting, 50 epochs training.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────
PROCESSED = os.path.join("..", "data", "processed")
MODELS    = os.path.join("..", "models")
REPORTS   = os.path.join("..", "reports", "day6")

# ══════════════════════════════════════════════════════════
# STEP 1 – Load Daily Sales
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading Daily Sales Data")
print("=" * 60)
daily_sales = pd.read_csv(os.path.join(PROCESSED, "day4", "daily_sales.csv"), parse_dates=["Date"])
daily_sales = daily_sales.sort_values("Date").reset_index(drop=True)

# Fill missing dates
full_range = pd.date_range(start=daily_sales["Date"].min(), end=daily_sales["Date"].max(), freq="D")
daily_sales = daily_sales.set_index("Date").reindex(full_range, fill_value=0).reset_index()
daily_sales.columns = ["Date", "Quantity", "Rolling_7d", "Rolling_30d"]
daily_sales["Rolling_7d"] = daily_sales["Quantity"].rolling(7, min_periods=1).mean()
daily_sales["Rolling_30d"] = daily_sales["Quantity"].rolling(30, min_periods=1).mean()

values = daily_sales["Quantity"].values.astype(float)
print(f"  Total data points: {len(values)}")

# ══════════════════════════════════════════════════════════
# STEP 2 – Normalize
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Normalizing Data (MinMaxScaler)")
print("=" * 60)
scaler = MinMaxScaler(feature_range=(0, 1))
values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
print(f"  Scaled range: [{values_scaled.min():.4f}, {values_scaled.max():.4f}]")

# Save scaler for later use
with open(os.path.join(MODELS, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ══════════════════════════════════════════════════════════
# STEP 3 – Create Sequences
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Creating Sequences (lookback=30)")
print("=" * 60)

LOOKBACK = 30

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(values_scaled, LOOKBACK)
print(f"  Total sequences: {len(X)}")

# Train/Test split (last 30 days as test)
test_size = 30
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]
print(f"  Training sequences: {len(X_train)}")
print(f"  Testing sequences:  {len(X_test)}")

# ══════════════════════════════════════════════════════════
# STEP 4 – PyTorch LSTM Model
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Building PyTorch LSTM Model")
print("=" * 60)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("  ⚠ PyTorch not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "torch"])
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).unsqueeze(-1).to(device)  # (N, 30, 1)
y_train_t = torch.FloatTensor(y_train).to(device)
X_test_t  = torch.FloatTensor(X_test).unsqueeze(-1).to(device)
y_test_t  = torch.FloatTensor(y_test).to(device)

# DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take last time step
        out = self.fc(out)
        return out.squeeze()


model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"  Model architecture:")
print(f"    LSTM: input=1, hidden=64, layers=2, dropout=0.2")
print(f"    Dense: 64 → 1")
total_params = sum(p.numel() for p in model.parameters())
print(f"    Total parameters: {total_params:,}")

# ══════════════════════════════════════════════════════════
# STEP 5 – Train LSTM
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Training LSTM (50 epochs)")
print("=" * 60)

EPOCHS = 50
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    n_batches = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    avg_train_loss = epoch_loss / n_batches
    train_losses.append(avg_train_loss)

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_t)
        val_loss = criterion(val_pred, y_test_t).item()
        val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch + 1:3d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f}")

# ══════════════════════════════════════════════════════════
# STEP 6 – Evaluate
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Evaluation")
print("=" * 60)

model.eval()
with torch.no_grad():
    predictions_scaled = model(X_test_t).cpu().numpy()
    actual_scaled = y_test_t.cpu().numpy()

# Inverse transform
predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
actual = scaler.inverse_transform(actual_scaled.reshape(-1, 1)).flatten()

# Metrics
mask = actual != 0
mape = np.mean(np.abs((actual[mask] - predictions[mask]) / actual[mask])) * 100
rmse = np.sqrt(np.mean((actual - predictions) ** 2))
mae = np.mean(np.abs(actual - predictions))

print(f"  MAPE: {mape:.2f}%")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE:  {mae:.2f}")

# Save metrics
metrics = pd.DataFrame({
    "Model": ["LSTM"],
    "MAPE": [round(mape, 2)],
    "RMSE": [round(rmse, 2)],
    "MAE": [round(mae, 2)]
})
os.makedirs(os.path.join(PROCESSED, "day6"), exist_ok=True)
metrics.to_csv(os.path.join(PROCESSED, "day6", "lstm_metrics.csv"), index=False)

# ══════════════════════════════════════════════════════════
# STEP 7 – Visualizations
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Visualizations")
print("=" * 60)

# Loss curve
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", color="#3498db", linewidth=2)
ax.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss", color="#e74c3c", linewidth=2)
ax.set_title("LSTM Training Loss Curve", fontsize=16, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("MSE Loss", fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "loss_curve.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day6/loss_curve.png")

# Actual vs Predicted
fig, ax = plt.subplots(figsize=(12, 5))
test_dates = daily_sales["Date"].iloc[-test_size:].values
ax.plot(test_dates, actual, label="Actual", color="#3498db", linewidth=2, marker="o", markersize=4)
ax.plot(test_dates, predictions, label="LSTM Predicted", color="#e74c3c", linewidth=2, marker="s", markersize=4)
ax.set_title(f"LSTM Forecast vs Actual (MAPE: {mape:.1f}%)", fontsize=16, fontweight="bold")
ax.set_xlabel("Date", fontsize=13)
ax.set_ylabel("Quantity", fontsize=13)
ax.legend(fontsize=11)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "lstm_forecast.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day6/lstm_forecast.png")

# ══════════════════════════════════════════════════════════
# STEP 8 – Save Model & Forecast
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Saving Model & Forecast")
print("=" * 60)

# Save PyTorch model
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_losses": train_losses,
    "val_losses": val_losses,
    "lookback": LOOKBACK,
    "hidden_size": 64,
    "num_layers": 2
}, os.path.join(MODELS, "lstm_model.pth"))
print("  ✓ Saved: models/lstm_model.pth")

# Save forecast
lstm_forecast = pd.DataFrame({
    "Date": test_dates,
    "Actual": actual,
    "Prediction": predictions
})
os.makedirs(os.path.join(PROCESSED, "day6"), exist_ok=True)
lstm_forecast.to_csv(os.path.join(PROCESSED, "day6", "lstm_forecast.csv"), index=False)
print("  ✓ Saved: data/processed/day6/lstm_forecast.csv")

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DAY 6 COMPLETE ✓")
print("=" * 60)
print(f"""
Deliverables:
  ✓ models/lstm_model.pth
  ✓ models/scaler.pkl
  ✓ data/processed/lstm_forecast.csv
  ✓ data/processed/lstm_metrics.csv
  ✓ reports/day6/loss_curve.png
  ✓ reports/day6/lstm_forecast.png

LSTM Architecture:
  Input  → LSTM(64, layers=2, dropout=0.2) → Dense(1)
  Epochs: {EPOCHS}, Batch Size: 32, Lookback: {LOOKBACK}

Metrics:
  MAPE: {mape:.2f}%
  RMSE: {rmse:.2f}
  MAE:  {mae:.2f}
""")
