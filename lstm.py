# -*- coding: ascii -*-
"""
lstm_precipitation_forecast.py  v3
------------------------------------
Core fix: LSTM now does same-day prediction (like XGBoost), not next-day
forecast.  Window covers rows [t-seq_len+1 .. t]; label = y[t].
This gives the model access to same-day correlated features (RSK, TMK, etc.)
exactly as XGBoost does, making the comparison fair.

Other improvements over v2:
  - Bidirectional LSTM (2x hidden states, better pattern extraction)
  - Multi-head self-attention over LSTM outputs
  - Larger hidden dim (256)
  - weights_only=True on torch.load (suppresses FutureWarning)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ==========================================================
# 0. CONFIG
# ==========================================================
CSV_PATH     = "dataset/joined_by_date.csv"
TARGET       = "Precipitation"
TEST_SIZE    = 0.20
SEQ_LEN      = 60
BATCH_SIZE   = 512
EPOCHS       = 100
LR           = 5e-4
HIDDEN_DIM   = 256
NUM_LAYERS   = 2
DROPOUT      = 0.3
WEIGHT_DECAY = 1e-4
PATIENCE     = 15
LOG1P_TARGET = True
N_HEADS      = 4           # attention heads

FIGSIZE      = (16, 9)
DPI          = 300
MONTHLY_ONLY = True
ZOOM_YEARS   = 15
CKPT_PATH    = "models/lstm_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================================
# 1. LOAD & PARSE
# ==========================================================
df = pd.read_csv(CSV_PATH)

if '__date__' in df.columns:
    df['Date_parsed'] = pd.to_datetime(df['__date__'])
else:
    df['Date_parsed'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

df = df.sort_values('Date_parsed').reset_index(drop=True)
print(f"Loaded {len(df)} rows | {df['Date_parsed'].min().date()} -> {df['Date_parsed'].max().date()}")

# ==========================================================
# 2. CLEAN
# ==========================================================
df = df.replace(-999.0, np.nan)

drop_cols = ['Date', '__date__', 'MESS_DATUM', 'STATIONS_ID', 'eor',
             'Date_parsed', 'QN_3', 'QN_4']
drop_cols = [c for c in drop_cols if c in df.columns]

feature_df    = df.drop(columns=[TARGET] + drop_cols)
feature_df    = feature_df.apply(pd.to_numeric, errors='coerce')
target_series = df[TARGET].clip(lower=0)

nan_pct  = feature_df.isna().mean().sort_values(ascending=False)
bad_cols = nan_pct[nan_pct > 0.40].index.tolist()
if bad_cols:
    print(f"Dropping high-NaN cols (>40%): {bad_cols}")
    feature_df = feature_df.drop(columns=bad_cols)

feature_df    = feature_df.ffill().bfill().fillna(0)
target_series = target_series.ffill().bfill().fillna(0)

# ==========================================================
# 3. CYCLICAL TIME FEATURES
# ==========================================================
dates = df['Date_parsed']
doy   = dates.dt.dayofyear.values
month = dates.dt.month.values
feature_df['sin_doy']   = np.sin(2 * np.pi * doy   / 365.25)
feature_df['cos_doy']   = np.cos(2 * np.pi * doy   / 365.25)
feature_df['sin_month'] = np.sin(2 * np.pi * month / 12.0)
feature_df['cos_month'] = np.cos(2 * np.pi * month / 12.0)

print(f"Feature matrix: {feature_df.shape}")
print(f"Features: {list(feature_df.columns)}")

X_raw = feature_df.values.astype(np.float32)
y_raw = target_series.values.astype(np.float32)

# ==========================================================
# 4. LOG1p TRANSFORM ON TARGET
# ==========================================================
if LOG1P_TARGET:
    y_model = np.log1p(y_raw)
    print(f"\nTarget BEFORE log1p: mean={y_raw.mean():.3f}, std={y_raw.std():.3f}, max={y_raw.max():.1f}")
    print(f"Target AFTER  log1p: mean={y_model.mean():.3f}, std={y_model.std():.3f}, max={y_model.max():.2f}")
else:
    y_model = y_raw.copy()

# ==========================================================
# 5. CHRONOLOGICAL SPLIT
# ==========================================================
split_idx  = int(len(df) * (1 - TEST_SIZE))
split_date = df['Date_parsed'].iloc[split_idx]
print(f"\nSplit at row {split_idx} | date: {split_date.date()}")

# ==========================================================
# 6. SCALE ON TRAIN ONLY
# ==========================================================
feat_scaler   = StandardScaler()
target_scaler = StandardScaler()
feat_scaler.fit(X_raw[:split_idx])
target_scaler.fit(y_model[:split_idx].reshape(-1, 1))

X_scaled = feat_scaler.transform(X_raw).astype(np.float32)
y_scaled = target_scaler.transform(y_model.reshape(-1, 1)).flatten().astype(np.float32)

print(f"X_scaled train mean={X_scaled[:split_idx].mean():.4f}, std={X_scaled[:split_idx].std():.4f}")
print(f"y_scaled train mean={y_scaled[:split_idx].mean():.4f}, std={y_scaled[:split_idx].std():.4f}")

# ==========================================================
# 7. BUILD SEQUENCES  --  SAME-DAY PREDICTION
# ----------------------------------------------------------
# WHY THIS MATTERS:
#   XGBoost sees features at day t and predicts Precipitation at day t.
#   (Same-day features like RSK, TMK, etc. are highly correlated with
#    same-day precipitation.)
#
#   Old LSTM was predicting y[t+1] from X[t-seq+1:t+1] -- a TRUE
#   next-day FORECAST, fundamentally harder & incomparable to XGBoost.
#
#   NEW: window covers [t-seq+1 .. t], label = y[t].
#   The LAST timestep in the sequence is the current day, so the model
#   sees today's correlated features, just like XGBoost.
# ==========================================================
def make_sequences(X, y, seq_len):
    """
    Window i -> X[i : i+seq_len],  label = y[i+seq_len-1]  (LAST day in window)
    Produces N - seq_len + 1 samples.
    """
    xs = np.lib.stride_tricks.sliding_window_view(X, (seq_len, X.shape[1]))
    xs = xs[:, 0, :, :]           # (N-seq_len+1, seq_len, n_features)
    ys = y[seq_len - 1:]          # (N-seq_len+1,)  label = last day of window
    assert len(xs) == len(ys), f"Shape mismatch xs={len(xs)} ys={len(ys)}"
    return xs.astype(np.float32), ys.astype(np.float32)

X_seq, y_seq  = make_sequences(X_scaled, y_scaled, SEQ_LEN)
dates_seq     = dates.iloc[SEQ_LEN - 1:].reset_index(drop=True)  # date of last step

# Train/test split: sequences whose label-day is before split_idx
adj_split     = split_idx - SEQ_LEN + 1
X_train, X_test = X_seq[:adj_split], X_seq[adj_split:]
y_train, y_test = y_seq[:adj_split], y_seq[adj_split:]

print(f"\nX_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"y_train: {y_train.shape} | y_test: {y_test.shape}")

# ==========================================================
# 8. DATALOADERS
# ==========================================================
train_ds     = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds      = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

# ==========================================================
# 9. MODEL  -- BiLSTM + Multi-Head Attention
# ==========================================================
class Attention(nn.Module):
    """Scaled dot-product multi-head self-attention over time axis."""
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads,
                                          dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)     # residual + norm


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, n_heads):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True          # 2x hidden_dim output
        )
        bidir_dim = hidden_dim * 2
        self.attn    = Attention(bidir_dim, n_heads)
        self.norm    = nn.LayerNorm(bidir_dim)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(bidir_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _  = self.lstm(x)       # (B, T, 2*H)
        attn_out     = self.attn(lstm_out) # (B, T, 2*H)
        # Global average pool + last step, concatenated
        pooled       = attn_out.mean(dim=1)          # (B, 2*H)
        last         = attn_out[:, -1, :]            # (B, 2*H)
        combined     = self.norm(pooled + last)      # residual sum
        combined     = self.dropout(combined)
        return self.head(combined).squeeze(-1)        # (B,)


n_features = X_seq.shape[2]
model = LSTMForecaster(n_features, HIDDEN_DIM, NUM_LAYERS, DROPOUT, N_HEADS).to(DEVICE)
print(f"\nModel:\n{model}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = nn.HuberLoss(delta=1.0)

# ==========================================================
# 10. TRAINING WITH EARLY STOPPING
# ==========================================================
best_val, no_improve = float('inf'), 0
train_losses, val_losses = [], []

print("\nTraining...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item() * len(xb)
    train_loss = running / len(train_ds)

    model.eval()
    running = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            running += criterion(model(xb), yb).item() * len(xb)
    val_loss = running / len(test_ds)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step()

    if val_loss < best_val:
        best_val, no_improve = val_loss, 0
        torch.save(model.state_dict(), CKPT_PATH)
    else:
        no_improve += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:>3}/{EPOCHS}  Train={train_loss:.5f}  Val={val_loss:.5f}"
              f"  Best={best_val:.5f}  Patience={no_improve}/{PATIENCE}")

    if no_improve >= PATIENCE:
        print(f"\nEarly stop at epoch {epoch}")
        break

model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
print(f"Loaded best checkpoint (val={best_val:.5f})")

# ==========================================================
# 11. FULL-HISTORY INFERENCE
# ==========================================================
model.eval()
chunks = []
with torch.no_grad():
    for i in range(0, len(X_seq), BATCH_SIZE):
        xb = torch.from_numpy(X_seq[i:i + BATCH_SIZE]).to(DEVICE)
        chunks.append(model(xb).cpu().numpy())
all_preds_scaled = np.concatenate(chunks)

# Inverse StandardScaler
all_preds_log   = target_scaler.inverse_transform(all_preds_scaled.reshape(-1, 1)).flatten()
all_actuals_log = target_scaler.inverse_transform(y_seq.reshape(-1, 1)).flatten()

# Inverse log1p
if LOG1P_TARGET:
    all_preds   = np.expm1(all_preds_log).clip(min=0)
    all_actuals = np.expm1(all_actuals_log).clip(min=0)
else:
    all_preds   = all_preds_log.clip(min=0)
    all_actuals = all_actuals_log.clip(min=0)

test_preds  = all_preds[adj_split:]
test_actual = all_actuals[adj_split:]

r2   = r2_score(test_actual, test_preds)
rmse = np.sqrt(mean_squared_error(test_actual, test_preds))
mae  = mean_absolute_error(test_actual, test_preds)
print(f"\nTest R^2={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")

# ==========================================================
# 12. BUILD PLOT DATAFRAME
# ==========================================================
plot_df = pd.DataFrame({
    'Date'     : dates_seq.values,
    'Actual'   : all_actuals,
    'Predicted': all_preds
})
if MONTHLY_ONLY:
    plot_df = plot_df.set_index('Date').resample('ME').mean().reset_index()

# ==========================================================
# 13. FORECAST PLOT
# ==========================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 1, figsize=FIGSIZE,
                         gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.30})

ax1 = axes[0]
ax1.fill_between(plot_df['Date'], plot_df['Actual'], plot_df['Predicted'],
                 alpha=0.15, color='crimson', label='Error Band')
ax1.plot(plot_df['Date'], plot_df['Actual'],
         color='#1a1a1a', linewidth=1.8, label='Actual', zorder=3)
ax1.plot(plot_df['Date'], plot_df['Predicted'],
         color='#1565C0', linewidth=1.8, alpha=0.85,
         label='BiLSTM+Attn (R^2=%.2f)' % r2, zorder=2)
ax1.axvline(x=split_date, color='red', linestyle='--', linewidth=2.2,
            label='Train/Test Split', zorder=4)
ax1.set_title('LSTM Forecasting - Full History (Monthly Means)',
              fontsize=15, fontweight='bold', pad=12)
ax1.set_ylabel('Precipitation (mm)', fontsize=13)
ax1.legend(loc='upper left', fontsize=11, frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', labelsize=11)

ax2 = axes[1]
zoom_start = plot_df['Date'].max() - pd.DateOffset(years=ZOOM_YEARS)
zoom_df    = plot_df[plot_df['Date'] >= zoom_start].copy()
ax2.plot(zoom_df['Date'], zoom_df['Actual'],
         marker='o', markersize=3.5, color='#1a1a1a', linewidth=1.2, label='Actual')
ax2.plot(zoom_df['Date'], zoom_df['Predicted'],
         marker='s', markersize=3, color='#1565C0', linewidth=1.2, alpha=0.8, label='Predicted')
ax2.fill_between(zoom_df['Date'], zoom_df['Actual'], zoom_df['Predicted'],
                 alpha=0.15, color='crimson')
ax2.set_title('Zoom: Last %d Years - Test Period (Monthly)' % ZOOM_YEARS,
              fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Date', fontsize=13)
ax2.set_ylabel('Precipitation (mm)', fontsize=12)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', labelsize=10)

plt.savefig('output_graphs/lstm_forecast_CLEAR.png', dpi=DPI, bbox_inches='tight')
print("Saved -> lstm_forecast_CLEAR.png")

# ==========================================================
# 14. TRAINING CURVE
# ==========================================================
fig2, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_losses, label='Train', color='#1565C0', linewidth=1.8)
ax.plot(val_losses,   label='Val',   color='crimson',  linewidth=1.8)
ax.set_title('BiLSTM+Attn Training Loss (Huber)', fontsize=13, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output_graphs/lstm_training_curves.png', dpi=DPI, bbox_inches='tight')
print("Saved -> lstm_training_curves.png")