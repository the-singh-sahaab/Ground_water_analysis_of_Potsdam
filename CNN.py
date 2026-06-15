# -*- coding: ascii -*-
"""
CNN.py  v3
-----------
1D-CNN for Precipitation forecasting.
Drop-in replacement for any previous CNN.py on your HPC.

Root cause of R2=-0.26 in previous runs:
  The windowing code predicted y[i] from X[i-W:i] (NEXT-DAY forecast).
  XGBoost gets 0.98 because it sees same-day features (RSK, TMK, ...).
  This version predicts y[i] from X[i-W+1:i+1] (SAME-DAY, last step = today).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # HPC: no display server needed
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ===========================================================
# 0. CONFIG
# ===========================================================
CSV_PATH     = "dataset/joined_by_date.csv"
TARGET       = "Precipitation"
TEST_SIZE    = 0.20
WINDOW       = 60          # days of look-back context
EPOCHS       = 100
BATCH_SIZE   = 512
LR           = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 15
LOG1P_TARGET = True
CKPT_PATH    = "models/cnn_best.pt"

FIGSIZE      = (16, 9)
DPI          = 300
ZOOM_YEARS   = 15

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[0] Device : {DEVICE}")

# ===========================================================
# 1. LOAD & PARSE
# ===========================================================
df = pd.read_csv(CSV_PATH)

if '__date__' in df.columns:
    df['Date_parsed'] = pd.to_datetime(df['__date__'], errors='coerce')
else:
    df['Date_parsed'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

df = df.dropna(subset=['Date_parsed'])           # drop rows with unparseable dates
df = df.sort_values('Date_parsed').reset_index(drop=True)

print(f"[1] Rows   : {len(df)}")
print(f"    Range  : {df['Date_parsed'].min().date()} -> {df['Date_parsed'].max().date()}")
print(f"    NaT dates removed: confirmed {df['Date_parsed'].isna().sum()} remaining")

# ===========================================================
# 2. CLEAN
# ===========================================================
df = df.replace(-999.0, np.nan)

meta_cols = ['Date', '__date__', 'MESS_DATUM', 'STATIONS_ID',
             'eor', 'Date_parsed', 'QN_3', 'QN_4']
drop_cols  = [c for c in meta_cols if c in df.columns]

feature_df    = df.drop(columns=[TARGET] + drop_cols)
feature_df    = feature_df.apply(pd.to_numeric, errors='coerce')
target_series = df[TARGET].clip(lower=0)

nan_pct  = feature_df.isna().mean().sort_values(ascending=False)
bad_cols = nan_pct[nan_pct > 0.40].index.tolist()
print(f"\n[2] NaN > 40% -> dropping: {bad_cols}")
feature_df = feature_df.drop(columns=bad_cols)

feature_df    = feature_df.ffill().bfill().fillna(0)
target_series = target_series.ffill().bfill().fillna(0)

print(f"    Feature matrix : {feature_df.shape}")
print(f"    Features used  : {list(feature_df.columns)}")

# ===========================================================
# 3. CYCLICAL TIME FEATURES
# ===========================================================
dates = df['Date_parsed'].reset_index(drop=True)
doy   = dates.dt.dayofyear.values
month = dates.dt.month.values

feature_df['sin_doy']   = np.sin(2 * np.pi * doy   / 365.25)
feature_df['cos_doy']   = np.cos(2 * np.pi * doy   / 365.25)
feature_df['sin_month'] = np.sin(2 * np.pi * month / 12.0)
feature_df['cos_month'] = np.cos(2 * np.pi * month / 12.0)

X_raw = feature_df.values.astype(np.float32)
y_raw = target_series.values.astype(np.float32)

print(f"[3] Added sin/cos features -> final shape: {X_raw.shape}")

# ===========================================================
# 4. LOG1p TRANSFORM
# ===========================================================
if LOG1P_TARGET:
    y_model = np.log1p(y_raw)
    print(f"\n[4] log1p transform:")
    print(f"    Before: mean={y_raw.mean():.3f}  std={y_raw.std():.3f}  max={y_raw.max():.1f}")
    print(f"    After : mean={y_model.mean():.3f}  std={y_model.std():.3f}  max={y_model.max():.2f}")
else:
    y_model = y_raw.copy()
    print("[4] log1p skipped")

# ===========================================================
# 5. CHRONOLOGICAL SPLIT  (on raw rows, BEFORE windowing)
# ===========================================================
split_idx  = int(len(df) * (1 - TEST_SIZE))
split_date = dates.iloc[split_idx]

print(f"\n[5] Chronological split:")
print(f"    split_idx  = {split_idx}")
print(f"    split_date = {split_date.date()}")
assert pd.notna(split_date), "split_date is NaT -- check date parsing above"

# ===========================================================
# 6. SCALE ON TRAIN ONLY  (fit before split_idx)
# ===========================================================
feat_scaler   = StandardScaler()
target_scaler = StandardScaler()

feat_scaler.fit(X_raw[:split_idx])
target_scaler.fit(y_model[:split_idx].reshape(-1, 1))

X_sc = feat_scaler.transform(X_raw).astype(np.float32)
y_sc = target_scaler.transform(y_model.reshape(-1, 1)).flatten().astype(np.float32)

print(f"\n[6] Scaler fit on train rows 0..{split_idx-1} only:")
print(f"    X_sc  train mean={X_sc[:split_idx].mean():.4f}  std={X_sc[:split_idx].std():.4f}")
print(f"    y_sc  train mean={y_sc[:split_idx].mean():.4f}  std={y_sc[:split_idx].std():.4f}")

# ===========================================================
# 7. SLIDING WINDOW  --  SAME-DAY PREDICTION
# -----------------------------------------------------------
# WINDOW covers rows [i .. i+WINDOW-1].
# Label = y[i+WINDOW-1]  <-- LAST row in window = TODAY.
# The model sees today's own features (RSK, TMK, VPM, ...)
# in the final time-step, matching exactly what XGBoost sees.
#
# Old (broken) approach: label = y[i+WINDOW] = TOMORROW.
# That turned a regression into a forecast; hence R2 ~ -0.26.
# ===========================================================
def make_windows(X, y, W):
    N   = len(X)
    n   = N - W + 1              # number of valid windows
    out_x = np.empty((n, X.shape[1], W), dtype=np.float32)
    out_y = np.empty(n, dtype=np.float32)
    for i in range(n):
        out_x[i] = X[i : i + W].T    # (n_feat, W)  -- CNN channel-first
        out_y[i] = y[i + W - 1]      # label = LAST day in window
    return out_x, out_y

print("\n[7] Building windows (same-day label) ...")
X_win, y_win = make_windows(X_sc, y_sc, WINDOW)

# Dates: each window's date = last day of window
dates_win = dates.iloc[WINDOW - 1 :].reset_index(drop=True)

assert len(X_win) == len(dates_win), \
    f"X_win len {len(X_win)} != dates_win len {len(dates_win)}"

# Train/test split on windowed data
# Window i covers rows [i .. i+W-1]; its label row is i+W-1.
# We want all windows whose label-row < split_idx -> i+W-1 < split_idx -> i < split_idx-W+1
adj_split    = split_idx - WINDOW + 1
split_date_w = dates_win.iloc[adj_split]

print(f"    X_win shape    : {X_win.shape}")
print(f"    adj_split      : {adj_split}")
print(f"    split_date_w   : {split_date_w.date()}")
print(f"    Train windows  : {adj_split}")
print(f"    Test  windows  : {len(X_win) - adj_split}")
assert pd.notna(split_date_w), "split_date_w is NaT -- windowing date mismatch"

X_train_np = X_win[:adj_split]
y_train_np = y_win[:adj_split]
X_test_np  = X_win[adj_split:]
y_test_np  = y_win[adj_split:]

X_tr = torch.from_numpy(X_train_np)
y_tr = torch.from_numpy(y_train_np)
X_te = torch.from_numpy(X_test_np)
y_te = torch.from_numpy(y_test_np)

train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(TensorDataset(X_te, y_te),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

# ===========================================================
# 8. MODEL  --  Residual 1D-CNN
# ===========================================================
class ResBlock(nn.Module):
    def __init__(self, ch, ks=3):
        super().__init__()
        p = ks // 2
        self.body = nn.Sequential(
            nn.Conv1d(ch, ch, ks, padding=p), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, ks, padding=p), nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.body(x))


class CNN1D(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_feat, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU()
        )
        self.layer1 = nn.Sequential(ResBlock(64),  ResBlock(64))
        self.up1    = nn.Sequential(nn.Conv1d(64,  128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.layer2 = nn.Sequential(ResBlock(128), ResBlock(128))
        self.up2    = nn.Sequential(nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU())
        self.layer3 = nn.Sequential(ResBlock(256), ResBlock(256))
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64,  1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.up1(x)
        x = self.layer2(x); x = self.up2(x)
        x = self.layer3(x)
        return self.head(self.pool(x)).squeeze(-1)


n_feat = X_win.shape[1]
model  = CNN1D(n_feat).to(DEVICE)
print(f"\n[8] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = nn.HuberLoss(delta=1.0)

# ===========================================================
# 9. TRAIN
# ===========================================================
best_val, no_improve = float('inf'), 0
train_losses, val_losses = [], []

print("\n[9] Training ...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    run = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        run += loss.item() * len(xb)
    t_loss = run / len(X_tr)

    model.eval()
    run = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            run += criterion(model(xb), yb).item() * len(xb)
    v_loss = run / len(X_te)

    train_losses.append(t_loss)
    val_losses.append(v_loss)
    scheduler.step()

    if v_loss < best_val:
        best_val, no_improve = v_loss, 0
        torch.save(model.state_dict(), CKPT_PATH)
    else:
        no_improve += 1

    if epoch % 5 == 0 or epoch == 1:
        lr_now = scheduler.get_last_lr()[0]
        print(f"    Epoch {epoch:>3}/{EPOCHS}  Train={t_loss:.5f}  Val={v_loss:.5f}"
              f"  Best={best_val:.5f}  LR={lr_now:.6f}  Patience={no_improve}/{PATIENCE}")

    if no_improve >= PATIENCE:
        print(f"    Early stop at epoch {epoch}")
        break

model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
print(f"    Loaded best checkpoint (val={best_val:.5f})")

# ===========================================================
# 10. FULL-HISTORY INFERENCE  +  INVERSE TRANSFORMS
# ===========================================================
model.eval()
chunks = []
all_tensor = torch.from_numpy(X_win)
with torch.no_grad():
    for i in range(0, len(X_win), BATCH_SIZE):
        xb = all_tensor[i : i + BATCH_SIZE].to(DEVICE)
        chunks.append(model(xb).cpu().numpy())
pred_sc = np.concatenate(chunks)

# Inverse StandardScaler
pred_log = target_scaler.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
act_log  = target_scaler.inverse_transform(y_win.reshape(-1, 1)).flatten()

# Inverse log1p
if LOG1P_TARGET:
    all_preds   = np.expm1(pred_log).clip(min=0)
    all_actuals = np.expm1(act_log).clip(min=0)
else:
    all_preds   = pred_log.clip(min=0)
    all_actuals = act_log.clip(min=0)

test_preds  = all_preds[adj_split:]
test_actual = all_actuals[adj_split:]

r2   = r2_score(test_actual, test_preds)
rmse = np.sqrt(mean_squared_error(test_actual, test_preds))
mae  = mean_absolute_error(test_actual, test_preds)

print(f"\n[10] Test metrics:")
print(f"     R2   = {r2:.4f}")
print(f"     RMSE = {rmse:.4f}")
print(f"     MAE  = {mae:.4f}")

# ===========================================================
# 11. PLOT
# ===========================================================
plot_df = pd.DataFrame({
    'Date'     : dates_win.values,
    'Actual'   : all_actuals,
    'Predicted': all_preds
}).dropna(subset=['Date'])

monthly = plot_df.set_index('Date').resample('ME').mean().reset_index()

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 1, figsize=FIGSIZE,
                         gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.30})

ax1 = axes[0]
ax1.fill_between(monthly['Date'], monthly['Actual'], monthly['Predicted'],
                 alpha=0.15, color='crimson', label='Error Band')
ax1.plot(monthly['Date'], monthly['Actual'],
         color='#1a1a1a', linewidth=1.8, label='Actual', zorder=3)
ax1.plot(monthly['Date'], monthly['Predicted'],
         color='#2E7D32', linewidth=1.8, alpha=0.85,
         label='1D-CNN (R2=%.2f)' % r2, zorder=2)
ax1.axvline(split_date_w, color='red', linestyle='--',
            linewidth=2.2, label='Train/Test Split', zorder=4)
ax1.set_title('1D-CNN Precipitation Forecast (Full History - Monthly Means)',
              fontsize=15, fontweight='bold', pad=12)
ax1.set_ylabel('Precipitation (mm)', fontsize=13)
ax1.legend(loc='upper left', fontsize=11, frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', labelsize=11)

ax2 = axes[1]
zoom_start = monthly['Date'].max() - pd.DateOffset(years=ZOOM_YEARS)
z = monthly[monthly['Date'] >= zoom_start]
ax2.plot(z['Date'], z['Actual'],
         marker='o', markersize=3.5, color='#1a1a1a', linewidth=1.2, label='Actual')
ax2.plot(z['Date'], z['Predicted'],
         marker='s', markersize=3, color='#2E7D32', linewidth=1.2, alpha=0.8, label='Predicted')
ax2.fill_between(z['Date'], z['Actual'], z['Predicted'], alpha=0.15, color='crimson')
ax2.set_title('Zoom: Last %d Years of Test Period (Monthly)' % ZOOM_YEARS,
              fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Date', fontsize=13)
ax2.set_ylabel('Precipitation (mm)', fontsize=12)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', labelsize=10)

fig.savefig('output_graphs/cnn_forecast.png', dpi=DPI, bbox_inches='tight')
print("\n[11] Saved -> cnn_forecast.png")

fig2, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_losses, label='Train', color='#2E7D32', linewidth=1.8)
ax.plot(val_losses,   label='Val',   color='crimson',  linewidth=1.8)
ax.set_title('1D-CNN Training & Validation Loss (Huber)', fontsize=13, fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig('output_graphs/cnn_training_curves.png', dpi=DPI, bbox_inches='tight')
print("     Saved -> cnn_training_curves.png")