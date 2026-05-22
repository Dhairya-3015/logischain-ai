import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import os

# ── PATHS ────────────────────────────────
FEATURES_PATH = "data/features/features.csv"
MODEL_OUTPUT = "data/processed/model_results"

# ── STEP 1: GENERATE TIME SERIES DATA ────
def generate_time_series():
    """
    Generate realistic supply chain time series
    Based on document: port throughput, freight rates
    """
    print("\n🔧 Generating time series data...")
    np.random.seed(42)

    # 3 years of daily data
    n_days = 365 * 3
    dates = pd.date_range('2021-01-01', periods=n_days, freq='D')

    # ── PORT THROUGHPUT (TEU per day) ─────
    # From document: Port of Shanghai ~47M TEU/year
    # Daily = 47M/365 = ~128,767 TEU
    base_throughput = 128000

    # Seasonal pattern (Chinese New Year dip in Feb)
    seasonal = (
        3000 * np.sin(2 * np.pi * np.arange(n_days) / 365) +
        2000 * np.sin(4 * np.pi * np.arange(n_days) / 365)
    )

    # COVID spike effect in 2021
    covid_effect = np.zeros(n_days)
    covid_effect[:180] = -8000  # reduced in early 2021
    covid_effect[180:365] = 5000  # recovery

    # Chinese New Year effect
    cny_effect = np.zeros(n_days)
    # Feb 2021 (day 32-46)
    cny_effect[32:46] = -15000
    # Feb 2022 (day 397-411)
    cny_effect[397:411] = -15000
    # Feb 2023 (day 762-776)
    cny_effect[762:776] = -15000

    # Random noise
    noise = np.random.normal(0, 2000, n_days)

    # Trend (slight growth)
    trend = np.linspace(0, 5000, n_days)

    throughput = (base_throughput + seasonal +
                  covid_effect + cny_effect +
                  noise + trend)
    throughput = np.clip(throughput, 80000, 180000)

    # ── FREIGHT RATES (USD per FEU) ───────
    # Shanghai → Rotterdam
    base_rate = 2000

    # COVID freight spike 2021-2022
    freight_spike = np.zeros(n_days)
    freight_spike[180:550] = 8000  # peak COVID rates
    freight_spike[550:730] = 4000  # gradual decline

    freight_noise = np.random.normal(0, 200, n_days)
    freight_trend = np.linspace(0, -500, n_days)

    freight_rates = (base_rate + freight_spike +
                     freight_noise + freight_trend)
    freight_rates = np.clip(freight_rates, 500, 15000)

    # ── OTIF RATE (daily) ─────────────────
    base_otif = 0.88
    otif_noise = np.random.normal(0, 0.03, n_days)
    # OTIF drops during COVID spike
    otif_drop = np.zeros(n_days)
    otif_drop[180:550] = -0.08
    otif = np.clip(
        base_otif + otif_noise + otif_drop, 0.5, 1.0
    )

    # Create DataFrame
    df_ts = pd.DataFrame({
        'date': dates,
        'port_throughput': throughput,
        'freight_rate': freight_rates,
        'otif_rate': otif,
    })

    print(f"✅ Time series generated: {len(df_ts)} days")
    print(f"✅ Date range: {df_ts['date'].min().date()} to {df_ts['date'].max().date()}")
    print(f"✅ Avg throughput: {df_ts['port_throughput'].mean():,.0f} TEU/day")
    print(f"✅ Avg freight rate: ${df_ts['freight_rate'].mean():,.0f}/FEU")
    print(f"✅ Avg OTIF: {df_ts['otif_rate'].mean():.1%}")

    # Save
    os.makedirs("data/raw", exist_ok=True)
    df_ts.to_csv("data/raw/time_series.csv", index=False)
    print("✅ Saved to data/raw/time_series.csv")

    return df_ts

# ── STEP 2: PREPARE SEQUENCES ────────────
def prepare_sequences(df_ts, target_col='port_throughput',
                       input_len=128, forecast_horizons=[30, 60, 90]):
    """
    Create input/output sequences for TCN
    From document: input_chunk_length=128
    Output: 30/60/90 day forecasts
    """
    print(f"\n🔧 Preparing sequences...")
    print(f"   Input length:  {input_len} days")
    print(f"   Forecast:      {forecast_horizons} days")

    # Normalize
    scaler = MinMaxScaler()
    values = scaler.fit_transform(
        df_ts[target_col].values.reshape(-1, 1)
    ).flatten()

    # Create sequences
    max_horizon = max(forecast_horizons)
    X, y_30, y_60, y_90 = [], [], [], []

    for i in range(len(values) - input_len - max_horizon):
        # Input sequence
        X.append(values[i:i+input_len])

        # Output at different horizons
        y_30.append(values[i+input_len:i+input_len+30].mean())
        y_60.append(values[i+input_len:i+input_len+60].mean())
        y_90.append(values[i+input_len:i+input_len+90].mean())

    X = np.array(X)
    y_30 = np.array(y_30)
    y_60 = np.array(y_60)
    y_90 = np.array(y_90)

    # Split train/test (80/20)
    split = int(len(X) * 0.8)
    X_train = torch.tensor(X[:split], dtype=torch.float).unsqueeze(1)
    X_test = torch.tensor(X[split:], dtype=torch.float).unsqueeze(1)
    y_train = torch.tensor(
        np.stack([y_30[:split], y_60[:split], y_90[:split]], axis=1),
        dtype=torch.float
    )
    y_test = torch.tensor(
        np.stack([y_30[split:], y_60[split:], y_90[split:]], axis=1),
        dtype=torch.float
    )

    print(f"✅ Train samples: {len(X_train)}")
    print(f"✅ Test samples:  {len(X_test)}")
    print(f"✅ Input shape:   {X_train.shape}")
    print(f"✅ Output shape:  {y_train.shape}")

    return X_train, X_test, y_train, y_test, scaler

# ── STEP 3: TCN MODEL ─────────────────────
class CausalConv1d(nn.Module):
    """
    Causal Convolution - only uses past information
    From document: strictly causal architecture
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding
        )

    def forward(self, x):
        out = self.conv(x)
        # Remove future padding (causal)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    """
    Residual TCN Block
    From document: dilated causal conv + residual connection
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, dilation, dropout=0.2):
        super().__init__()

        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation
        )
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation
        )

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + residual)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network
    From document: dilation factors [1,2,4,8,16,32,64]
    Achieves 128-day receptive field with 7 layers
    """
    def __init__(self, in_channels=1, n_filters=64,
                 kernel_size=3, dropout=0.2):
        super().__init__()

        # Dilation factors from document
        dilations = [1, 2, 4, 8, 16, 32, 64]

        # Build TCN layers
        layers = []
        channels = in_channels

        for i, dilation in enumerate(dilations):
            out_ch = n_filters
            layers.append(
                TCNBlock(channels, out_ch,
                         kernel_size, dilation, dropout)
            )
            channels = out_ch

        self.tcn = nn.Sequential(*layers)

        # Multi-horizon output heads
        # From document: 30/60/90 day forecasts
        self.head_30 = nn.Linear(n_filters, 1)
        self.head_60 = nn.Linear(n_filters, 1)
        self.head_90 = nn.Linear(n_filters, 1)

    def forward(self, x):
        # x shape: (batch, channels, sequence_length)
        out = self.tcn(x)

        # Take last timestep
        last = out[:, :, -1]

        # Multi-horizon predictions
        pred_30 = self.head_30(last)
        pred_60 = self.head_60(last)
        pred_90 = self.head_90(last)

        return torch.cat([pred_30, pred_60, pred_90], dim=1)

# ── STEP 4: TRAIN TCN ─────────────────────
def train_tcn(X_train, y_train, X_test, y_test):
    print("\n🔧 Training TCN...")

    model = TCNModel(
        in_channels=1,
        n_filters=64,
        kernel_size=3,
        dropout=0.2
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    # Training
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(200):
        # Train
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()
        train_losses.append(loss.item())

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test)
            val_loss = F.mse_loss(val_pred, y_test)
        val_losses.append(val_loss.item())

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 
                      f'{MODEL_OUTPUT}/tcn_best.pt')
        else:
            patience_counter += 1

        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: "
                  f"Train Loss={loss.item():.6f}, "
                  f"Val Loss={val_loss.item():.6f}")

    # Load best model
    model.load_state_dict(
        torch.load(f'{MODEL_OUTPUT}/tcn_best.pt')
    )
    print("✅ Training complete!")
    return model, train_losses, val_losses

# ── STEP 5: EVALUATE ──────────────────────
def evaluate_tcn(model, X_test, y_test, scaler):
    print("\n📊 Evaluating TCN...")

    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()

    actuals = y_test.numpy()

    # Inverse transform
    pred_30 = scaler.inverse_transform(
        predictions[:, 0].reshape(-1, 1)
    ).flatten()
    pred_60 = scaler.inverse_transform(
        predictions[:, 1].reshape(-1, 1)
    ).flatten()
    pred_90 = scaler.inverse_transform(
        predictions[:, 2].reshape(-1, 1)
    ).flatten()

    act_30 = scaler.inverse_transform(
        actuals[:, 0].reshape(-1, 1)
    ).flatten()
    act_60 = scaler.inverse_transform(
        actuals[:, 1].reshape(-1, 1)
    ).flatten()
    act_90 = scaler.inverse_transform(
        actuals[:, 2].reshape(-1, 1)
    ).flatten()

    # MAPE from document
    mape_30 = mean_absolute_percentage_error(act_30, pred_30) * 100
    mape_60 = mean_absolute_percentage_error(act_60, pred_60) * 100
    mape_90 = mean_absolute_percentage_error(act_90, pred_90) * 100

    print(f"\n{'='*50}")
    print(f"TCN PERFORMANCE")
    print(f"{'='*50}")
    print(f"MAPE 30-day: {mape_30:.2f}%  (target < 12%)")
    print(f"MAPE 60-day: {mape_60:.2f}%  (target < 15%)")
    print(f"MAPE 90-day: {mape_90:.2f}%")

    if mape_30 < 12:
        print(f"✅ 30-day forecast meets target!")
    else:
        print(f"⚠️  30-day forecast above target")

    return pred_30, pred_60, pred_90, act_30

def plot_results(train_losses, val_losses,
                 pred_30, act_30,
                 pred_60, pred_90):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training Loss
    axes[0,0].plot(train_losses, label='Train', color='blue')
    axes[0,0].plot(val_losses, label='Val', color='orange')
    axes[0,0].set_title('TCN Training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('MSE Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)

    # Plot 2: 30-day forecast
    n = min(100, len(act_30))
    axes[0,1].plot(act_30[:n], label='Actual', color='blue')
    axes[0,1].plot(pred_30[:n], label='Predicted', color='red')
    axes[0,1].set_title('30-Day Forecast (Port Throughput)')
    axes[0,1].set_xlabel('Test Sample')
    axes[0,1].set_ylabel('TEU/day')
    axes[0,1].legend()
    axes[0,1].grid(True)

    # Plot 3: 60-day forecast
    axes[1,0].plot(act_30[:n], label='Actual', color='blue')
    axes[1,0].plot(pred_60[:n], label='60-day Pred', 
                   color='green', linestyle='--')
    axes[1,0].set_title('60-Day Forecast')
    axes[1,0].set_xlabel('Test Sample')
    axes[1,0].set_ylabel('TEU/day')
    axes[1,0].legend()
    axes[1,0].grid(True)

    # Plot 4: 90-day forecast
    axes[1,1].plot(act_30[:n], label='Actual', color='blue')
    axes[1,1].plot(pred_90[:n], label='90-day Pred',
                   color='purple', linestyle='--')
    axes[1,1].set_title('90-Day Forecast')
    axes[1,1].set_xlabel('Test Sample')
    axes[1,1].set_ylabel('TEU/day')
    axes[1,1].legend()
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.savefig(f'{MODEL_OUTPUT}/tcn_results.png')
    plt.show()
    print("✅ TCN plots saved")

if __name__ == "__main__":
    os.makedirs(MODEL_OUTPUT, exist_ok=True)

    # Generate time series
    df_ts = generate_time_series()

    # Prepare sequences
    X_train, X_test, y_train, y_test, scaler = prepare_sequences(
        df_ts, target_col='port_throughput'
    )

    # Train
    model, train_losses, val_losses = train_tcn(
        X_train, y_train, X_test, y_test
    )

    # Evaluate
    pred_30, pred_60, pred_90, act_30 = evaluate_tcn(
        model, X_test, y_test, scaler
    )

    # Plot
    plot_results(train_losses, val_losses,
                 pred_30, act_30,
                 pred_60, pred_90)
