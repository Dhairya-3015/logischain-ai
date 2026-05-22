import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# ── PATHS ────────────────────────────────
FEATURES_PATH = "data/features/features.csv"
MODEL_OUTPUT = "data/processed/model_results"

# ── STEP 1: CREATE SHIPMENT SEQUENCES ────
def create_shipment_sequences(df):
    print("\n🔧 Creating shipment event sequences...")
    np.random.seed(42)

    # ── CHECK COLUMNS ─────────────────────
    # Add carrier_reliability if not exists
    if 'carrier_reliability' not in df.columns:
        carrier_otif = df.groupby('shipping_carriers')['otif'].mean()
        df = df.copy()
        df['carrier_reliability'] = df['shipping_carriers'].map(carrier_otif)
        print("✅ carrier_reliability computed")

    # Encode routes
    route_map = {'Route A': 0.0, 'Route B': 0.5, 'Route C': 1.0}
    df = df.copy()
    df['route_encoded_float'] = df['routes'].map(route_map).fillna(0.0)
    print("✅ routes encoded")

    sequences = []
    labels = []

    for _, row in df.iterrows():

        # Event 1: Booking
        booking = [
            row['lead_times'] / 30,
            row['order_quantities'] / 100,
            row['price'] / 100,
            row['supplier_concentration_hhi'],
            0.0,
            1.0 / 7,
            0.0,
        ]

        # Event 2: Loading
        loading = [
            row['stock_levels'] / 100,
            row['manufacturing_lead_time'] / 30,
            row['manufacturing_costs'] / 100,
            row['defect_rates'],
            0.0,
            2.0 / 7,
            row['freight_cost_per_unit'],
        ]

        # Event 3: Departure
        departure = [
            row['shipping_times'] / 30,
            row['carrier_reliability'],    # ✅ computed above
            row['freight_cost_per_unit'],
            row['route_encoded_float'],    # ✅ float not string
            0.0,
            3.0 / 7,
            row['transit_time_variance'] / 10,
        ]

        # Event 4: Transhipment
        transhipment = [
            max(0, row['shipment_delay_days']) / 30,
            row['supplier_concentration_hhi'],
            row['ccc'] / 100,
            row['otif'],
            max(0, row['shipment_delay_days']) / 30,
            4.0 / 7,
            row['safety_stock_days'] / 30,
        ]

        # Event 5: Arrival
        arrival = [
            row['shipping_times'] / 30,
            row['shipment_delay_days'] / 30,
            row['otif'],
            row['freight_cost_per_unit'],
            max(0, row['shipment_delay_days']) / 30,
            5.0 / 7,
            row['transit_time_variance'] / 10,
        ]

        # Event 6: Customs
        customs = [
            row['lead_times'] / 30,
            row['defect_rates'],
            row['supplier_concentration_hhi'],
            row['inventory_turnover_capped'] / 20,
            max(0, row['shipment_delay_days']) / 30,
            6.0 / 7,
            row['ccc'] / 100,
        ]

        # Event 7: Delivery
        delivery = [
            row['shipping_times'] / 30,
            row['otif'],
            row['freight_cost_per_unit'],
            row['defect_rates'],
            max(0, row['shipment_delay_days']) / 30,
            7.0 / 7,
            row['safety_stock_days'] / 30,
        ]

        # Stack → shape (7 events, 7 features)
        sequence = np.array([
            booking, loading, departure,
            transhipment, arrival, customs, delivery
        ], dtype=np.float32)

        # Replace any nan/inf
        sequence = np.nan_to_num(
            sequence, nan=0.0, posinf=1.0, neginf=0.0
        )

        sequences.append(sequence)

        # Label
        label = 1 if row['inspection_results'] == 'Fail' else 0
        labels.append(label)

    sequences = np.array(sequences)
    labels = np.array(labels)

    print(f"✅ Sequences shape: {sequences.shape}")
    print(f"   (shipments, events, features) = {sequences.shape}")
    print(f"✅ Fail: {labels.sum()}, Pass: {(labels==0).sum()}")

    return sequences, labels


# ── STEP 2: ADD SYNTHETIC DATA ────────────
# Current noise is too random
# Better: generate based on risk patterns

def augment_data(sequences, labels, n_synthetic=2000):
    print(f"\n🔧 Adding {n_synthetic} synthetic sequences...")
    np.random.seed(42)

    syn_sequences = []
    syn_labels = []

    for i in range(n_synthetic):
        # Pick random real sequence
        idx = np.random.randint(0, len(sequences))
        base_seq = sequences[idx].copy()
        label = labels[idx]

        if label == 1:
            # HIGH RISK pattern
            noise = np.random.normal(0, 0.08, base_seq.shape)
            syn_seq = base_seq + noise

            # Transhipment event → more delay
            syn_seq[3, 4] = min(1.0, base_seq[3, 4] +
                                np.random.uniform(0.1, 0.3))

            # Arrival event → worse OTIF
            syn_seq[4, 2] = max(0.0, base_seq[4, 2] -
                                np.random.uniform(0.1, 0.2))

            # Customs event → more defects
            syn_seq[5, 1] = min(1.0, base_seq[5, 1] +
                                np.random.uniform(0.1, 0.2))

        else:
            # LOW RISK pattern
            noise = np.random.normal(0, 0.05, base_seq.shape)
            syn_seq = base_seq + noise

            # Delivery event → better OTIF
            syn_seq[6, 1] = min(1.0, base_seq[6, 1] +
                                np.random.uniform(0.05, 0.15))

            # Arrival event → less delay
            syn_seq[4, 4] = max(0.0, base_seq[4, 4] -
                                np.random.uniform(0.05, 0.1))

        syn_seq = np.clip(syn_seq, 0, 1)
        syn_seq = np.nan_to_num(syn_seq, nan=0.0)
        syn_sequences.append(syn_seq)
        syn_labels.append(label)

    syn_sequences = np.array(syn_sequences)
    syn_labels = np.array(syn_labels)

    all_sequences = np.concatenate(
        [sequences, syn_sequences], axis=0
    )
    all_labels = np.concatenate(
        [labels, syn_labels], axis=0
    )

    print(f"✅ Original:  {len(sequences)}")
    print(f"✅ Synthetic: {n_synthetic}")
    print(f"✅ Total:     {len(all_sequences)}")
    print(f"✅ Fail rate: {all_labels.mean():.1%}")

    return all_sequences, all_labels


# ── STEP 3: TRANSFORMER MODEL ─────────────
class ShipmentTransformer(nn.Module):
    """
    Transformer Encoder for Shipment Risk
    From document:
    - Multi-head self-attention (4+ heads)
    - Event sequence encoding
    - Interpretable attention weights
    """
    def __init__(self, event_dim=7, d_model=64,
                 n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()

        # Event embedding
        self.event_embedding = nn.Linear(event_dim, d_model)

        # Positional encoding
        self.pos_encoding = nn.Embedding(10, d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,          # 4 heads from document
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True        # (batch, seq, features)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # Risk prediction head
        # From document risk outputs:
        # Delay probability, Damage probability,
        # Documentation discrepancy, Total risk score
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Store attention weights for interpretability
        self.attention_weights = None

    def forward(self, x):
        """
        x shape: (batch, n_events, event_dim)
        """
        batch_size, n_events, _ = x.shape

        # Embed events
        x = self.event_embedding(x)

        # Add positional encoding
        positions = torch.arange(n_events).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        x = x + self.pos_encoding(positions)

        # Transformer encoding
        encoded = self.transformer(x)

        # Global average pooling over events
        pooled = encoded.mean(dim=1)

        # Risk score
        risk = self.risk_head(pooled)

        return risk.squeeze(1)

    def get_attention_weights(self, x):
        """
        Extract attention weights for interpretability
        From document: attention weights are interpretable
        """
        batch_size, n_events, _ = x.shape

        x_emb = self.event_embedding(x)
        positions = torch.arange(n_events).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        x_emb = x_emb + self.pos_encoding(positions)

        # Get attention from first layer
        attention_weights = []

        # Hook to capture attention
        def hook_fn(module, input, output):
            attention_weights.append(output)

        # Manual attention computation
        d_model = x_emb.shape[-1]
        n_heads = 4
        head_dim = d_model // n_heads

        return x_emb  # Return embeddings for visualization


# ── STEP 4: TRAIN TRANSFORMER ─────────────
def train_transformer(X_train, y_train, X_test, y_test):
    print("\n🔧 Training Transformer...")

    model = ShipmentTransformer(
        event_dim=7,
        d_model=64,
        n_heads=4,
        n_layers=3,
        dropout=0.1    # back to 0.1
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,      # back to 0.001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    X_tr = torch.tensor(X_train, dtype=torch.float)
    y_tr = torch.tensor(y_train, dtype=torch.float)
    X_te = torch.tensor(X_test, dtype=torch.float)
    y_te = torch.tensor(y_test, dtype=torch.float)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = F.binary_cross_entropy(pred, y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_te)
            val_loss = F.binary_cross_entropy(
                val_pred, y_te
            )
        val_losses.append(val_loss.item())

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                      f'{MODEL_OUTPUT}/transformer_best.pt')
        else:
            patience_counter += 1

        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: "
                  f"Train={loss.item():.4f}, "
                  f"Val={val_loss.item():.4f}")

    model.load_state_dict(
        torch.load(f'{MODEL_OUTPUT}/transformer_best.pt')
    )
    print("✅ Training complete!")
    return model, train_losses, val_losses


# ── STEP 5: EVALUATE ──────────────────────
def evaluate_transformer(model, X_test, y_test):
    print("\n📊 Evaluating Transformer...")

    X_te = torch.tensor(X_test, dtype=torch.float)

    model.eval()
    with torch.no_grad():
        y_prob = model(X_te).numpy()

    # Metrics from document
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    gini = 2 * auc - 1

    print(f"\n{'='*50}")
    print(f"TRANSFORMER PERFORMANCE")
    print(f"{'='*50}")
    print(f"AUC-ROC:     {auc:.3f}  (target > 0.80)")
    print(f"Gini:        {gini:.3f}")
    print(f"Brier Score: {brier:.3f}  (target < 0.18)")

    if auc > 0.80:
        print(f"✅ AUC meets target!")
    else:
        print(f"⚠️  AUC below target")

    if brier < 0.18:
        print(f"✅ Brier Score meets target!")
    else:
        print(f"⚠️  Brier Score above target")

    return y_prob


# ── STEP 6: ATTENTION VISUALIZATION ───────
def plot_attention(model, X_test):
    """
    Visualize which shipment events matter most
    From document: attention weights are interpretable
    """
    print("\n📊 Visualizing attention weights...")

    event_names = [
        'Booking', 'Loading', 'Departure',
        'Transhipment', 'Arrival', 'Customs', 'Delivery'
    ]

    # Get sample
    sample = torch.tensor(
        X_test[:1], dtype=torch.float
    )

    # Compute event importance via gradient
    sample.requires_grad_(True)
    model.eval()
    output = model(sample)
    output.backward()

    # Gradient magnitude = importance
    importance = sample.grad.abs().mean(dim=-1).squeeze()
    importance = importance.detach().numpy()
    importance = importance / importance.sum()

    return importance, event_names


def plot_results(train_losses, val_losses,
                 y_prob, y_test,
                 importance, event_names):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training Loss
    axes[0,0].plot(train_losses, label='Train', color='blue')
    axes[0,0].plot(val_losses, label='Val', color='orange')
    axes[0,0].set_title('Transformer Training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('BCE Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)

    # Plot 2: Risk Score Distribution
    axes[0,1].hist(y_prob[y_test==0], bins=20,
                   alpha=0.6, color='green', label='Pass')
    axes[0,1].hist(y_prob[y_test==1], bins=20,
                   alpha=0.6, color='red', label='Fail')
    axes[0,1].axvline(x=0.5, color='black',
                      linestyle='--', label='Threshold')
    axes[0,1].set_title('Shipment Risk Score Distribution')
    axes[0,1].set_xlabel('Risk Score')
    axes[0,1].set_ylabel('Count')
    axes[0,1].legend()
    axes[0,1].grid(True)

    # Plot 3: Event Importance (Attention)
    colors = plt.cm.RdYlGn(
        [1 - imp for imp in importance]
    )
    bars = axes[1,0].bar(event_names, importance,
                          color=colors)
    axes[1,0].set_title('Event Importance\n(Which events drive risk?)')
    axes[1,0].set_xlabel('Shipment Event')
    axes[1,0].set_ylabel('Importance Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, axis='y')

    # Add value labels
    for bar, imp in zip(bars, importance):
        axes[1,0].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.001,
            f'{imp:.1%}',
            ha='center', va='bottom', fontsize=9
        )

    # Plot 4: Risk Score vs Actual
    axes[1,1].scatter(
        range(len(y_prob[:50])),
        y_prob[:50],
        c=['red' if y==1 else 'green' for y in y_test[:50]],
        alpha=0.7
    )
    axes[1,1].axhline(y=0.5, color='black',
                       linestyle='--', label='Threshold')
    axes[1,1].set_title('Risk Scores (Red=Fail, Green=Pass)')
    axes[1,1].set_xlabel('Shipment Index')
    axes[1,1].set_ylabel('Predicted Risk Score')
    axes[1,1].legend()
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.savefig(f'{MODEL_OUTPUT}/transformer_results.png')
    plt.show()
    print("✅ Transformer plots saved")


if __name__ == "__main__":
    os.makedirs(MODEL_OUTPUT, exist_ok=True)

    # Load data
    df = pd.read_csv(FEATURES_PATH)
    print(f"✅ Loaded: {df.shape[0]} rows")

    # Handle pending
    df = df[df['inspection_results'] != 'Pending'].copy()
    print(f"✅ After dropping Pending: {df.shape[0]} rows")

    # Create sequences
    sequences, labels = create_shipment_sequences(df)

    # Augment with synthetic data
    sequences, labels = augment_data(
        sequences, labels, n_synthetic=500
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    print(f"\n✅ Train: {len(X_train)}, Test: {len(X_test)}")

    # Train
    model, train_losses, val_losses = train_transformer(
        X_train, y_train, X_test, y_test
    )

    # Evaluate
    y_prob = evaluate_transformer(model, X_test, y_test)

    # Attention visualization
    importance, event_names = plot_attention(model, X_test)

    # Plot
    plot_results(
        train_losses, val_losses,
        y_prob, y_test,
        importance, event_names
    )
