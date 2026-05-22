import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── PATHS ────────────────────────────────
FEATURES_PATH = "data/features/features.csv"
EDA_OUTPUT = "data/processed/eda_plots"

def load_data():
    df = pd.read_csv(FEATURES_PATH)
    print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def plot_otif(df):
    """OTIF Rate by Supplier"""
    plt.figure(figsize=(8, 5))

    otif_by_supplier = df.groupby('supplier_name')['otif'].mean() * 100

    colors = ['red' if x < 85 else 'green' for x in otif_by_supplier]

    otif_by_supplier.plot(kind='bar', color=colors)
    plt.axhline(y=85, color='orange', linestyle='--', label='85% Threshold')
    plt.title('OTIF Rate by Supplier')
    plt.xlabel('Supplier')
    plt.ylabel('OTIF Rate (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{EDA_OUTPUT}/otif_by_supplier.png')
    plt.show()
    print("✅ OTIF plot saved")

def plot_delay(df):
    """Shipment Delay Distribution"""
    plt.figure(figsize=(8, 5))

    sns.histplot(df['shipment_delay_days'], bins=20, color='blue', kde=True)
    plt.axvline(x=5, color='red', linestyle='--', label='>5 days = LC Risk')
    plt.axvline(x=0, color='green', linestyle='--', label='On Time')
    plt.title('Shipment Delay Distribution')
    plt.xlabel('Delay Days (negative = early)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{EDA_OUTPUT}/delay_distribution.png')
    plt.show()
    print("✅ Delay plot saved")

def plot_inventory_turnover(df):
    """Inventory Turnover by Product Type"""
    plt.figure(figsize=(8, 5))

    turnover_by_product = df.groupby('product_type')['inventory_turnover_capped'].median()
    
    colors = ['red' if x < 5 else 'orange' if x < 8 else 'green' 
              for x in turnover_by_product]
    
    turnover_by_product.plot(kind='bar', color=colors)
    plt.axhline(y=5, color='red', linestyle='--', label='5x Covenant Breach')
    plt.axhline(y=8, color='orange', linestyle='--', label='8x Warning')
    plt.title('Inventory Turnover by Product Type')
    plt.xlabel('Product Type')
    plt.ylabel('Inventory Turnover (x)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{EDA_OUTPUT}/inventory_turnover.png')
    plt.show()
    print("✅ Inventory turnover plot saved")

def plot_correlation(df):
    """Correlation between SC Metrics"""
    plt.figure(figsize=(10, 8))
    
    # Check what columns we actually have
    print(f"\n📋 Available columns:\n{list(df.columns)}")
    
    # Only use columns that exist in our dataset
    cols = [
        'otif',
        'shipment_delay_days',
        'freight_cost_per_unit',
        'inventory_turnover_capped',
        'ccc',
        'transit_time_variance',
        'defect_rates',
        'supplier_concentration_hhi'
    ]
    
    corr = df[cols].corr()
    
    sns.heatmap(corr, 
                annot=True, 
                fmt='.2f',
                cmap='RdYlGn',
                center=0)
    plt.title('Correlation: Supply Chain Metrics')
    plt.tight_layout()
    plt.savefig(f'{EDA_OUTPUT}/correlation_heatmap.png')
    plt.show()
    print("✅ Correlation heatmap saved")

def plot_carrier_reliability(df):
    """Carrier Reliability Score"""
    plt.figure(figsize=(8, 5))

    carrier_otif = df.groupby('shipping_carriers')['otif'].mean() * 100
    
    colors = ['red' if x < 85 else 'green' for x in carrier_otif]
    
    carrier_otif.plot(kind='bar', color=colors)
    plt.axhline(y=85, color='orange', linestyle='--', label='85% Threshold')
    plt.title('Carrier Reliability (OTIF Rate)')
    plt.xlabel('Carrier')
    plt.ylabel('OTIF Rate (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{EDA_OUTPUT}/carrier_reliability.png')
    plt.show()
    print("✅ Carrier reliability plot saved")

def plot_transport_risk(df):
    """Risk by Transportation Mode"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # OTIF by transport mode
    otif_by_transport = (df.groupby('transportation_modes')['otif']
                          .mean() * 100)
    colors = ['red' if x < 85 else 'green' 
              for x in otif_by_transport]
    otif_by_transport.plot(kind='bar', 
                           color=colors, 
                           ax=axes[0])
    axes[0].axhline(y=85, color='orange', 
                    linestyle='--', 
                    label='85% Threshold')
    axes[0].set_title('OTIF Rate by Transport Mode')
    axes[0].set_xlabel('Transport Mode')
    axes[0].set_ylabel('OTIF Rate (%)')
    axes[0].legend()

    # Delay by transport mode
    delay_by_transport = (df.groupby('transportation_modes')
                           ['shipment_delay_days'].mean())
    colors2 = ['red' if x > 5 else 'green' 
               for x in delay_by_transport]
    delay_by_transport.plot(kind='bar',
                            color=colors2,
                            ax=axes[1])
    axes[1].axhline(y=5, color='red',
                    linestyle='--',
                    label='>5 days = LC Risk')
    axes[1].axhline(y=0, color='green',
                    linestyle='--',
                    label='On Time')
    axes[1].set_title('Avg Delay by Transport Mode')
    axes[1].set_xlabel('Transport Mode')
    axes[1].set_ylabel('Avg Delay (days)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{EDA_OUTPUT}/transport_risk.png')
    plt.show()
    print("✅ Transport risk plot saved")


if __name__ == "__main__":
    # Create output folder
    os.makedirs(EDA_OUTPUT, exist_ok=True)
    
    df = load_data()
    
    print("\n📊 Generating plots...")
    plot_otif(df)
    plot_delay(df)
    plot_inventory_turnover(df)
    plot_correlation(df)
    plot_carrier_reliability(df)
    plot_transport_risk(df)
    
    print(f"📁 All plots saved to: {EDA_OUTPUT}")