import pandas as pd
import numpy as np
import os

## PATH
PROCESSED_PATH = "data/processed/supply_chain_clean.csv"
FEATURES_PATH = "data/features/features.csv"

def load_data():
    df = pd.read_csv(PROCESSED_PATH)
    print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def compute_features(df):

    # ── 1. OTIF RATE ─────────────────────────────────
    # From document: delivered on promised date WITH complete quantity
    # On Time: shipping_times <= lead_times
    # In Full: number_of_products_sold >= order_quantities
    # availability = current stock, NOT quantity delivered
    # so we use number_of_products_sold as delivered quantity
    # Threshold: below 85% = credit downgrade signal
    on_time = df['shipping_times'] <= df['lead_times']
    in_full = df['number_of_products_sold'] >= df['order_quantities']
    df['otif'] = (on_time & in_full).astype(int)
    otif_rate = df['otif'].mean()
    print(f"✅ OTIF Rate: {otif_rate:.1%}")
    if otif_rate < 0.85:
        print(f"⚠️  BELOW 85% threshold → Credit downgrade signal!")
    else:
        print(f"✅ Above 85% benchmark")

    # ── 2. PORT CONGESTION INDEX ──────────────────────
    # From document: >5 day avg delay = LC expiry risk
    # Proxy: actual shipping time - planned lead time
    # Positive = delayed, Negative = early
    df['shipment_delay_days'] = df['shipping_times'] - df['lead_times']
    avg_delay = df['shipment_delay_days'].mean()
    print(f"\n✅ Avg Shipment Delay: {avg_delay:.1f} days")
    if avg_delay > 5:
        print(f"⚠️  ABOVE 5 days → LC expiry risk!")
    else:
        print(f"✅ Below 5 day threshold")


    # ── 3. FREIGHT COST PER UNIT ──────────────────────
    # From document: total transportation spend / units shipped
    df['freight_cost_per_unit'] = (df['shipping_costs'] /
                                    df['number_of_products_sold'])
    print(f"\n✅ Avg Freight Cost per Unit: ${df['freight_cost_per_unit'].mean():.2f}")
    print(f"✅ Freight Cost Volatility (std): ${df['freight_cost_per_unit'].std():.2f}")

    # ── 4. INVENTORY TURNOVER ─────────────────────────
    # From document: COGS / average inventory value
    # Total COGS = manufacturing_costs × number_of_products_sold
    # Threshold: decline 8x to 5x = covenant breach risk
    df['total_cogs'] = df['manufacturing_costs'] * df['number_of_products_sold']
    df['inventory_value'] = df['stock_levels'] * df['price']
    df['inventory_turnover'] = (df['total_cogs'] /
                                 df['inventory_value'].replace(0, np.nan))
    median_turnover = df['inventory_turnover'].median()
    print(f"\n✅ Inventory Turnover (median): {median_turnover:.2f}x")
    if median_turnover < 5:
        print(f"⚠️  BELOW 5x → Working capital stress, covenant breach risk!")
    elif median_turnover < 8:
        print(f"⚠️  Between 5x-8x → Monitor closely")
    else:
        print(f"✅ Above 8x → Healthy")

    # Also cap extreme outliers for ML model
    df['inventory_turnover_capped'] = df['inventory_turnover'].clip(
        upper=df['inventory_turnover'].quantile(0.95)
    )
    print(f"✅ Capped turnover (95th pct): {df['inventory_turnover_capped'].max():.2f}x")

    # ── 5. CASH CONVERSION CYCLE (CCC) ───────────────
    # From document: Days Inventory Outstanding + Days sales Outstanding - Days Payable Outstanding
    # Threshold: extend +20 days = liquidity deterioration
    # Using available columns as proxies:
    # DIO proxy = lead_times (inventory days)
    # DSO proxy = shipping_times (days to complete sale)
    # DPO proxy = manufacturing_lead_time (days to pay suppliers)
    df['ccc'] = (df['lead_times'] +
                  df['shipping_times'] -
                  df['manufacturing_lead_time'])
    avg_ccc = df['ccc'].mean()
    print(f"✅ DIO proxy (lead_times):            {df['lead_times'].mean():.1f} days")
    print(f"✅ DSO proxy (shipping_times):         {df['shipping_times'].mean():.1f} days")
    print(f"✅ DPO proxy (manufacturing_lead_time):{df['manufacturing_lead_time'].mean():.1f} days")
    print(f"\n✅ Avg Cash Conversion Cycle: {avg_ccc:.1f} days")
    if avg_ccc > df['ccc'].median() + 20:
        print(f"⚠️  CCC extended +20 days → Liquidity deterioration!")
    else:
        print(f"✅ CCC within normal range")

    # ── 6. SUPPLIER CONCENTRATION ─────────────────────
    # From document: >60% single source = counterparty concentration risk
    # Per location HHI (row level)
    def compute_hhi(group):
        shares = (group['revenue_generated'] / 
                group['revenue_generated'].sum())
        return (shares ** 2).sum()

    hhi_per_location = (df.groupby('location')
                        .apply(compute_hhi)
                        .reset_index())
    hhi_per_location.columns = ['location', 
                                'supplier_concentration_hhi']
    df = df.merge(hhi_per_location, 
                on='location', how='left')

    # Global supplier share (for reporting)
    supplier_revenue = df.groupby('supplier_name')['revenue_generated'].sum()
    total_revenue = supplier_revenue.sum()
    supplier_share = supplier_revenue / total_revenue
    max_share = supplier_share.max()
    max_supplier = supplier_share.idxmax()

    print(f"✅ Supplier HHI per location computed")
    print(f"✅ HHI range: {df['supplier_concentration_hhi'].min():.3f} - {df['supplier_concentration_hhi'].max():.3f}")
    print(f"✅ Largest Supplier: {max_supplier} = {max_share:.1%}")
    if max_share > 0.60:
        print(f"⚠️  ABOVE 60% → Require credit insurance!")
    else:
        print(f"✅ Below 60% threshold")

    # ── 7. TRANSIT TIME VARIANCE ──────────────────────
    # From document: σ_LT increase +3 days = safety stock inflation
    # Safety Stock = 1.28 × σ_LT (at 90% service level)
    # Per supplier std of shipping times (row level)
    transit_var = (df.groupby('supplier_name')['shipping_times']
                    .std()
                    .reset_index())
    transit_var.columns = ['supplier_name', 'transit_time_variance']
    df = df.merge(transit_var, on='supplier_name', how='left')

    # Safety stock per supplier
    df['safety_stock_days'] = 1.28 * df['transit_time_variance']

    print(f"\n✅ Transit variance per supplier computed")
    print(f"✅ Variance range: {df['transit_time_variance'].min():.2f} - {df['transit_time_variance'].max():.2f} days")
    print(f"✅ Safety stock range: {df['safety_stock_days'].min():.2f} - {df['safety_stock_days'].max():.2f} days")

    # Flag suppliers above 3 days variance
    risky_suppliers = (df.groupby('supplier_name')['transit_time_variance']
                    .first()[lambda x: x > 3].index.tolist())
    num_risky = len(risky_suppliers)
    if risky_suppliers:
        print(f"⚠️  Suppliers above 3 days variance: {risky_suppliers}")
    else:
        print(f"✅ All suppliers within 3 day variance")


    # ── 8. TRANSPORT MODE RISK ────────────────────────
    # From document: each mode has different risk profile
    transport_otif = (df.groupby('transportation_modes')['otif']
                    .mean()
                    .reset_index())
    transport_otif.columns = ['transportation_modes', 
                            'transport_otif_rate']
    df = df.merge(transport_otif, 
                on='transportation_modes', how='left')

    transport_delay = (df.groupby('transportation_modes')
                        ['shipment_delay_days']
                        .mean()
                        .reset_index())
    transport_delay.columns = ['transportation_modes',
                                'transport_avg_delay']
    df = df.merge(transport_delay,
                on='transportation_modes', how='left')

    print(f"\n✅ Transport Mode Risk computed")
    print(f"OTIF by mode:")
    print(df.groupby('transportation_modes')['otif'].mean())



    # ── FINANCIAL SIGNAL SUMMARY ──────────────────────
    print("\n" + "="*50)
    print("📊 FINANCIAL SIGNAL SUMMARY")
    print("="*50)
    signals = {
        'OTIF Rate':              f"{otif_rate:.1%} {'⚠️ RISK' if otif_rate < 0.85 else '✅ OK'}",
        'Avg Delay (days)':       f"{avg_delay:.1f} {'⚠️ RISK' if avg_delay > 5 else '✅ OK'}",
        'Freight Cost/Unit':      f"${df['freight_cost_per_unit'].mean():.4f}",
        'Inventory Turnover':     f"{median_turnover:.2f}x {'⚠️ RISK' if median_turnover < 5 else '✅ OK'}",
        'CCC (days)':             f"{avg_ccc:.1f}",
        'Supplier Concentration': f"{max_share:.1%} {'⚠️ RISK' if max_share > 0.60 else '✅ OK'}",
        'Transit Variance (σ)':   f"{df['transit_time_variance'].mean():.2f} {'⚠️ RISK' if num_risky > 0 else '✅ OK'}"
    }
    for metric, value in signals.items():
        print(f"  {metric:<28} → {value}")

    return df

def save_features(df):
    os.makedirs("data/features", exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"\n✅ Saved to: {FEATURES_PATH}")
    print(f"📊 Final shape: {df.shape[0]} rows, {df.shape[1]} columns")

if __name__ == "__main__":
    df = load_data()
    df = compute_features(df)
    save_features(df)
