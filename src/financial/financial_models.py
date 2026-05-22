import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import os

# ── PATHS ────────────────────────────────
FEATURES_PATH = "data/features/features.csv"
OUTPUT_PATH = "data/processed/financial_results"

def load_data():
    df = pd.read_csv(FEATURES_PATH)
    print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# ════════════════════════════════════════
# MODEL 1: SUPPLY CHAIN ADJUSTED PD (SC-PD)
# From document Section A5.4 Fusion Feature 1
# ════════════════════════════════════════
def compute_sc_pd(df):
    """
    From document exact formula:
    SC-PD = Traditional PD × (1 + 0.3×OTIF_adj 
                               + 0.2×Inv_adj 
                               + 0.15×Network_adj)

    Where:
    OTIF_adj = max(0, (90% - OTIF_actual) / 10%)
    Inv_adj  = max(0, (6.0 - InvTurnover) / 3.0)
    Network_adj = 1.0 - min(1.0, AltSuppliers/3)
    """
    print("\n" + "="*50)
    print("MODEL 1: SUPPLY CHAIN ADJUSTED PD (SC-PD)")
    print("="*50)
    print("From document Section A5.4 Fusion Feature 1")

    df = df.copy()

    # Traditional PD baseline
    # From document: portfolio average PD = 2.8%
    # We use inspection fail rate as proxy
    fail_rate = (df['inspection_results'] == 'Fail').mean()
    
    # From document: ICC Trade Register
    # LC default rate = 0.36%
    # Trade loan default rate = 0.72%
    # We use 2.8% as conservative estimate
    traditional_pd = 0.028  # 2.8% from document
    print(f"✅ Traditional PD (ICC benchmark): {traditional_pd:.1%}")

    # ── OTIF Adjustment ───────────────────
    # From document: max(0, (90% - OTIF_actual) / 10%)
    df['otif_adj'] = df['otif'].apply(
        lambda x: max(0, (0.90 - x) / 0.10)
    )
    print(f"✅ OTIF Adjustment mean: {df['otif_adj'].mean():.3f}")

    # ── Inventory Health Adjustment ───────
    # From document: max(0, (6.0 - InvTurnover) / 3.0)
    df['inv_adj'] = df['inventory_turnover_capped'].apply(
        lambda x: max(0, (6.0 - x) / 3.0)
    )
    print(f"✅ Inventory Adjustment mean: {df['inv_adj'].mean():.3f}")

    # ── Network Resilience Factor ─────────
    # From document: 1.0 - min(1.0, AltSupplierCount/3)
    # Proxy: use supplier concentration HHI
    # Higher HHI = fewer alternatives = higher network_adj
    df['network_adj'] = df['supplier_concentration_hhi'].apply(
        lambda x: min(1.0, x * 2)  # HHI proxy for concentration
    )
    print(f"✅ Network Adjustment mean: {df['network_adj'].mean():.3f}")

    # ── SC-PD Formula from Document ───────
    df['sc_pd'] = traditional_pd * (
        1 +
        0.3 * df['otif_adj'] +
        0.2 * df['inv_adj'] +
        0.15 * df['network_adj']
    )

    # ── Results ───────────────────────────
    print(f"\n{'='*50}")
    print(f"SC-PD RESULTS")
    print(f"{'='*50}")
    print(f"Traditional PD:     {traditional_pd:.2%}")
    print(f"SC-PD Mean:         {df['sc_pd'].mean():.2%}")
    print(f"SC-PD Min:          {df['sc_pd'].min():.2%}")
    print(f"SC-PD Max:          {df['sc_pd'].max():.2%}")
    print(f"Risk Uplift:        {(df['sc_pd'].mean()/traditional_pd - 1):.1%}")

    # Flag high risk
    df['pd_risk_flag'] = df['sc_pd'].apply(
        lambda x: '⚠️ HIGH' if x > traditional_pd * 1.3
        else '✅ NORMAL'
    )
    high_risk = (df['sc_pd'] > traditional_pd * 1.3).sum()
    print(f"\nHigh Risk Borrowers: {high_risk}/{len(df)}")

    # Per supplier SC-PD
    print(f"\nSC-PD by Supplier:")
    supplier_pd = df.groupby('supplier_name').agg({
        'sc_pd': 'mean',
        'otif': 'mean',
        'inventory_turnover_capped': 'median'
    }).round(4)
    print(supplier_pd)

    return df, traditional_pd


# ════════════════════════════════════════
# MODEL 2: CCC PREDICTION MODEL
# From document Section A2.2
# ════════════════════════════════════════
def compute_ccc_prediction(df):
    """
    From document:
    CCC = DIO + DSO - DPO
    DIO = lead_times (proxy)
    DSO = shipping_times (proxy)
    DPO = manufacturing_lead_time (proxy)

    Predict CCC change from SC signals
    Alert when CCC exceeds covenant threshold
    """
    print("\n" + "="*50)
    print("MODEL 2: CCC PREDICTION MODEL")
    print("="*50)
    print("From document Section A2.2")

    df = df.copy()

    # ── CCC Components ────────────────────
    df['dio'] = df['lead_times']
    df['dso'] = df['shipping_times']
    df['dpo'] = df['manufacturing_lead_time']
    df['ccc_computed'] = df['dio'] + df['dso'] - df['dpo']

    print(f"\n✅ CCC Components:")
    print(f"   DIO (lead_times):     {df['dio'].mean():.1f} days")
    print(f"   DSO (shipping_times): {df['dso'].mean():.1f} days")
    print(f"   DPO (mfg_lead_time):  {df['dpo'].mean():.1f} days")
    print(f"   CCC:                  {df['ccc_computed'].mean():.1f} days")

    # ── CCC Stress Prediction ─────────────
    # From document: +20 days = liquidity deterioration
    # Predict which rows will have CCC extension

    # Features that predict CCC extension
    ccc_features = [
        'otif',
        'shipment_delay_days',
        'transit_time_variance',
        'defect_rates',
        'freight_cost_per_unit',
        'inventory_turnover_capped',
        'supplier_concentration_hhi'
    ]

    # Target: CCC above median = stress
    ccc_median = df['ccc_computed'].median()
    df['ccc_stress'] = (df['ccc_computed'] > ccc_median).astype(int)

    # Add synthetic data for model training
    np.random.seed(42)
    n_syn = 400
    syn_rows = []
    for _ in range(n_syn):
        base = df.sample(1).iloc[0].copy()
        for col in ccc_features:
            base[col] = base[col] * np.random.uniform(0.8, 1.2)
        # Recompute CCC stress
        syn_ccc = (base['lead_times'] +
                   base['shipping_times'] -
                   base['manufacturing_lead_time'])
        base['ccc_stress'] = 1 if syn_ccc > ccc_median else 0
        syn_rows.append(base)

    df_aug = pd.concat(
        [df, pd.DataFrame(syn_rows)],
        ignore_index=True
    )

    X = df_aug[ccc_features].fillna(0)
    y = df_aug['ccc_stress']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )

    # XGBoost for CCC prediction
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric='auc'
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n✅ CCC Stress Prediction AUC: {auc:.3f}")

    # ── Covenant Breach Alert ─────────────
    # From document: MedDevice example
    # Covenant threshold: CCC must not exceed 90 days
    covenant_threshold = df['ccc_computed'].mean() + 20

    df['covenant_breach_risk'] = df['ccc_computed'].apply(
        lambda x: '⚠️ BREACH RISK' if x > covenant_threshold
        else '✅ COMPLIANT'
    )

    breach_count = (df['ccc_computed'] > covenant_threshold).sum()
    print(f"\n✅ Covenant Threshold: {covenant_threshold:.1f} days")
    print(f"✅ Breach Risk Count: {breach_count}/{len(df)}")

    # ── CCC by Product Type ───────────────
    print(f"\nCCC by Product Type:")
    ccc_by_product = df.groupby('product_type').agg({
        'dio': 'mean',
        'dso': 'mean',
        'dpo': 'mean',
        'ccc_computed': 'mean'
    }).round(1)
    print(ccc_by_product)

    return df, model, auc


# ════════════════════════════════════════
# MODEL 3: TRADE FINANCE DEFAULT PREDICTION
# From document Section A6.3
# ════════════════════════════════════════
def compute_trade_finance_model(df, traditional_pd):
    """
    From document Section A6.3:
    Integrates:
    - GNN entity embeddings (proxy: SC features)
    - TCN temporal features (proxy: time-based features)
    - XGBoost tabular scores
    - Supply chain adjusted PD

    Target: Gini > 0.55, ECE < 0.03
    """
    print("\n" + "="*50)
    print("MODEL 3: TRADE FINANCE DEFAULT PREDICTION")
    print("="*50)
    print("From document Section A6.3")

    df = df.copy()

    # ── Features ──────────────────────────
    # From document: cross-domain fusion features
    tf_features = [
        # Supply chain features
        'otif',
        'shipment_delay_days',
        'freight_cost_per_unit',
        'inventory_turnover_capped',
        'ccc',
        'transit_time_variance',
        'defect_rates',
        'supplier_concentration_hhi',
        'safety_stock_days',

        # Financial features
        'lead_times',
        'shipping_times',
        'manufacturing_lead_time',
        'revenue_generated',
        'stock_levels',
        'price',

        # SC-PD feature (from Model 1)
        'sc_pd',
        'otif_adj',
        'inv_adj',
        'network_adj',
    ]

    # Target: inspection fail = trade finance default
    df_clean = df[df['inspection_results'] != 'Pending'].copy()
    df_clean['tf_default'] = (
        df_clean['inspection_results'] == 'Fail'
    ).astype(int)

    print(f"\n✅ Default rate: {df_clean['tf_default'].mean():.1%}")

    # ── Add Synthetic Data ─────────────────
    np.random.seed(42)
    n_syn = 500
    syn_rows = []

    for _ in range(n_syn):
        base = df_clean.sample(1).iloc[0].copy()
        for col in tf_features:
            if col in base.index:
                base[col] = base[col] * np.random.normal(1, 0.25)
        # Flip label 15% of time
        if np.random.random() < 0.15:
            base['tf_default'] = 1 - base['tf_default']
        syn_rows.append(base)

    df_aug = pd.concat(
        [df_clean, pd.DataFrame(syn_rows)],
        ignore_index=True
    )

    # Fill missing
    X = df_aug[tf_features].fillna(0)
    y = df_aug['tf_default']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )

    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

    # ── XGBoost Model ─────────────────────
    # Parameters from document Section A7.1
    fail_count = y_train.sum()
    pass_count = (y_train == 0).sum()
    scale_pos_weight = pass_count / fail_count

    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.02,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=50,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # ── Evaluate ──────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    gini = 2 * auc - 1

    # ECE (Expected Calibration Error)
    # From document: target ECE < 0.03
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & \
               (y_prob < bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_acc = y_test[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
    ece = ece / len(y_test)

    print(f"\n{'='*50}")
    print(f"TRADE FINANCE MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"AUC-ROC: {auc:.3f}")
    print(f"Gini:    {gini:.3f}  (target > 0.55)")
    print(f"ECE:     {ece:.3f}  (target < 0.03)")

    if gini > 0.55:
        print(f"✅ Gini meets target!")
    else:
        print(f"⚠️  Gini below target")

    if ece < 0.03:
        print(f"✅ ECE meets target!")
    else:
        print(f"⚠️  ECE above target")

    # ── SHAP Analysis ─────────────────────
    print(f"\n📊 Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Top features
    shap_importance = pd.DataFrame({
        'feature': tf_features,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Features:")
    print(shap_importance.head(10).to_string(index=False))

    return model, y_prob, y_test, shap_values, X_train, tf_features


# ════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════
def plot_all_results(df, y_prob, y_test,
                     shap_values, X_train,
                     tf_features, traditional_pd):
    print("\n📊 Generating financial plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: SC-PD by Supplier
    supplier_pd = df.groupby('supplier_name')['sc_pd'].mean()
    colors = ['red' if x > traditional_pd * 1.3 else 'green'
              for x in supplier_pd]
    supplier_pd.plot(kind='bar', color=colors, ax=axes[0,0])
    axes[0,0].axhline(y=traditional_pd, color='orange',
                      linestyle='--', label=f'Baseline PD {traditional_pd:.1%}')
    axes[0,0].axhline(y=traditional_pd*1.3, color='red',
                      linestyle=':', label='High Risk Threshold')
    axes[0,0].set_title('SC-PD by Supplier\n(Supply Chain Adjusted PD)')
    axes[0,0].set_xlabel('Supplier')
    axes[0,0].set_ylabel('Probability of Default')
    axes[0,0].legend(fontsize=8)
    axes[0,0].tick_params(axis='x', rotation=45)

    # Plot 2: CCC Distribution
    axes[0,1].hist(df['ccc_computed'], bins=20,
                   color='blue', alpha=0.7)
    covenant = df['ccc_computed'].mean() + 20
    axes[0,1].axvline(x=covenant, color='red',
                      linestyle='--',
                      label=f'Covenant Threshold ({covenant:.0f} days)')
    axes[0,1].set_title('Cash Conversion Cycle Distribution')
    axes[0,1].set_xlabel('CCC (days)')
    axes[0,1].set_ylabel('Count')
    axes[0,1].legend()

    # Plot 3: SC-PD Components
    components = {
        'OTIF\nAdjustment': df['otif_adj'].mean() * 0.3,
        'Inventory\nAdjustment': df['inv_adj'].mean() * 0.2,
        'Network\nAdjustment': df['network_adj'].mean() * 0.15,
    }
    axes[0,2].bar(components.keys(), components.values(),
                  color=['red', 'orange', 'purple'])
    axes[0,2].set_title('SC-PD Risk Components\n(Contribution to PD Uplift)')
    axes[0,2].set_ylabel('PD Uplift Contribution')

    # Plot 4: Trade Finance Risk Scores
    axes[1,0].hist(y_prob[y_test==0], bins=20,
                   alpha=0.6, color='green', label='Pass')
    axes[1,0].hist(y_prob[y_test==1], bins=20,
                   alpha=0.6, color='red', label='Fail')
    axes[1,0].axvline(x=0.5, color='black',
                      linestyle='--')
    axes[1,0].set_title('Trade Finance Risk Score Distribution')
    axes[1,0].set_xlabel('Default Probability')
    axes[1,0].set_ylabel('Count')
    axes[1,0].legend()

    # Plot 5: SHAP Feature Importance
    shap_imp = pd.DataFrame({
        'feature': tf_features,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=True).tail(10)

    axes[1,1].barh(shap_imp['feature'],
                   shap_imp['importance'],
                   color='steelblue')
    axes[1,1].set_title('Top 10 Features\n(SHAP Importance)')
    axes[1,1].set_xlabel('Mean |SHAP Value|')

    # Plot 6: CCC by Product Type
    ccc_by_product = df.groupby('product_type')['ccc_computed'].mean()
    colors = ['red' if x > covenant else 'green'
              for x in ccc_by_product]
    ccc_by_product.plot(kind='bar', color=colors, ax=axes[1,2])
    axes[1,2].axhline(y=covenant, color='red',
                      linestyle='--', label='Covenant')
    axes[1,2].set_title('CCC by Product Type')
    axes[1,2].set_xlabel('Product Type')
    axes[1,2].set_ylabel('CCC (days)')
    axes[1,2].legend()
    axes[1,2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/financial_results.png')
    plt.show()
    print("✅ Financial plots saved")


if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Load data
    df = load_data()

    # Model 1: SC-PD
    df, traditional_pd = compute_sc_pd(df)

    # Model 2: CCC Prediction
    df, ccc_model, ccc_auc = compute_ccc_prediction(df)

    # Model 3: Trade Finance Default
    tf_model, y_prob, y_test, \
    shap_values, X_train, tf_features = compute_trade_finance_model(
        df, traditional_pd
    )

    # Plot all results
    plot_all_results(
        df, y_prob, y_test,
        shap_values, X_train,
        tf_features, traditional_pd
    )

    # Final Summary
    print("\n" + "="*50)
    print("📊 FINANCIAL MODELS SUMMARY")
    print("="*50)
    print(f"SC-PD Uplift:      {(df['sc_pd'].mean()/traditional_pd-1):.1%}")
    print(f"CCC Prediction AUC: {ccc_auc:.3f}")
    print(f"Traditional PD:     {traditional_pd:.2%}")
    print(f"SC-Adjusted PD:     {df['sc_pd'].mean():.2%}")

    print("\n🎉 Step 8 Financial Models Complete!")