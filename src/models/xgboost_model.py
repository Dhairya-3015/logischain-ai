import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report,
                             roc_auc_score,
                             confusion_matrix)
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os

# ── PATHS ────────────────────────────────
FEATURES_PATH = "data/features/features.csv"
MODEL_OUTPUT = "data/processed/model_results"

def load_data():
    df = pd.read_csv(FEATURES_PATH)
    print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def add_synthetic_data(df):
    """
    Add synthetic data with realistic variation
    """
    print("\n🔧 Adding synthetic data...")
    np.random.seed(42)

    n_synthetic = 400
    numeric_cols = df.select_dtypes(
                    include=[np.number]).columns.tolist()

    synthetic_rows = []
    for _ in range(n_synthetic):
        base_row = df.sample(1).iloc[0].copy()

        for col in numeric_cols:
            if col != 'target':
                # Larger noise = more realistic variation
                noise = np.random.normal(0, 0.3)  # 30% noise
                base_row[col] = base_row[col] * (1 + noise)

                # Sometimes flip the target randomly
                if np.random.random() < 0.1:  # 10% chance
                    base_row['inspection_results'] = np.random.choice(
                        ['Pass', 'Fail']
                    )

        synthetic_rows.append(base_row)

    synthetic_df = pd.DataFrame(synthetic_rows)
    combined_df = pd.concat([df, synthetic_df], 
                             ignore_index=True)

    print(f"✅ Original rows:  {len(df)}")
    print(f"✅ Synthetic rows: {n_synthetic}")
    print(f"✅ Total rows:     {len(combined_df)}")

    return combined_df

def prepare_data(df):
    print("\n🔧 Preparing data...")

    # Drop pending
    df = df[df['inspection_results'] != 'Pending'].copy()
    print(f"✅ After dropping Pending: {df.shape[0]} rows")

    # Add synthetic data
    df = add_synthetic_data(df)

    # Target variable
    df['target'] = (df['inspection_results'] == 'Fail').astype(int)
    print(f"\n✅ Target distribution:")
    print(f"   Pass (0): {(df['target']==0).sum()} rows")
    print(f"   Fail (1): {(df['target']==1).sum()} rows")
    print(f"   Fail rate: {df['target'].mean():.1%}")

    # Encode categorical
    le = LabelEncoder()
    df['carrier_encoded']   = le.fit_transform(df['shipping_carriers'].astype(str))
    df['location_encoded']  = le.fit_transform(df['location'].astype(str))
    df['product_encoded']   = le.fit_transform(df['product_type'].astype(str))
    df['route_encoded']     = le.fit_transform(df['routes'].astype(str))
    df['transport_encoded'] = le.fit_transform(df['transportation_modes'].astype(str))

    # Features
    feature_cols = [
        'otif',
        'shipment_delay_days',
        'freight_cost_per_unit',
        'inventory_turnover_capped',
        'ccc',
        'transit_time_variance',
        'defect_rates',
        'supplier_concentration_hhi',
        'safety_stock_days',
        'lead_times',
        'shipping_times',
        'manufacturing_lead_time',
        'price',
        'revenue_generated',
        'stock_levels',
        'carrier_encoded',
        'location_encoded',
        'product_encoded',
        'route_encoded',
        'transport_encoded'
    ]

    X = df[feature_cols]
    y = df['target']

    print(f"\n✅ Total features: {len(feature_cols)}")
    return X, y, feature_cols

def train_model(X, y):
    print("\n🔧 Training XGBoost model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"✅ Train: {X_train.shape[0]} rows")
    print(f"✅ Test:  {X_test.shape[0]} rows")

    # Class weight
    fail_count = y_train.sum()
    pass_count = (y_train == 0).sum()
    scale_pos_weight = pass_count / fail_count
    print(f"✅ scale_pos_weight: {scale_pos_weight:.2f}")

    # XGBoost
    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=4,        # Reduced to avoid overfitting
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
        verbose=100
    )

    # Cross validation score
    cv_scores = cross_val_score(
        xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42
        ),
        X, y, cv=5, scoring='roc_auc'
    )
    print(f"\n✅ Cross Validation AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    print("\n📊 Evaluating model...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    gini = 2 * auc - 1

    print(f"\n{'='*50}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"AUC-ROC: {auc:.3f}  (target > 0.82)")
    print(f"Gini:    {gini:.3f}  (target > 0.55)")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Pass', 'Fail'],
                                zero_division=0))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_prob

def plot_shap(model, X_train, feature_cols):
    print("\n📊 Computing SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values,
                      X_train,
                      feature_names=feature_cols,
                      show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{MODEL_OUTPUT}/shap_importance.png')
    plt.show()
    print("✅ SHAP plot saved")

def plot_risk_scores(y_prob, y_test):
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob[y_test==0], bins=20,
             alpha=0.6, color='green', label='Pass')
    plt.hist(y_prob[y_test==1], bins=20,
             alpha=0.6, color='red', label='Fail')
    plt.axvline(x=0.5, color='black',
                linestyle='--', label='Decision Boundary')
    plt.title('Risk Score Distribution')
    plt.xlabel('Predicted Risk Score')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{MODEL_OUTPUT}/risk_scores.png')
    plt.show()
    print("✅ Risk score plot saved")

if __name__ == "__main__":
    os.makedirs(MODEL_OUTPUT, exist_ok=True)

    df = load_data()
    X, y, feature_cols = prepare_data(df)
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    y_prob = evaluate_model(model, X_test, y_test)
    plot_shap(model, X_train, feature_cols)
    plot_risk_scores(y_prob, y_test)
