import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

from scorer.ml_model import IELTSModel
from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features

def main():
    print("Loading datasets...")
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    test_df = pd.read_csv("data/test.csv")
    
    # 1. Band Distribution
    print("="*60)
    print("1. BAND DISTRIBUTION")
    print("="*60)
    
    bands = sorted(train_df['Overall'].unique())
    print(f"{'Band':>5} | {'Train':>6} | {'Val':>6} | {'Test':>6}")
    print("-" * 35)
    for b in bands:
        n_tr = sum(train_df['Overall'] == b)
        n_va = sum(val_df['Overall'] == b)
        n_te = sum(test_df['Overall'] == b)
        print(f"{b:>5.1f} | {n_tr:>6} | {n_va:>6} | {n_te:>6}")
        
    # Prepare features for train_df
    print("\nExtracting features for train.csv...")
    train_data = _load_dataset(Path("data/train.csv"))
    
    features_list = []
    true_bands = []
    
    for sample in train_data:
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        true_band = sample.get("band", 5.0)
        
        if not essay.strip() or not prompt.strip():
            continue
            
        raw_feat = extract_all_features(essay, prompt)
        features_list.append(raw_feat)
        true_bands.append(true_band)
        
    feat_df = pd.DataFrame(features_list)
    feat_df['true_band'] = true_bands
    
    # 2. Feature Correlation
    print("\n" + "="*60)
    print("2. FEATURE CORRELATION WITH TRUE BAND (Train Set)")
    print("="*60)
    correlations = {}
    for col in FEATURE_KEYS:
        if col in feat_df.columns:
            corr, _ = pearsonr(feat_df[col], feat_df['true_band'])
            correlations[col] = corr
            
    corr_series = pd.Series(correlations).sort_values(ascending=False)
    for feat, corr in corr_series.items():
        print(f"{feat:>30} : {corr:+.3f}")
        
    # 3. Feature Importance (from LightGBM)
    print("\n" + "="*60)
    print("3. LIGHTGBM FEATURE IMPORTANCE")
    print("="*60)
    model = IELTSModel.load("data/ml_model.pkl")
    lgb_model = model.models.get("overall")
    if lgb_model:
        # Get importances (split and gain)
        importance_split = lgb_model.booster_.feature_importance(importance_type='split')
        importance_gain = lgb_model.booster_.feature_importance(importance_type='gain')
        
        imp_df = pd.DataFrame({
            'Feature': FEATURE_KEYS,
            'Split': importance_split,
            'Gain': importance_gain
        }).sort_values(by='Gain', ascending=False)
        
        print(f"{'Feature':>30} | {'Split':>6} | {'Gain':>8}")
        print("-" * 50)
        for _, row in imp_df.iterrows():
            print(f"{row['Feature']:>30} | {row['Split']:>6.0f} | {row['Gain']:>8.1f}")
            
        top_features = imp_df['Feature'].head(5).tolist()
    else:
        print("Overall model not found.")
        top_features = corr_series.head(5).index.tolist()
        
    # 4. Feature Distribution by Band (Top 5 features)
    print("\n" + "="*60)
    print("4. FEATURE DISTRIBUTION BY BAND (Top 5 Features by Gain)")
    print("="*60)
    
    for feat in top_features:
        print(f"\n--- {feat} ---")
        print(f"{'Band':>5} | {'Mean':>8} | {'Median':>8} | {'Std':>8}")
        print("-" * 40)
        grouped = feat_df.groupby('true_band')[feat]
        for b in sorted(feat_df['true_band'].unique()):
            g = grouped.get_group(b)
            print(f"{b:>5.1f} | {g.mean():>8.3f} | {g.median():>8.3f} | {g.std():>8.3f}")

    # 5. Prediction Distribution
    print("\n" + "="*60)
    print("5. RAW PREDICTION DISTRIBUTION BY TRUE BAND (Train Set)")
    print("="*60)
    
    preds_raw = []
    for raw_feat in features_list:
        norm = normalize_features(raw_feat)
        X = features_to_array(norm, FEATURE_KEYS).reshape(1, -1)
        # using the array directly or .item() since it's 1 value
        pred = float(model.predict(X)["overall"])
        preds_raw.append(pred)
        
    feat_df['pred_raw'] = preds_raw
    
    print(f"{'True':>5} | {'N':>5} | {'Mean Pred':>9} | {'Median':>8} | {'Std':>6} | {'Bias':>6}")
    print("-" * 55)
    grouped_pred = feat_df.groupby('true_band')['pred_raw']
    for b in sorted(feat_df['true_band'].unique()):
        g = grouped_pred.get_group(b)
        mean_p = g.mean()
        med_p = g.median()
        std_p = g.std()
        n_p = len(g)
        bias = mean_p - b
        print(f"{b:>5.1f} | {n_p:>5} | {mean_p:>9.3f} | {med_p:>8.3f} | {std_p:>6.3f} | {bias:>+6.3f}")

if __name__ == "__main__":
    main()
