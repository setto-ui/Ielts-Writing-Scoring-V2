import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
from tqdm import tqdm

from evaluate import _load_dataset, extract_all_features, FEATURE_KEYS
from scorer.normalizer import normalize_features

def main():
    print("Loading train.csv...")
    dataset = _load_dataset(Path("data/train.csv"))
    
    features_list = []
    true_bands = []
    
    print("Extracting features...")
    for sample in tqdm(dataset):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        true_band = sample.get("band", 5.0)
        
        if not essay.strip() or not prompt.strip():
            continue
            
        raw_feat = extract_all_features(essay, prompt)
        features_list.append(raw_feat)
        true_bands.append(true_band)
        
    df = pd.DataFrame(features_list)
    df['true_band'] = true_bands
    
    print("\n============================================================")
    print("  FEATURE LEVEL STATISTICS (Train Set)")
    print("============================================================")
    print(f"{'Feature':>25} | {'Mean':>7} | {'Std':>7} | {'Min':>7} | {'Max':>7} | {'Unique':>6} | {'Corr':>6}")
    print("-" * 80)
    
    stats_list = []
    
    for feat in FEATURE_KEYS:
        if feat not in df.columns:
            continue
        
        series = df[feat]
        mean_val = series.mean()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()
        unique_cnt = series.nunique()
        
        if std_val > 1e-6:
            corr, _ = pearsonr(series, df['true_band'])
        else:
            corr = 0.0
            
        stats_list.append((feat, mean_val, std_val, min_val, max_val, unique_cnt, corr))
        
    # Sort by absolute correlation
    stats_list.sort(key=lambda x: abs(x[6]), reverse=True)
    
    for feat, mean_val, std_val, min_val, max_val, unique_cnt, corr in stats_list:
        print(f"{feat:>25} | {mean_val:>7.3f} | {std_val:>7.3f} | {min_val:>7.3f} | {max_val:>7.3f} | {unique_cnt:>6} | {corr:>+6.3f}")

if __name__ == "__main__":
    main()
