import pandas as pd
import json
from sklearn.model_selection import train_test_split
from pathlib import Path

def main():
    print("Loading dataset...")
    df = pd.read_csv("data/ielts_writing_dataset.csv")
    
    # 1. Deduplicate
    df = df.dropna(subset=["Overall"])
    df['Overall'] = df['Overall'].astype(float)
    df['normalized_essay'] = df['Essay'].astype(str).str.strip().str.lower()
    df_dedup = df.drop_duplicates(subset=['normalized_essay'], keep='first').copy()
    
    print(f"Original shape: {df.shape}, Dedup shape: {df_dedup.shape}")
    
    # 2. Filter out rare bands to allow stratification
    # Drop bands that have fewer than 6 samples (Band 1.0, 3.0, 3.5)
    counts = df_dedup['Overall'].value_counts()
    valid_bands = counts[counts >= 6].index
    
    df_valid = df_dedup[df_dedup['Overall'].isin(valid_bands)].copy()
    print(f"Shape after dropping rare bands (<6 samples): {df_valid.shape}")
    
    # Reset index so we can use row indices or IDs if we don't have a stable ID
    # We will use the original CSV index as 'sample_id'
    df_valid['sample_id'] = df_valid.index
    
    # 3. Stratified Split 70/15/15
    # First split: Train (70%) and Temp (30%)
    train_df, temp_df = train_test_split(
        df_valid, 
        test_size=0.30, 
        random_state=42, 
        stratify=df_valid['Overall']
    )
    
    # Second split: Val (50% of Temp = 15%) and Test (50% of Temp = 15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df['Overall']
    )
    
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Drop temp columns
    for d in [train_df, val_df, test_df]:
        d.drop(columns=['normalized_essay'], inplace=True, errors='ignore')
        
    # 4. Save to CSV
    data_dir = Path("data")
    train_df.to_csv(data_dir / "train.csv", index=False)
    val_df.to_csv(data_dir / "val.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)
    
    # 5. Create Manifest
    manifest = {
        "random_state": 42,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "total_samples": len(df_valid),
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "ids": {
            "train": train_df['sample_id'].tolist(),
            "val": val_df['sample_id'].tolist(),
            "test": test_df['sample_id'].tolist()
        }
    }
    
    manifest_path = data_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Saved splits and manifest to {data_dir}/")
    
if __name__ == "__main__":
    main()
