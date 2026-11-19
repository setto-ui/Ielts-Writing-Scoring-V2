import pandas as pd
import json

def main():
    print("Loading dataset...")
    df = pd.read_csv("data/ielts_writing_dataset.csv")
    print(f"Initial shape: {df.shape}")

    # Check for missing overall bands
    df = df.dropna(subset=["Overall"])
    print(f"Shape after dropping missing 'Overall': {df.shape}")
    
    # Ensure band is float
    df['Overall'] = df['Overall'].astype(float)

    # 1. Check duplicate essays
    # A single prompt with multiple identical essays is a duplicate.
    # We can check exact matches on 'Essay' column.
    
    # Normalize essays slightly for duplicate detection (strip whitespace, lower)
    df['normalized_essay'] = df['Essay'].astype(str).str.strip().str.lower()
    
    duplicates = df.duplicated(subset=['normalized_essay'], keep=False)
    num_duplicates = duplicates.sum()
    print(f"\n[LEAKAGE CHECK] Found {num_duplicates} rows that share the same essay text.")
    
    if num_duplicates > 0:
        print("Example duplicates:")
        # Show first few duplicates
        dup_df = df[duplicates].sort_values(by="normalized_essay")
        for idx, row in dup_df.head(4).iterrows():
            print(f"  ID: {idx} | Band: {row['Overall']} | Essay[:50]: {row['Essay'][:50]}...")
            
    # Drop duplicates for the actual dataset we'll split
    df_dedup = df.drop_duplicates(subset=['normalized_essay'], keep='first').copy()
    print(f"\nShape after deduplication: {df_dedup.shape}")
    
    # 2. Check band distribution
    print("\n[BAND DISTRIBUTION]")
    dist = df_dedup['Overall'].value_counts().sort_index()
    for band, count in dist.items():
        print(f"  Band {band}: {count} samples")
        
    # Warn for rare bands (less than 3, which causes issues for 70/15/15 stratify)
    rare_bands = dist[dist < 3]
    if not rare_bands.empty:
        print("\n[WARNING] The following bands have fewer than 3 samples. Stratified split will fail for them:")
        print(rare_bands)
    
if __name__ == "__main__":
    main()
