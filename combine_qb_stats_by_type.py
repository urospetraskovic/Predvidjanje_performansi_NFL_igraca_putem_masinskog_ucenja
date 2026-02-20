#!/usr/bin/env python3
"""
Combine QB stats by stat type (passing, advancing passing, rushing, etc.)
Creates separate CSV files for each stat category across all QBs.
"""

import pandas as pd
import os
from pathlib import Path

# Path to QB raw data
qb_raw_path = Path("data/raw/qb")

# Stat types to combine
STAT_TYPES = {
    'passing.csv': 'all_passing.csv',
    'advanced_passing.csv': 'all_advanced_passing.csv',
    'adjusted_passing.csv': 'all_adjusted_passing.csv',
    'rushing_receiving.csv': 'all_rushing_receiving.csv',
    'advanced_rushing_receiving.csv': 'all_advanced_rushing_receiving.csv',
    'defense_fumbles.csv': 'all_defense_fumbles.csv',
    'snap_counts.csv': 'all_snap_counts.csv'
}

# Get all QB folders
qb_folders = sorted([f for f in os.listdir(qb_raw_path) if os.path.isdir(qb_raw_path / f)])

print(f"Found {len(qb_folders)} QBs")
print(f"Combining {len(STAT_TYPES)} stat types...\n")

# For each stat type, combine across all QBs
for source_file, output_file in STAT_TYPES.items():
    print(f"Processing {source_file}...")
    combined_data = []
    
    for qb_name in qb_folders:
        qb_path = qb_raw_path / qb_name
        file_path = qb_path / source_file
        
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                # Add player name if not already present
                if 'Player' not in df.columns:
                    df['Player'] = qb_name.replace('_', ' ')
                combined_data.append(df)
            except Exception as e:
                print(f"  ✗ Error reading {qb_name}/{source_file}: {e}")
        # else:
        #     print(f"  - {qb_name}: {source_file} not found")
    
    # Combine all data
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Save combined file
        output_path = Path("data/processed") / output_file
        combined_df.to_csv(output_path, index=False)
        print(f"  ✓ {output_file}: {len(combined_df)} rows × {len(combined_df.columns)} columns")
        print(f"    Saved to: {output_path}")
    else:
        print(f"  ✗ No data found for {source_file}")

print("\n" + "="*70)
print("✓ All QB stats by type have been combined!")
print("="*70)
print("\nFiles created in data/processed/:")
for source_file, output_file in STAT_TYPES.items():
    output_path = Path("data/processed") / output_file
    if output_path.exists():
        df = pd.read_csv(output_path)
        print(f"  ✓ {output_file} ({len(df)} rows)")
