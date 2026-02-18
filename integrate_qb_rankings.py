#!/usr/bin/env python3
"""
Integrate QB Rankings files (career + 2025) with existing QB stats.
Creates enriched QB datasets with historical and advanced metrics.
"""

import pandas as pd
import os
from pathlib import Path

# Directories
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RANKINGS_DIR = os.path.join(DATA_DIR, "rankings")

# Create rankings directory if it doesn't exist
os.makedirs(RANKINGS_DIR, exist_ok=True)

# Source files
CAREER_RANKINGS = "c:\\Users\\Korisnik\\Downloads\\QB Rankings career.csv"
RANKINGS_2025 = "c:\\Users\\Korisnik\\Downloads\\QB Rankings 2025.csv"

def copy_rankings_files():
    """Copy ranking files to project directory."""
    print("Copying ranking files...")
    
    if os.path.exists(CAREER_RANKINGS):
        dest = os.path.join(RANKINGS_DIR, "qb_rankings_career.csv")
        df = pd.read_csv(CAREER_RANKINGS)
        df.to_csv(dest, index=False)
        print(f"  ✓ Career rankings: {len(df)} rows -> {dest}")
    else:
        print(f"  ✗ Career rankings not found")
    
    if os.path.exists(RANKINGS_2025):
        dest = os.path.join(RANKINGS_DIR, "qb_rankings_2025.csv")
        df = pd.read_csv(RANKINGS_2025)
        df.to_csv(dest, index=False)
        print(f"  ✓ 2025 rankings: {len(df)} rows -> {dest}")
    else:
        print(f"  ✗ 2025 rankings not found")


def load_existing_qb_stats():
    """Load existing QB stats from processed folder."""
    print("\nLoading existing QB stats...")
    
    passing_file = os.path.join(PROCESSED_DIR, "all_passing.csv")
    
    if os.path.exists(passing_file):
        df = pd.read_csv(passing_file)
        print(f"  ✓ Loaded {passing_file}: {len(df)} rows, {df['Player'].nunique()} unique players")
        return df
    else:
        print(f"  ✗ File not found: {passing_file}")
        return None


def integrate_2025_rankings():
    """Integrate 2025 rankings with existing 2025 QB stats."""
    print("\nIntegrating 2025 rankings with existing stats...")
    
    # Load existing 2025 QB stats
    existing_stats = load_existing_qb_stats()
    if existing_stats is None:
        return
    
    # Filter for 2025 only
    existing_2025 = existing_stats[existing_stats['Season'] == 2025].copy()
    print(f"  Found {len(existing_2025)} existing 2025 QB records from {existing_2025['Player'].nunique()} players")
    
    # Load 2025 rankings
    rankings_file = os.path.join(RANKINGS_DIR, "qb_rankings_2025.csv")
    if not os.path.exists(rankings_file):
        print(f"  ✗ Rankings file not found: {rankings_file}")
        return
    
    rankings_2025 = pd.read_csv(rankings_file)
    print(f"  Loaded rankings 2025: {len(rankings_2025)} rows")
    
    # Merge on QB name and season
    # Normalize names for matching
    existing_2025['QB_normalized'] = existing_2025['Player'].str.lower().str.strip()
    rankings_2025['QB_normalized'] = rankings_2025['QB'].str.lower().str.strip()
    
    merged = existing_2025.merge(
        rankings_2025,
        left_on='QB_normalized',
        right_on='QB_normalized',
        how='left',
        suffixes=('_stats', '_rankings')
    )
    
    print(f"  Merged: {len(merged)} records with rankings data")
    print(f"  Match rate: {merged['Points'].notna().sum()} / {len(merged)} ({merged['Points'].notna().sum()/len(merged)*100:.1f}%)")
    
    # Save integrated file
    output_file = os.path.join(PROCESSED_DIR, "qb_stats_with_rankings_2025.csv")
    merged.to_csv(output_file, index=False)
    print(f"  ✓ Saved to: {output_file}")
    
    # Show sample columns available
    print(f"\n  Available ranking metrics:")
    ranking_cols = ['Points', 'QB Elo', 'QBR', 'WPA / DB', 'Total WPA', 'Change (Year)', 'Success']
    for col in ranking_cols:
        if col in merged.columns:
            print(f"    - {col}")


def create_historical_comparison():
    """Create comparison file with career rankings and 2025 stats."""
    print("\nCreating historical comparison dataset...")
    
    # Load career rankings
    career_file = os.path.join(RANKINGS_DIR, "qb_rankings_career.csv")
    if not os.path.exists(career_file):
        print(f"  ✗ Career rankings not found")
        return
    
    career_rankings = pd.read_csv(career_file)
    
    # Load 2025 rankings
    rankings_2025_file = os.path.join(RANKINGS_DIR, "qb_rankings_2025.csv")
    if not os.path.exists(rankings_2025_file):
        print(f"  ✗ 2025 rankings not found")
        return
    
    rankings_2025 = pd.read_csv(rankings_2025_file)
    
    # Filter career rankings to get only historical data
    # Get QBs in 2025, find their best season in career rankings
    qbs_2025 = set(rankings_2025['QB'].unique())
    career_filtered = career_rankings[career_rankings['QB'].isin(qbs_2025)].copy()
    
    # Summary stats per QB
    summary = career_filtered.groupby('QB').agg({
        'Season': 'count',
        'Points': ['mean', 'max'],
        'QBR': 'mean',
        'QB Elo': 'max',
        'W': 'sum',
        'L': 'sum'
    }).reset_index()
    
    summary.columns = ['QB', 'Seasons_in_Career_Ranking', 'Avg_Points', 'Max_Points', 
                       'Avg_QBR', 'Max_Elo', 'Career_Wins', 'Career_Losses']
    
    # Merge with 2025 rankings
    rankings_2025_summary = rankings_2025[['QB', 'Season', 'Points', 'QBR', 'QB Elo', 'W', 'L']].copy()
    rankings_2025_summary.rename(columns={
        'Points': 'Points_2025',
        'QBR': 'QBR_2025',
        'QB Elo': 'Elo_2025',
        'W': 'Wins_2025',
        'L': 'Losses_2025'
    }, inplace=True)
    
    comparison = summary.merge(rankings_2025_summary, on='QB', how='left')
    
    # Save comparison
    output_file = os.path.join(PROCESSED_DIR, "qb_career_vs_2025_comparison.csv")
    comparison.to_csv(output_file, index=False)
    print(f"  ✓ Created comparison: {len(comparison)} QBs -> {output_file}")
    
    # Print summary
    print("\n  Sample comparison (Top 5 QBs in career rankings):")
    top_qbs = comparison.nlargest(5, 'Max_Points')[['QB', 'Seasons_in_Career_Ranking', 'Avg_Points', 'Points_2025']]
    print(top_qbs.to_string(index=False))


def create_rankings_summary():
    """Create summary of all ranking files."""
    print("\nAvailable ranking files:")
    
    if os.path.exists(RANKINGS_DIR):
        files = os.listdir(RANKINGS_DIR)
        for f in sorted(files):
            fpath = os.path.join(RANKINGS_DIR, f)
            if f.endswith('.csv'):
                df = pd.read_csv(fpath)
                print(f"  • {f}: {len(df)} rows, {df.shape[1]} columns")
                if 'QB' in df.columns and 'Season' in df.columns:
                    print(f"    - QBs: {df['QB'].nunique()}, Seasons: {sorted(df['Season'].unique())}")


def main():
    print("=" * 70)
    print("QB RANKINGS INTEGRATION SCRIPT")
    print("=" * 70)
    
    # Copy files to project directory
    copy_rankings_files()
    
    # Integrate 2025 rankings with existing stats
    integrate_2025_rankings()
    
    # Create historical comparison
    create_historical_comparison()
    
    # Show summary
    create_rankings_summary()
    
    print("\n" + "=" * 70)
    print("INTEGRATION COMPLETE")
    print("=" * 70)
    print("\nNew files created in data/processed/:")
    print("  • qb_stats_with_rankings_2025.csv - 2025 QB stats merged with rankings")
    print("  • qb_career_vs_2025_comparison.csv - Career history vs 2025 performance")
    print("\nRanking files stored in data/rankings/:")
    print("  • qb_rankings_career.csv - Historical QB rankings")
    print("  • qb_rankings_2025.csv - 2025 season QB rankings")


if __name__ == '__main__':
    main()
