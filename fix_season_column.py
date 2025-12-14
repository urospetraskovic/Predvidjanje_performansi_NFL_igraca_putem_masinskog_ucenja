"""
Quick fix: Post-process scraped CSV files to correct Season column
"""

import pandas as pd
import os

def fix_season_in_csv(filename, player_name, birth_year):
    """
    Fix Season column in a scraped CSV
    
    Args:
        filename: Path to CSV file
        player_name: Player name (for reference)
        birth_year: Birth year of player
    """
    
    print(f"\nProcessing: {player_name}")
    print(f"  File: {filename}")
    
    try:
        df = pd.read_csv(filename)
        
        print(f"  Columns: {list(df.columns)[:8]}...")
        print(f"  Rows: {len(df)}")
        
        # The first column (currently labeled "Season") actually contains Age
        if 'Season' in df.columns:
            # Show before
            print(f"  Before: Season={df['Season'].iloc[0]} (should be year)")
            
            # Convert Season to Age
            df['Age'] = pd.to_numeric(df['Season'], errors='coerce')
            
            # Calculate Season from birth year + age
            df['Season'] = birth_year + df['Age']
            
            # Reorder columns
            cols = df.columns.tolist()
            cols.remove('Season')
            cols.remove('Age')
            df = df[['Season', 'Age'] + cols]
            
            # Show after
            first_age = df['Age'].iloc[0]
            first_season = df['Season'].iloc[0]
            print(f"  After: Season={int(first_season)}, Age={int(first_age)}")
            
            # Save back
            df.to_csv(filename, index=False)
            print(f"  ✓ Fixed and saved")
            
            return df
        else:
            print(f"  ✗ Season column not found")
            return None
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


# Players and their birth years
PLAYERS_TO_FIX = [
    ('data/raw/BradTo00_stats.csv', 'Tom Brady', 1977),
    ('data/raw/AlleJo02_stats.csv', 'Josh Allen', 1996),
    ('data/raw/JackLa00_stats.csv', 'Lamar Jackson', 1997),
    ('data/raw/HurtJa00_stats.csv', 'Jalen Hurts', 1998),
    ('data/raw/MontJo01_stats.csv', 'Joe Montana', 1956),
]


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FIXING SEASON COLUMN IN SCRAPED CSVs")
    print("="*70)
    
    fixed_count = 0
    
    for filepath, player_name, birth_year in PLAYERS_TO_FIX:
        if os.path.exists(filepath):
            result = fix_season_in_csv(filepath, player_name, birth_year)
            if result is not None:
                fixed_count += 1
        else:
            print(f"\n✗ File not found: {filepath}")
    
    print(f"\n" + "="*70)
    print(f"✓ Fixed {fixed_count}/{len(PLAYERS_TO_FIX)} files")
    print("="*70)
    
    # Show sample of fixed data
    print("\n\nSample of fixed data (Josh Allen):")
    if os.path.exists('data/raw/AlleJo02_stats.csv'):
        df = pd.read_csv('data/raw/AlleJo02_stats.csv')
        print(df[['Season', 'Age', 'Team', 'Pos', 'G', 'Cmp', 'Yds', 'TD']].head(10))
