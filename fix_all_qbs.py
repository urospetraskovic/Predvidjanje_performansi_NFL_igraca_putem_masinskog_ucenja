#!/usr/bin/env python3
"""
Fix all QB CSV files - correct Season, Age, Team columns
"""

import pandas as pd
import os
from pathlib import Path

# QB birth years for Season calculation
QB_DATA = {
    'AlleJo02': 1996,
    'JackLa00': 1997,
    'BurrJo01': 1996,
    'HurtJa00': 1998,
    'HerbJu00': 1998,
    'RodgAa00': 1983,
    'MahoPa00': 1995,
    'PresDa01': 1992,
    'GoffJa00': 1992,
    'PurdBr00': 2001,
    'MayeDr00': 2004,
    'TagoTu00': 1998,
    'StroCJ00': 2003,
    'DarnSa00': 1997,
    'LawrTr00': 1999,
}

# Team abbreviations for each QB (2024-2025)
QB_TEAMS = {
    'AlleJo02': 'BUF',
    'JackLa00': 'BAL',
    'BurrJo01': 'CIN',
    'HurtJa00': 'PHI',
    'HerbJu00': 'LAC',
    'RodgAa00': 'NYJ',
    'MahoPa00': 'KC',
    'PresDa01': 'DAL',
    'GoffJa00': 'DET',
    'PurdBr00': 'SF',
    'MayeDr00': 'NE',
    'TagoTu00': 'MIA',
    'StroCJ00': 'HOU',
    'DarnSa00': 'TEN',
    'LawrTr00': 'JAX',
}

def fix_qb_file(player_id, birth_year):
    """Fix a single QB file"""
    # Check both raw/ and qb/ folders
    raw_path = f'data/raw/{player_id}_stats.csv'
    qb_path = f'data/raw/qb/{player_id}_stats.csv'
    
    input_file = None
    if os.path.exists(raw_path):
        input_file = raw_path
    elif os.path.exists(qb_path):
        input_file = qb_path
    else:
        print(f"✗ {player_id}: File not found")
        return False
    
    try:
        # Read input file
        df = pd.read_csv(input_file)
        
        # The issue: headers are offset from data
        # Expected headers: Season, Age, Team, Lg, Pos, G, GS, QBrec, Cmp, Att...
        # But data has: Age, Team, Lg, Pos, G, GS, QBrec, Cmp, Att... (missing Season at front)
        
        # Solution: The "Season" column actually contains Age values
        # "Age" column contains Team values
        # "Team" column contains Lg values
        # "Lg" column contains Pos values
        # etc.
        
        # Extract the misaligned data
        age_values = pd.to_numeric(df['Season'], errors='coerce')  # This is actually Age
        team_values = df['Age'].fillna('')  # This is actually Team
        
        # Calculate real Season from birth_year + age
        season_values = birth_year + age_values - 1
        
        # Get the rest of the columns (starting from 'Team')
        # which are actually Lg, Pos, G, GS, etc.
        df_data = df[['Team', 'Pos', 'G', 'GS', 'QBrec', 'Cmp', 'Att', 'Cmp%', 
                      'Yds', 'TD', 'TD%', 'Int', 'Int%', '1D', 'Succ%', 'Lng', 
                      'Y/A', 'AY/A', 'Y/C', 'Y/G', 'Rate', 'QBR', 'Sk', 'Yds.1', 
                      'Sk%', 'NY/A', 'ANY/A', '4QC', 'GWD', 'AV', 'Player', 'PlayerID']].copy()
        
        # Rename columns to match actual data
        df_data = df_data.rename(columns={
            'Team': 'Lg',
            'Pos': 'Pos_actual',
            'G': 'G_actual',
        })
        
        # Rebuild with correct columns
        df_fixed = pd.DataFrame({
            'Season': season_values.astype(int),
            'Age': age_values.astype(int),
            'Team': team_values,
        })
        
        # Add remaining columns
        remaining_cols = ['Lg', 'Pos', 'G', 'GS', 'QBrec', 'Cmp', 'Att', 'Cmp%', 
                         'Yds', 'TD', 'TD%', 'Int', 'Int%', '1D', 'Succ%', 'Lng', 
                         'Y/A', 'AY/A', 'Y/C', 'Y/G', 'Rate', 'QBR', 'Sk', 'Yds.1', 
                         'Sk%', 'NY/A', 'ANY/A', '4QC', 'GWD', 'AV', 'Player', 'PlayerID']
        
        for col in remaining_cols:
            if col in df.columns:
                df_fixed[col] = df[col]
        
        # Remove Lg column (all NFL)
        if 'Lg' in df_fixed.columns:
            df_fixed = df_fixed.drop('Lg', axis=1)
        
        # Fill Team with current team abbreviation if empty
        current_team = QB_TEAMS.get(player_id, '')
        df_fixed['Team'] = df_fixed['Team'].fillna(current_team)
        df_fixed['Team'] = df_fixed['Team'].replace('', current_team)
        
        # Remove rows with no data
        df_fixed = df_fixed[df_fixed['Season'].notna()]
        df_fixed = df_fixed[df_fixed['Season'] != '']
        
        # Save to qb folder
        os.makedirs('data/raw/qb', exist_ok=True)
        df_fixed.to_csv(qb_path, index=False)
        
        num_seasons = len(df_fixed)
        min_season = int(df_fixed['Season'].min())
        max_season = int(df_fixed['Season'].max())
        
        print(f"✓ {player_id}: {num_seasons} seasons ({min_season}-{max_season})")
        return True
        
    except Exception as e:
        print(f"✗ {player_id}: {str(e)[:60]}")
        return False

print("Fixing all QB files...\n")

success = 0
failed = 0

for player_id, birth_year in QB_DATA.items():
    if fix_qb_file(player_id, birth_year):
        success += 1
    else:
        failed += 1

print(f"\n✓ Fixed: {success}")
print(f"✗ Failed: {failed}")
