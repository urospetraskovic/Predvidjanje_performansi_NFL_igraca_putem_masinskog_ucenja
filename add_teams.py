#!/usr/bin/env python3
"""
Add Team abbreviations to QB CSV files in qb/ folder
"""

import pandas as pd
import os

# Team abbreviations for each QB  
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

print("Adding Team column to QB files...\n")

success = 0
failed = 0

for player_id, team in QB_TEAMS.items():
    qb_path = f'data/raw/qb/{player_id}_stats.csv'
    
    if not os.path.exists(qb_path):
        print(f"✗ {player_id}: File not found")
        failed += 1
        continue
    
    try:
        # Read file
        df = pd.read_csv(qb_path)
        
        # Fill Team column
        df['Team'] = df['Team'].fillna(team)
        df['Team'] = df['Team'].replace('', team)
        
        # Save
        df.to_csv(qb_path, index=False)
        print(f"✓ {player_id}: Added team '{team}'")
        success += 1
        
    except Exception as e:
        print(f"✗ {player_id}: {str(e)[:50]}")
        failed += 1

print(f"\n✓ Updated: {success}")
print(f"✗ Failed: {failed}")
