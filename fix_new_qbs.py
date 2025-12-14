import pandas as pd

players_to_fix = [
    ('data/raw/qb/BurrJo01_stats.csv', 'Joe Burrow', 1996),
    ('data/raw/qb/HerbJu00_stats.csv', 'Justin Herbert', 1998),
]

for filepath, player_name, birth_year in players_to_fix:
    print(f'Fixing {player_name}...')
    df = pd.read_csv(filepath)
    
    # Convert Season column (contains Age) to Age, calculate Season
    df['Age'] = pd.to_numeric(df['Season'], errors='coerce')
    df['Season'] = birth_year + df['Age']
    
    # Reorder columns: Season, Age first
    cols = df.columns.tolist()
    cols.remove('Season')
    cols.remove('Age')
    df = df[['Season', 'Age'] + cols]
    
    # Remove summary rows if any exist
    df = df[df['Season'].notna()]
    
    # Save
    df.to_csv(filepath, index=False)
    min_season = int(df['Season'].min())
    max_season = int(df['Season'].max())
    num_seasons = len(df)
    print(f'  ✓ {player_name}: {num_seasons} seasons ({min_season}-{max_season})')

print('\nAll players fixed!')
