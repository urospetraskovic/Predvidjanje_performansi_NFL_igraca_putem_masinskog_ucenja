import pandas as pd

df = pd.read_csv('data/processed/all_passing.csv')
season_2025 = df[df['Season'] == 2025]

print(f'Total 2025 records: {len(season_2025)}')
print(f'\n2025 Games played per QB:')
print(season_2025[['Player', 'G']].drop_duplicates().sort_values('G', ascending=False))
print(f'\nMin games: {season_2025["G"].min()}, Max games: {season_2025["G"].max()}')
print(f'\nSeason 2025 has {len(season_2025)} QBs with data')
