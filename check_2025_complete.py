import pandas as pd

df = pd.read_csv('data/processed/all_passing.csv')
season_2025 = df[df['Season'] == 2025]

print('2025 Games per QB (sorted by games played):')
print(season_2025[['Player', 'G']].drop_duplicates().sort_values('G', ascending=False))
print(f'\nTotal 2025 QBs: {len(season_2025["Player"].unique())}')
print(f'Max games: {season_2025["G"].max()}')
print(f'Min games: {season_2025["G"].min()}')

# Check Aaron Rodgers specifically
ar_2025 = season_2025[season_2025['Player'] == 'Aaron Rodgers']
if len(ar_2025) > 0:
    print(f'\nAaron Rodgers 2025:')
    print(f'  Games: {ar_2025["G"].values[0]}')
    print(f'  Team: {ar_2025["Team"].values[0]}')
    print(f'  Completions: {ar_2025["Cmp"].values[0]}')
