import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/all_qb_stats_combined.csv')

# SUPER CLEAN VERZIJA - samo najvažnije kolone za analizu
essential_cols = [
    'Player', 'Season', 'Age', 'Team', 'G', 'GS',
    'Cmp', 'Att', 'Cmp%', 'Yds', 'TD', 'Int', 'Rate',
    'Rush_Att', 'Rush_Yds', 'Rush_TD',
    'Rec', 'Rec_Yds', 'Rec_TD'
]

df_essential = df[essential_cols].copy()
df_essential = df_essential.sort_values(['Player', 'Season']).reset_index(drop=True)

df_essential.to_csv('data/processed/QB_stats_essential.csv', index=False)

print("✓ QB_stats_essential.csv - Super clean verzija sa 18 bitnih kolona")
print(f"  Redovi: {len(df_essential)}")
print(f"  Kolone: {len(df_essential.columns)}")

# Kreiraj summary po igraču
summary_list = []
for player in sorted(df['Player'].unique()):
    player_data = df[df['Player'] == player]
    summary_list.append({
        'Player': player,
        'Seasons': len(player_data),
        'Years': f"{int(player_data['Season'].min())}-{int(player_data['Season'].max())}",
        'Teams': ', '.join(player_data['Team'].unique()),
        'Games': int(player_data['G'].sum()),
        'Starts': int(player_data['GS'].sum()),
        'Total_Passes': int(player_data['Cmp'].sum()),
        'Completions': int(player_data['Cmp'].sum()),
        'Pass_Yds': int(player_data['Yds'].sum()),
        'Pass_TD': int(player_data['TD'].sum()),
        'Interceptions': int(player_data['Int'].sum()),
        'Rush_Att': int(player_data['Rush_Att'].sum()),
        'Rush_Yds': int(player_data['Rush_Yds'].sum()),
        'Rush_TD': int(player_data['Rush_TD'].sum()),
        'Avg_PassYds': round(player_data['Yds'].mean(), 1),
        'Avg_PassTD': round(player_data['TD'].mean(), 1),
        'Avg_Int': round(player_data['Int'].mean(), 1),
    })

df_summary = pd.DataFrame(summary_list)
df_summary = df_summary.sort_values('Pass_Yds', ascending=False).reset_index(drop=True)
df_summary.to_csv('data/processed/QB_summary_statistics.csv', index=False)

print("\n✓ QB_summary_statistics.csv - Sumarne statistike po QB-u")
print(f"  QB-a: {len(df_summary)}")
print("\nTop 10 po ukupnim passing yardsima:")
print(df_summary[['Player', 'Seasons', 'Teams', 'Pass_Yds', 'Pass_TD', 'Interceptions']].head(10).to_string(index=False))

# Kreiraj verziju sa svim detaljima za advanced analizu
df.to_csv('data/processed/QB_stats_all_details.csv', index=False)
print(f"\n✓ QB_stats_all_details.csv - Sve 142 kolone za advanced analizu")
print(f"  Redovi: {len(df)}")
print(f"  Kolone: {len(df.columns)}")

print("\n" + "="*80)
print("GOTOVO! QBs BAZA JE SPREMNA")
print("="*80)
print("\nDATOTEKE:")
print("1. all_qb_stats_combined.csv    - 218 redova x 52 kolone (MAIN)")
print("2. QB_stats_essential.csv       - 218 redova x 18 kolona (CLEAN)")
print("3. QB_summary_statistics.csv    - 30 QB-a sa summary statistikama")
print("4. QB_stats_all_details.csv     - 218 redova x 142 kolone (FULL)")
print("\nSVE DATOTEKE JE U: data/processed/")
