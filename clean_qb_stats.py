import pandas as pd
import numpy as np
from pathlib import Path

# Učitaj kombinovanu bazu
df = pd.read_csv('data/processed/all_qb_combined_stats.csv')

print("="*70)
print("QB BAZA - ČIŠĆENJE I ORGANIZACIJA")
print("="*70)

# Organizuj kolone logički
priority_cols = [
    'Player', 'Season', 'Age', 'Team', 'Pos',
    'G', 'GS', 'QBrec'
]

# PASSING STATS
passing_cols = [
    'Cmp', 'Att', 'Cmp%', 'Yds', 'TD', 'TD%', 'Int', 'Int%', 
    'Y/A', 'AY/A', 'Y/C', 'Rate', 'Sk', 'Yds_Lost', 'Sk%', 
    'NY/A', 'ANY/A', '4QC', 'GWD'
]

# ADVANCED PASSING
advanced_pass = [
    'pass_target_yds', 'pass_air_yds', 'pass_air_yds_per_cmp',
    'pass_air_yds_per_att', 'pass_yac', 'pass_yac_per_cmp',
    'pass_drops', 'pass_drop_pct', 'pass_poor_throws', 'pass_poor_throw_pct',
    'pass_pressured', 'pass_pressured_pct', 'pocket_time'
]

# RUSHING
rushing_cols = [
    'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rush_Y/A', 'Rush_Y/G', 'Rush_Lng'
]

# RECEIVING
receiving_cols = [
    'Tgt', 'Rec', 'Rec_Yds', 'Rec_TD', 'catch_pct', 'rec_yds_per_tgt'
]

# Filtriraj samo kolone koje postoje
available_cols = []
for col_list in [priority_cols, passing_cols, advanced_pass, rushing_cols, receiving_cols]:
    for col in col_list:
        if col in df.columns and col not in available_cols:
            available_cols.append(col)

# Kreiraj finalni dataframe
df_clean = df[available_cols].copy()

# Popuni NaN vrednosti
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)

# Sortiraj po player imena i godini
df_clean = df_clean.sort_values(['Player', 'Season']).reset_index(drop=True)

print(f"\n✓ Čišćeni podaci: {df_clean.shape[0]} redova × {df_clean.shape[1]} kolona")
print(f"\nKolone ({df_clean.shape[1]} total):")
for i, col in enumerate(df_clean.columns, 1):
    print(f"  {i:2d}. {col}")

# Sačuvaj clean verziju
output_path = Path("data/processed/all_qb_stats_combined.csv")
df_clean.to_csv(output_path, index=False)
print(f"\n✓ Sačuvano (clean): {output_path}")

# Sačuvaj i verziju sa svim mogućim podacima
output_all = Path("data/processed/all_qb_stats_full.csv")
df.to_csv(output_all, index=False)
print(f"✓ Sačuvano (full): {output_all}")

print("\n" + "="*70)
print("PREGLED PODATAKA")
print("="*70)

# Prikaži statistiku po QB-u
print("\nStatistika po QB-u (sezona po sezona):")
for player in sorted(df_clean['Player'].unique())[:5]:
    player_data = df_clean[df_clean['Player'] == player]
    print(f"\n{player}: {len(player_data)} sezona")
    print(f"  Seasons: {player_data['Season'].min():.0f} - {player_data['Season'].max():.0f}")
    print(f"  Teams: {', '.join(player_data['Team'].unique())}")
    avg_yards = player_data['Yds'].mean()
    avg_td = player_data['TD'].mean()
    print(f"  Prosečno: {avg_yards:.0f} yds/sez, {avg_td:.1f} TD/sez")

print("\n" + "="*70)
print(f"✓ GOTOVO! Svi QB podaci su kombinovani!")
print(f"✓ Glavna datoteka: all_qb_stats_combined.csv ({df_clean.shape[0]} redova)")
print("="*70)
