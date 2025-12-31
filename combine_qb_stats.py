import pandas as pd
import os
from pathlib import Path

# Putanja do QB raw podataka
qb_raw_path = Path("data/raw/qb")

# Dobij sve QB foldera
qb_folders = sorted([f for f in os.listdir(qb_raw_path) if os.path.isdir(qb_raw_path / f)])

print(f"Pronađeno {len(qb_folders)} QB-a: {', '.join(qb_folders)}")

# Dictionary za čuvanje svih podataka
all_data = []

# Iteriraj kroz svakog QB-a
for qb_name in qb_folders:
    qb_path = qb_raw_path / qb_name
    
    # Pronađi sve CSV fajlove za tog QB-a
    csv_files = sorted([f for f in os.listdir(qb_path) if f.endswith('.csv')])
    
    print(f"\n{qb_name}:")
    
    # Prvo učitaj osnovni passing.csv kao polaznu tačku
    dfs = {}
    for csv_file in csv_files:
        try:
            file_path = qb_path / csv_file
            df = pd.read_csv(file_path)
            dfs[csv_file.replace('.csv', '')] = df
            print(f"  ✓ {csv_file}: {df.shape[0]} redova, {df.shape[1]} kolona")
        except Exception as e:
            print(f"  ✗ {csv_file}: Greška - {e}")
    
    # Ako postoji passing.csv, počni sa njim
    if 'passing' in dfs:
        merged_df = dfs['passing'].copy()
        merged_df['Player'] = qb_name.replace('_', ' ')
        
        # Dodaj ostale statistike
        for stat_type in ['advanced_passing', 'rushing_receiving', 'advanced_rushing_receiving', 
                          'snap_counts', 'defense_fumbles', 'adjusted_passing']:
            if stat_type in dfs:
                # Odredi key kolone za merge (obično Year/Season i Age)
                merge_keys = []
                if 'Year' in merged_df.columns and 'Year' in dfs[stat_type].columns:
                    merge_keys = ['Year']
                elif 'Season' in merged_df.columns and 'Season' in dfs[stat_type].columns:
                    merge_keys = ['Season']
                
                if merge_keys:
                    # Ukloni duplikate pre merge-a
                    stat_df = dfs[stat_type].copy()
                    stat_df_cols = [col for col in stat_df.columns if col not in merged_df.columns or col in merge_keys]
                    stat_df = stat_df[stat_df_cols]
                    
                    merged_df = merged_df.merge(stat_df, on=merge_keys, how='left', suffixes=('', f'_{stat_type}'))
        
        all_data.append(merged_df)

# Kombinuj sve QB podatke
print("\n" + "="*60)
print("Kombinujem sve QB podatke...")
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✓ Kombinovano: {combined_df.shape[0]} redova, {combined_df.shape[1]} kolona")
    
    # Sortuj po QB imena i godini
    if 'Player' in combined_df.columns:
        if 'Year' in combined_df.columns:
            combined_df = combined_df.sort_values(['Player', 'Year']).reset_index(drop=True)
        elif 'Season' in combined_df.columns:
            combined_df = combined_df.sort_values(['Player', 'Season']).reset_index(drop=True)
    
    # Sačuvaj u processed foldera
    output_path = Path("data/processed/all_qb_combined_stats.csv")
    combined_df.to_csv(output_path, index=False)
    print(f"✓ Sačuvano: {output_path}")
    print(f"\nFinalne dimenzije: {combined_df.shape[0]} redova × {combined_df.shape[1]} kolona")
    
    # Prikaži primere
    print(f"\nPrimeri kolona: {', '.join(combined_df.columns[:15].tolist())}")
    print("\nPrimeri QB-a iz baze:")
    print(combined_df[['Player', 'Year' if 'Year' in combined_df.columns else 'Season', 'Team', 'Cmp', 'Yds', 'TD']].head(10))
else:
    print("✗ Nema podataka za kombinovanje!")
