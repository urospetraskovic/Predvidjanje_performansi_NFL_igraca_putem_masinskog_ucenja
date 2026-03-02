import pandas as pd
import os
import sys
import warnings

warnings.filterwarnings('ignore')

RAW_DIR = 'data/raw/te'
PROCESSED_DIR = os.path.join('data', 'processed')
OUTPUT_FILE = os.path.join('data', 'fully combined', 'te_master.csv')

KEY_COLS = ['Season', 'Player', 'PlayerID']
SHARED_COLS = ['Age', 'Team', 'Lg', 'Pos', 'G', 'GS']


def load_table(folder_path, filename):
    filepath = os.path.join(folder_path, filename)
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None
        if 'Season' in df.columns:
            df['Season'] = pd.to_numeric(df['Season'], errors='coerce')
            df = df.dropna(subset=['Season'])
            df['Season'] = df['Season'].astype(int)
        # drop Lg column everywhere - not needed
        if 'Lg' in df.columns:
            df = df.drop('Lg', axis=1)
        return df
    except Exception as e:
        print(f"  Warning: Error reading {filepath}: {e}")
        return None


def prefix_columns(df, prefix, keep_cols):
    rename_map = {}
    for col in df.columns:
        if col not in keep_cols:
            rename_map[col] = f"{prefix}_{col}"
    return df.rename(columns=rename_map)


def process_receiving_rushing(df):
    if df is None:
        return None

    renames = {
        'Yds': 'Rec_Yds',
        'Y/R': 'Rec_Y/R',
        'TD': 'Rec_TD',
        '1D': 'Rec_1D',
        'rec_success': 'Rec_Succ%',
        'rec_long': 'Rec_Lng',
        'R/G': 'Rec/G',
        'Rec_R/G': 'Rec/G',         
        'Y/G': 'Rec_Y/G',
        'catch_pct': 'Catch%',
        'Y/Tgt': 'Rec_Y/Tgt',
        'Att': 'Rush_Att',
        'Yds.1': 'Rush_Yds',
        'TD.1': 'Rush_TD',
        '1D.1': 'Rush_1D',
        'Succ%': 'Rush_Succ%',
        'Lng': 'Rush_Lng',
        'Y/A': 'Rush_Y/A',
        'Y/G.1': 'Rush_Y/G',
        'A/G': 'Rush_A/G',
        'Touch': 'Touches',
        'yds_per_touch': 'Y/Touch',
        'yds_from_scrimmage': 'Scrimmage_Yds',
        'rush_receive_td': 'Rush_Rec_TD',
        'av': 'AV',
        'awards': 'Awards',
        'fumbles': 'Fmb',
        'Fmb': 'Fmb',   
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    df = df.drop(columns=['Lg'], errors='ignore')

    return df


def process_advanced_receiving_rushing(df):
    if df is None:
        return None

    renames = {
        'Tgt': '_drop_Tgt',
        'Rec': '_drop_Rec',
        'Yds': '_drop_Yds',
        '1D': '_drop_1D',
        'YBC': 'Rec_YBC',
        'YBC/R': 'Rec_YBC/R',
        'YAC': 'Rec_YAC',
        'YAC/R': 'Rec_YAC/R',
        'ADOT': 'Rec_aDOT',
        'BrkTkl': 'Rec_BrkTkl',
        'Rec/Br': 'Rec_Att/Br',
        'Drop': 'Rec_Drops',
        'Drop%': 'Rec_Drop%',
        'Int': 'Rec_Int',
        'rec_pass_rating': 'Rec_PassRtg',
        'Rec_AirYds': 'Rec_AirYds',
        'Rec_AirYds/R': 'Rec_AirYds/R',
        'Rec_BrkTkl/R': 'Rec_BrkTkl/R',
        'Att': '_drop_Att',
        'Yds.1': '_drop_Yds1',
        '1D.1': '_drop_1D1',
        'YBC.1': 'Rush_YBC',
        'YBC/Att': 'Rush_YBC/A',
        'Rush_YBC/A': 'Rush_YBC/A',  
        'YAC.1': 'Rush_YAC',
        'YAC/Att': 'Rush_YAC/A',
        'Rush_YAC/A': 'Rush_YAC/A',  
        'BrkTkl.1': 'Rush_BrkTkl',
        'Att/Br': 'Rush_Att/Br',
        'awards': '_drop_awards',
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    drop_cols = [c for c in df.columns if c.startswith('_drop_')]
    drop_cols += [c for c in SHARED_COLS if c in df.columns]
    drop_cols += ['Awards', 'Lg']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'adv', keep)
    return df


def process_snap_counts(df):
    if df is None:
        return None

    renames = {
        'offense': 'Off_Snaps',
        'defense': 'Def_Snaps',
        'special_teams': 'ST_Snaps',
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    drop_cols = ['No.', 'Lg'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'snp', keep)
    return df


def add_team_change_and_prev_yards(df):
    df = df.sort_values(['Player', 'Season']).reset_index(drop=True)

    prev_team = df.groupby('Player')['Team'].shift(1)
    df['Team_Changed'] = ((df['Team'] != prev_team) & prev_team.notna()).astype(int)

    prev_yards = df.groupby('Player')['Rec_Yds'].shift(1)
    df['Prev_Season_Yds'] = prev_yards.fillna(0)

    return df


def combine_intermediate():
    print(f"\nWriting intermediate all_te_*.csv files to {PROCESSED_DIR}/...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    folders = sorted([f for f in os.listdir(RAW_DIR)
                      if os.path.isdir(os.path.join(RAW_DIR, f))])

    file_types = {}
    for folder in folders:
        folder_path = os.path.join(RAW_DIR, folder)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        for csv_file in csv_files:
            if csv_file not in file_types:
                file_types[csv_file] = []
            filepath = os.path.join(folder_path, csv_file)
            try:
                df = pd.read_csv(filepath)
                file_types[csv_file].append(df)
            except Exception as e:
                print(f"  Warning: Error reading {filepath}: {e}")

    for filename, dfs in file_types.items():
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            output_file = os.path.join(PROCESSED_DIR, f'all_te_{filename}')
            combined.to_csv(output_file, index=False)
            print(f"  [OK] {filename}: {len(dfs)} players, {len(combined)} rows -> {output_file}")



def build_master():
    print("=" * 70)
    print("BUILDING TE MASTER TABLE")
    print("=" * 70)

    combine_intermediate()

    folders = sorted([f for f in os.listdir(RAW_DIR)
                      if os.path.isdir(os.path.join(RAW_DIR, f))])
    print(f"\nFound {len(folders)} TE folders in {RAW_DIR}")

    table_configs = [
        ('receiving_rushing.csv', process_receiving_rushing),
        ('advanced_receiving_rushing.csv', process_advanced_receiving_rushing),
        ('snap_counts.csv', process_snap_counts),
    ]

    all_player_dfs = []
    table_counts = {t[0]: 0 for t in table_configs}

    for folder in folders:
        folder_path = os.path.join(RAW_DIR, folder)

        base_df = load_table(folder_path, 'receiving_rushing.csv')
        if base_df is None:
            print(f"  Skipping {folder} - no receiving_rushing")
            continue

        base_df = process_receiving_rushing(base_df)
        table_counts['receiving_rushing.csv'] += 1

        for filename, processor in table_configs[1:]:
            table_df = load_table(folder_path, filename)
            processed = processor(table_df)
            if processed is not None:
                table_counts[filename] += 1
                base_df = base_df.merge( # type: ignore
                    processed,
                    on=KEY_COLS,
                    how='left'
                )

        all_player_dfs.append(base_df)

    if not all_player_dfs:
        print("ERROR: No player data found!")
        sys.exit(1)


    master = pd.concat(all_player_dfs, ignore_index=True)
    print(f"\nCombined data: {len(master)} rows, {master['Player'].nunique()} TEs, {len(master.columns)} columns")

    print("\nTable coverage:")
    for filename, count in table_counts.items():
        label = filename.replace('.csv', '')
        pct = count / len(folders) * 100
        print(f"  {label:35s} {count:3d}/{len(folders)} TEs ({pct:.0f}%)")

    # reorder so identity comes first
    identity = ['Player', 'PlayerID', 'Season', 'Age', 'Team', 'Pos', 'G', 'GS']
    identity = [c for c in identity if c in master.columns]

    # Drop Lg if it somehow survived
    if 'Lg' in master.columns:
        master = master.drop('Lg', axis=1)

    
    rec_rush = [c for c in master.columns
                if c not in identity and not c.startswith(('adv_', 'snp_'))]
    
    adv = sorted([c for c in master.columns if c.startswith('adv_')])
    snaps = sorted([c for c in master.columns if c.startswith('snp_')])

    ordered = identity + rec_rush + adv + snaps

    missed = [c for c in master.columns if c not in ordered]
    if missed:
        print(f"\n  Note: {len(missed)} uncategorized columns appended: {missed}")
    ordered += missed

    master = master[[c for c in ordered if c in master.columns]]

    master = master.sort_values(['Player', 'Season']).reset_index(drop=True)

    # Add team change and previous season receiving yards columns
    master = add_team_change_and_prev_yards(master)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    master.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 70}")
    print(f"MASTER TABLE SAVED: {OUTPUT_FILE}")
    print(f"  Rows:    {len(master)} (TE-seasons)")
    print(f"  Columns: {len(master.columns)}")
    print(f"  TEs:     {master['Player'].nunique()}")
    print(f"  Seasons: {master['Season'].min()} - {master['Season'].max()}")
    print(f"  Size:    {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")
    print(f"{'=' * 70}")

    print(f"\nColumn groups:")
    print(f"  Identity:               {len(identity)}")
    print(f"  Receiving/Rushing:      {len(rec_rush)}")
    print(f"  Adv Rec/Rush (2018+):   {len(adv)}")
    print(f"  Snap Counts:            {len(snaps)}")
    print(f"  Total:                  {len(master.columns)}")

    return master


if __name__ == '__main__':
    build_master()
