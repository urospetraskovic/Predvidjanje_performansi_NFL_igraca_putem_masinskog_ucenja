import pandas as pd
import os
import sys
import warnings

warnings.filterwarnings('ignore')

RAW_DIR = 'data/raw/qb'
ELO_DIR = os.path.join('data', 'nfl elo data')
PROCESSED_DIR = os.path.join('data', 'processed', 'all_qb')
OUTPUT_FILE = os.path.join('data', 'fully combined', 'qb_master.csv')

# we use column prefixes to avoid collisions during merge 
# each table's non-key columns get a prefix so we know where they came from.

# columns that are shared "key" or identity columns (used to join, kept once)
KEY_COLS = ['Season', 'Player', 'PlayerID']
SHARED_COLS = ['Age', 'Team', 'Pos', 'G', 'GS']  


def load_table(folder_path, filename):
    filepath = os.path.join(folder_path, filename)
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None
        # ensure Season is numeric
        if 'Season' in df.columns:
            df['Season'] = pd.to_numeric(df['Season'], errors='coerce')
            df = df.dropna(subset=['Season'])
            df['Season'] = df['Season'].astype(int)
        return df
    except Exception as e:
        print(f"  Warning: Error reading {filepath}: {e}")
        return None


def prefix_columns(df, prefix, keep_cols):
    """Add prefix to all columns except those in keep_cols."""
    rename_map = {}
    for col in df.columns:
        if col not in keep_cols:
            rename_map[col] = f"{prefix}_{col}"
    return df.rename(columns=rename_map)


def process_passing(df):
    if df is None:
        return None
    return df


def process_adjusted_passing(df):
    if df is None:
        return None
    drop_cols = ['Awards'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'adj', keep)
    return df


def process_advanced_passing(df):
    if df is None:
        return None
    # drop duplicated basic columns and Awards
    drop_cols = ['Awards', 'Cmp', 'Att'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'adv_pass', keep)
    return df


def process_rushing_receiving(df):
    if df is None:
        return None
    drop_cols = ['Awards'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'rr', keep)
    return df


def process_advanced_rushing_receiving(df):
    if df is None:
        return None
    # drop duplicated rushing columns that are already in rushing_receiving
    dup_cols = ['Rush_Att', 'Rush_Yds', 'Rush_1D', 'Tgt', 'Rec', 'Rec_Yds', 'Rec_1D',
                'Awards'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in dup_cols if c in df.columns], errors='ignore')
    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'adv_rr', keep)
    return df


def process_defense_fumbles(df):
    if df is None:
        return None
    drop_cols = ['Awards'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'def', keep)
    return df


def process_snap_counts(df):
    if df is None:
        return None
    # drop No. (uniform number) and shared cols
    drop_cols = ['No.'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'snp', keep)
    return df


def load_rankings():
    filepath = os.path.join(ELO_DIR, 'qb_rankings_career.csv')
    if not os.path.exists(filepath):
        print("  Rankings file not found, skipping rankings merge.")
        return None

    df = pd.read_csv(filepath)
    print(f"  Rankings data loaded: {len(df)} rows, {df['QB'].nunique()} QBs")

    # rename QB -> Player for merge key
    df = df.rename(columns={'QB': 'Player'})

    # prefix all stat columns to avoid collisions
    keep = ['Player', 'Season']
    df = prefix_columns(df, 'elo', keep)

    return df


def combine_intermediate():
    print(f"\nWriting intermediate all_qb_*.csv files to {PROCESSED_DIR}/...")
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
            output_file = os.path.join(PROCESSED_DIR, f'all_qb_{filename}')
            combined.to_csv(output_file, index=False)
            print(f"  [OK] {filename}: {len(dfs)} players, {len(combined)} rows -> {output_file}")


def build_master():
    """Build the master QB CSV from all raw data + rankings."""

    print("=" * 70)
    print("BUILDING QB MASTER TABLE")
    print("=" * 70)

    combine_intermediate()

    # load all QB folders 
    folders = sorted([f for f in os.listdir(RAW_DIR)
                      if os.path.isdir(os.path.join(RAW_DIR, f))])
    print(f"\nFound {len(folders)} QB folders in {RAW_DIR}")

    # table configs: (filename, processor_function)
    table_configs = [
        ('passing.csv', process_passing),
        ('adjusted_passing.csv', process_adjusted_passing),
        ('advanced_passing.csv', process_advanced_passing),
        ('rushing_receiving.csv', process_rushing_receiving),
        ('advanced_rushing_receiving.csv', process_advanced_rushing_receiving),
        ('defense_fumbles.csv', process_defense_fumbles),
        ('snap_counts.csv', process_snap_counts),
    ]

    # process each QB folder
    all_player_dfs = []
    table_counts = {t[0]: 0 for t in table_configs}

    for folder in folders:
        folder_path = os.path.join(RAW_DIR, folder)

        # load passing.csv 
        base_df = load_table(folder_path, 'passing.csv')
        if base_df is None:
            print(f"  Skipping {folder} - no passing.csv")
            continue

        base_df = process_passing(base_df)
        table_counts['passing.csv'] += 1

        # merge other tables into the base using Season + Player + PlayerID
        for filename, processor in table_configs[1:]:  # skip passing.csv since it's already loaded
            table_df = load_table(folder_path, filename)
            processed = processor(table_df)
            if processed is not None:
                table_counts[filename] += 1
                base_df = base_df.merge(
                    processed,
                    on=KEY_COLS,
                    how='left'
                )

        all_player_dfs.append(base_df)

    if not all_player_dfs:
        print("ERROR: No player data found!")
        sys.exit(1)

    # concatenate all players 
    master = pd.concat(all_player_dfs, ignore_index=True)
    print(f"\nCombined PFR data: {len(master)} rows, {master['Player'].nunique()} QBs, {len(master.columns)} columns")

    print("\nTable coverage:")
    for filename, count in table_counts.items():
        label = filename.replace('.csv', '')
        pct = count / len(folders) * 100
        print(f"  {label:35s} {count:3d}/{len(folders)} QBs ({pct:.0f}%)")

    # merge rankings data
    print("\n--- Merging Rankings/Elo Data ---")
    rankings = load_rankings()
    if rankings is not None:
        # check name matching before merge
        pfr_names = set(master['Player'].unique())
        elo_names = set(rankings['Player'].unique())
        matched = pfr_names & elo_names
        unmatched_pfr = pfr_names - elo_names
        print(f"  Name matching: {len(matched)}/{len(pfr_names)} PFR QBs found in rankings")
        if unmatched_pfr:
            print(f"  Not in rankings: {sorted(unmatched_pfr)[:10]}{'...' if len(unmatched_pfr) > 10 else ''}")

        master = master.merge(
            rankings,
            on=['Player', 'Season'],
            how='left'
        )
        # count how many rows got rankings data
        elo_cols = [c for c in master.columns if c.startswith('elo_')]
        if elo_cols:
            has_elo = master[elo_cols[0]].notna().sum()
            print(f"  Rankings data matched: {has_elo}/{len(master)} season-rows ({has_elo/len(master)*100:.1f}%)")

    # reorder columns: identity first, then groups by prefix
    identity = ['Player', 'PlayerID', 'Season', 'Age', 'Team', 'Pos', 'G', 'GS']
    passing = [c for c in master.columns if c not in identity and not c.startswith(('adj_', 'adv_', 'rr_', 'def_', 'snp_', 'elo_'))]
    adj = sorted([c for c in master.columns if c.startswith('adj_')])
    adv_pass = sorted([c for c in master.columns if c.startswith('adv_pass_')])
    rr = sorted([c for c in master.columns if c.startswith('rr_')])
    adv_rr = sorted([c for c in master.columns if c.startswith('adv_rr_')])
    defense = sorted([c for c in master.columns if c.startswith('def_')])
    snaps = sorted([c for c in master.columns if c.startswith('snp_')])
    elo = sorted([c for c in master.columns if c.startswith('elo_')])

    ordered = identity + passing + adj + adv_pass + rr + adv_rr + defense + snaps + elo
    # find missing columns 
    missed = [c for c in master.columns if c not in ordered]
    if missed:
        print(f"\n  Note: {len(missed)} uncategorized columns appended: {missed}")
    ordered += missed

    master = master[[c for c in ordered if c in master.columns]]

    # sort and save
    master = master.sort_values(['Player', 'Season']).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    master.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 70}")
    print(f"MASTER TABLE SAVED: {OUTPUT_FILE}")
    print(f"  Rows:    {len(master)} (QB-seasons)")
    print(f"  Columns: {len(master.columns)}")
    print(f"  QBs:     {master['Player'].nunique()}")
    print(f"  Seasons: {master['Season'].min()} - {master['Season'].max()}")
    print(f"  Size:    {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")
    print(f"{'=' * 70}")

    print(f"\nColumn groups:")
    print(f"  Identity:               {len(identity)}")
    print(f"  Passing (core):         {len(passing)}")
    print(f"  Adjusted Passing:       {len(adj)}")
    print(f"  Advanced Passing:       {len(adv_pass)}")
    print(f"  Rushing/Receiving:      {len(rr)}")
    print(f"  Adv Rush/Rec:           {len(adv_rr)}")
    print(f"  Defense/Fumbles:        {len(defense)}")
    print(f"  Snap Counts:            {len(snaps)}")
    print(f"  Rankings/Elo:           {len(elo)}")
    print(f"  Total:                  {len(master.columns)}")

    return master


if __name__ == '__main__':
    build_master()
