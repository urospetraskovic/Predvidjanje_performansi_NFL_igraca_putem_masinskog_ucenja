#!/usr/bin/env python3
"""
Combine all TE stat tables from data/raw/te/ into a single master CSV.

Each row = one TE's season. Columns come from merging:
  1. receiving_rushing.csv             - Core receiving & rushing stats  (all TEs)
  2. advanced_receiving_rushing.csv    - YBC, YAC, ADOT, drops, BrkTkl  (2018+)
  3. defense_fumbles.csv              - Fumbles, tackles, INTs           (most TEs)
  4. snap_counts.csv                  - Snap counts & percentages        (2012+)

Also writes intermediate all_te_*.csv files to data/processed/.

Output: data/fully combined/te_master.csv
"""

import pandas as pd
import os
import sys
import warnings

warnings.filterwarnings('ignore')

RAW_DIR = 'data/raw/te'
PROCESSED_DIR = os.path.join('data', 'processed')
OUTPUT_FILE = os.path.join('data', 'fully combined', 'te_master.csv')

# ── Merge keys ───────────────────────────────────────────────────────────────
KEY_COLS = ['Season', 'Player', 'PlayerID']
SHARED_COLS = ['Age', 'Team', 'Lg', 'Pos', 'G', 'GS']  # kept from receiving_rushing


def load_table(folder_path, filename):
    """Load a single CSV from a TE folder, return DataFrame or None."""
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


# ── Table processors ─────────────────────────────────────────────────────────

def process_receiving_rushing(df):
    """Process receiving_rushing.csv - this is the BASE table for TEs.

    Columns (from PFR): Tgt, Rec, Yds, Y/R, TD, 1D, rec_success, rec_long,
      R/G, Y/G, catch_pct, Y/Tgt, Att, Yds, TD, 1D, Succ%, Lng, Y/A, Y/G,
      A/G, Touch, yds_per_touch, yds_from_scrimmage, rush_receive_td,
      fumbles, av, awards

    Note: Receiving columns come first, then rushing columns. Some column
    names are ambiguous (Yds, TD, 1D, Y/G appear twice - once for rec,
    once for rush). Pandas appends .1 to duplicates on read.
    """
    if df is None:
        return None

    # Rename ambiguous .1 columns to make them clear
    # In receiving_and_rushing table: Rec cols first, then Rush cols
    renames = {
        # Receiving (primary for TEs - comes first, no suffix)
        'Tgt': 'Tgt', 'Rec': 'Rec',
        'Yds': 'Rec_Yds', 'Y/R': 'Rec_Y/R', 'TD': 'Rec_TD',
        '1D': 'Rec_1D', 'R/G': 'Rec_R/G', 'Y/G': 'Rec_Y/G',
        # Rushing (secondary - gets .1 suffix from pandas on duplicate)
        'Att': 'Rush_Att',
        'Yds.1': 'Rush_Yds', 'TD.1': 'Rush_TD', '1D.1': 'Rush_1D',
        'Succ%': 'Rush_Succ%', 'Lng': 'Rush_Lng',
        'Y/A': 'Rush_Y/A', 'Y/G.1': 'Rush_Y/G', 'A/G': 'Rush_A/G',
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    # Drop awards (text)
    df = df.drop(columns=['awards'], errors='ignore')

    return df


def process_advanced_receiving_rushing(df):
    """Process advanced_receiving_rushing.csv - YBC, YAC, ADOT, drops (2018+).

    Columns (from PFR): Tgt, Rec, Yds, 1D, YBC, YBC/R, YAC, YAC/R, ADOT,
      BrkTkl, Rec/Br, Drop, Drop%, Int, rec_pass_rating,
      Att, Yds, 1D, YBC, YBC/Att, YAC, YAC/Att, BrkTkl, Att/Br, awards

    Receiving advanced columns first, then rushing advanced columns.
    Duplicate names (Yds, 1D, YBC, YAC, BrkTkl) get .1 suffix for rushing.
    """
    if df is None:
        return None

    # Rename .1 columns to distinguish receiving vs rushing
    renames = {
        # Advanced Receiving (primary for TEs)
        'Tgt': 'Adv_Tgt', 'Rec': 'Adv_Rec',
        'Yds': 'Adv_Rec_Yds', '1D': 'Adv_Rec_1D',
        'YBC': 'Rec_YBC', 'YBC/R': 'Rec_YBC/R',
        'YAC': 'Rec_YAC', 'YAC/R': 'Rec_YAC/R',
        'BrkTkl': 'Rec_BrkTkl', 'Rec/Br': 'Rec_Att/Br',
        'Int': 'Rec_Int',
        # Advanced Rushing (secondary, gets .1 from pandas)
        'Att': 'Adv_Rush_Att',
        'Yds.1': 'Adv_Rush_Yds', '1D.1': 'Adv_Rush_1D',
        'YBC.1': 'Rush_YBC', 'YBC/Att': 'Rush_YBC/Att',
        'YAC.1': 'Rush_YAC', 'YAC/Att': 'Rush_YAC/Att',
        'BrkTkl.1': 'Rush_BrkTkl', 'Att/Br': 'Rush_Att/Br',
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    # Drop duplicate basic columns already in receiving_rushing + awards
    drop_cols = ['Adv_Tgt', 'Adv_Rec', 'Adv_Rec_Yds', 'Adv_Rec_1D',
                 'Adv_Rush_Att', 'Adv_Rush_Yds', 'Adv_Rush_1D',
                 'awards'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Prefix remaining columns
    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'adv', keep)
    return df


def process_defense_fumbles(df):
    """Process defense_fumbles.csv - fumbles, interceptions, tackles.

    Columns: Int, Yds, IntTD, Lng, PD, FF, Fmb, FR, Yds.1, FRTD,
             Sk, Comb, Solo, Ast, tackles_loss, QBHits, safety_md, awards
    """
    if df is None:
        return None

    # Rename .1 columns
    renames = {
        'Int': 'Def_Int', 'Yds': 'Def_Int_Yds', 'IntTD': 'Def_IntTD',
        'Lng': 'Def_Lng', 'Yds.1': 'Def_FR_Yds',
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    # Drop shared cols and awards
    drop_cols = ['awards'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Prefix
    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'def', keep)
    return df


def process_snap_counts(df):
    """Process snap_counts.csv - snap counts and percentages (2012+).

    Columns: No., offense, Off%, defense, Def%, special_teams, ST%
    """
    if df is None:
        return None

    # Drop No. (jersey number) and shared cols
    drop_cols = ['No.'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Prefix
    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'snp', keep)
    return df


# ── Intermediate combine (all_te_*.csv to data/processed/) ──────────────────

def combine_intermediate():
    """Combine individual TE CSVs into per-table files in data/processed/."""
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


# ── Main builder ─────────────────────────────────────────────────────────────

def build_master():
    """Build the master TE CSV from all raw data."""

    print("=" * 70)
    print("BUILDING TE MASTER TABLE")
    print("=" * 70)

    # ── Step 0: Write intermediate combined files to data/processed/ ─────
    combine_intermediate()

    # ── Step 1: Find all TE folders ──────────────────────────────────────
    folders = sorted([f for f in os.listdir(RAW_DIR)
                      if os.path.isdir(os.path.join(RAW_DIR, f))])
    print(f"\nFound {len(folders)} TE folders in {RAW_DIR}")

    # Table configs: (filename, processor_function)
    table_configs = [
        ('receiving_rushing.csv', process_receiving_rushing),
        ('advanced_receiving_rushing.csv', process_advanced_receiving_rushing),
        ('defense_fumbles.csv', process_defense_fumbles),
        ('snap_counts.csv', process_snap_counts),
    ]

    # ── Step 2: Process each TE folder ───────────────────────────────────
    all_player_dfs = []
    table_counts = {t[0]: 0 for t in table_configs}

    for folder in folders:
        folder_path = os.path.join(RAW_DIR, folder)

        # Load receiving_rushing.csv first - it's the base
        base_df = load_table(folder_path, 'receiving_rushing.csv')
        if base_df is None:
            # Fallback: try defense_fumbles as base if no receiving_rushing
            base_df = load_table(folder_path, 'defense_fumbles.csv')
            if base_df is None:
                print(f"  Skipping {folder} - no receiving_rushing or defense_fumbles")
                continue
            base_df = process_defense_fumbles(base_df)
            table_counts['defense_fumbles.csv'] += 1
            # Try snap_counts
            snap_df = load_table(folder_path, 'snap_counts.csv')
            processed = process_snap_counts(snap_df)
            if processed is not None:
                table_counts['snap_counts.csv'] += 1
                base_df = base_df.merge(processed, on=KEY_COLS, how='left')
            all_player_dfs.append(base_df)
            continue

        base_df = process_receiving_rushing(base_df)
        table_counts['receiving_rushing.csv'] += 1

        # Merge other tables into the base
        for filename, processor in table_configs[1:]:
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

    # ── Step 3: Concatenate all players ──────────────────────────────────
    master = pd.concat(all_player_dfs, ignore_index=True)
    print(f"\nCombined data: {len(master)} rows, {master['Player'].nunique()} TEs, {len(master.columns)} columns")

    # Print table coverage
    print("\nTable coverage:")
    for filename, count in table_counts.items():
        label = filename.replace('.csv', '')
        pct = count / len(folders) * 100
        print(f"  {label:35s} {count:3d}/{len(folders)} TEs ({pct:.0f}%)")

    # ── Step 4: Reorder columns ──────────────────────────────────────────
    # Identity columns first
    identity = ['Player', 'PlayerID', 'Season', 'Age', 'Team', 'Lg', 'Pos', 'G', 'GS']
    identity = [c for c in identity if c in master.columns]

    # Core receiving/rushing (from base table, no prefix)
    rec_rush = [c for c in master.columns
                if c not in identity and not c.startswith(('adv_', 'def_', 'snp_'))]
    # Advanced receiving/rushing
    adv = sorted([c for c in master.columns if c.startswith('adv_')])
    # Defense/fumbles
    defense = sorted([c for c in master.columns if c.startswith('def_')])
    # Snap counts
    snaps = sorted([c for c in master.columns if c.startswith('snp_')])

    ordered = identity + rec_rush + adv + defense + snaps
    # Catch any missed columns
    missed = [c for c in master.columns if c not in ordered]
    if missed:
        print(f"\n  Note: {len(missed)} uncategorized columns appended: {missed}")
    ordered += missed

    master = master[[c for c in ordered if c in master.columns]]

    # ── Step 5: Sort and save ────────────────────────────────────────────
    master = master.sort_values(['Player', 'Season']).reset_index(drop=True)

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

    # Column group summary
    print(f"\nColumn groups:")
    print(f"  Identity:               {len(identity)}")
    print(f"  Receiving/Rushing:      {len(rec_rush)}")
    print(f"  Adv Rec/Rush (2018+):   {len(adv)}")
    print(f"  Defense/Fumbles:        {len(defense)}")
    print(f"  Snap Counts:            {len(snaps)}")
    print(f"  Total:                  {len(master.columns)}")

    return master


if __name__ == '__main__':
    build_master()
