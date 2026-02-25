#!/usr/bin/env python3
"""
Combine all QB stat tables from data/raw/qb/ into a single master CSV.

Each row = one QB's season. Columns come from merging:
  1. passing.csv           - Core passing stats (all QBs)
  2. adjusted_passing.csv  - Era-adjusted passing indexes (all QBs)
  3. advanced_passing.csv  - Air yards, drops, pressure, RPO, play action (2018+)
  4. rushing_receiving.csv - Rush/rec stats (all QBs)
  5. advanced_rushing_receiving.csv - YBC, YAC, broken tackles (2018+)
  6. defense_fumbles.csv   - Fumbles, tackles, sacks (all QBs)
  7. snap_counts.csv       - Snap counts & percentages (2017+)
  + data/rankings/qb_rankings_career.csv - Elo ratings, EPA components, WPA

Output: data/processed/qb_master.csv
"""

import pandas as pd
import os
import sys
import warnings

warnings.filterwarnings('ignore')

RAW_DIR = 'data/raw/qb'
RANKINGS_DIR = 'data/rankings'
OUTPUT_FILE = 'data/processed/qb_master.csv'


# ── Column prefixes to avoid collisions during merge ─────────────────────────
# Each table's non-key columns get a prefix so we know where they came from.
# Key columns (Season, Age, Team, Pos, G, GS, Player, PlayerID) are shared.

# Columns that are shared "key" or identity columns (used to join, kept once)
KEY_COLS = ['Season', 'Player', 'PlayerID']
SHARED_COLS = ['Age', 'Team', 'Pos', 'G', 'GS']  # kept from passing.csv


def load_table(folder_path, filename):
    """Load a single CSV from a QB folder, return DataFrame or None."""
    filepath = os.path.join(folder_path, filename)
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None
        # Ensure Season is numeric
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
    """Process passing.csv - this is the base table."""
    if df is None:
        return None
    # Drop Awards column (text, not useful for analysis)
    if 'Awards' in df.columns:
        df = df.drop('Awards', axis=1)
    return df


def process_adjusted_passing(df):
    """Process adjusted_passing.csv - era-adjusted indexes."""
    if df is None:
        return None
    drop_cols = ['Awards'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    # Prefix columns
    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'adj', keep)
    return df


def process_advanced_passing(df):
    """Process advanced_passing.csv - air yards, drops, pressure, etc."""
    if df is None:
        return None
    # Drop duplicated basic columns and Awards
    drop_cols = ['Awards', 'Cmp', 'Att'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    # Prefix
    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'adv_pass', keep)
    return df


def process_rushing_receiving(df):
    """Process rushing_receiving.csv - basic rush/rec stats."""
    if df is None:
        return None
    drop_cols = ['Awards'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'rr', keep)
    return df


def process_advanced_rushing_receiving(df):
    """Process advanced_rushing_receiving.csv - YBC, YAC, broken tackles."""
    if df is None:
        return None
    # Drop duplicated rushing columns that are already in rushing_receiving
    dup_cols = ['Rush_Att', 'Rush_Yds', 'Rush_1D', 'Tgt', 'Rec', 'Rec_Yds', 'Rec_1D',
                'Awards'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in dup_cols if c in df.columns], errors='ignore')
    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'adv_rr', keep)
    return df


def process_defense_fumbles(df):
    """Process defense_fumbles.csv - fumbles, tackles."""
    if df is None:
        return None
    drop_cols = ['Awards'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'def', keep)
    return df


def process_snap_counts(df):
    """Process snap_counts.csv - snap counts and percentages."""
    if df is None:
        return None
    # Drop No. (uniform number) and shared cols
    drop_cols = ['No.'] + [c for c in SHARED_COLS if c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    keep = KEY_COLS.copy()
    df = prefix_columns(df, 'snp', keep)
    return df


def load_rankings():
    """Load and prepare the career rankings data."""
    filepath = os.path.join(RANKINGS_DIR, 'qb_rankings_career.csv')
    if not os.path.exists(filepath):
        print("  Rankings file not found, skipping rankings merge.")
        return None

    df = pd.read_csv(filepath)
    print(f"  Rankings data loaded: {len(df)} rows, {df['QB'].nunique()} QBs")

    # Rename QB -> Player for merge key
    df = df.rename(columns={'QB': 'Player'})

    # Prefix all stat columns to avoid collisions
    keep = ['Player', 'Season']
    df = prefix_columns(df, 'elo', keep)

    return df


def build_master():
    """Build the master QB CSV from all raw data + rankings."""

    print("=" * 70)
    print("BUILDING QB MASTER TABLE")
    print("=" * 70)

    # ── Step 1: Load all QB folders ──────────────────────────────────────
    folders = sorted([f for f in os.listdir(RAW_DIR)
                      if os.path.isdir(os.path.join(RAW_DIR, f))])
    print(f"\nFound {len(folders)} QB folders in {RAW_DIR}")

    # Table configs: (filename, processor_function)
    table_configs = [
        ('passing.csv', process_passing),
        ('adjusted_passing.csv', process_adjusted_passing),
        ('advanced_passing.csv', process_advanced_passing),
        ('rushing_receiving.csv', process_rushing_receiving),
        ('advanced_rushing_receiving.csv', process_advanced_rushing_receiving),
        ('defense_fumbles.csv', process_defense_fumbles),
        ('snap_counts.csv', process_snap_counts),
    ]

    # ── Step 2: Process each QB folder ───────────────────────────────────
    all_player_dfs = []
    table_counts = {t[0]: 0 for t in table_configs}

    for folder in folders:
        folder_path = os.path.join(RAW_DIR, folder)

        # Load passing.csv first - it's the base
        base_df = load_table(folder_path, 'passing.csv')
        if base_df is None:
            print(f"  Skipping {folder} - no passing.csv")
            continue

        base_df = process_passing(base_df)
        table_counts['passing.csv'] += 1

        # Merge other tables into the base using Season + Player + PlayerID
        for filename, processor in table_configs[1:]:  # skip passing (already loaded)
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
    print(f"\nCombined PFR data: {len(master)} rows, {master['Player'].nunique()} QBs, {len(master.columns)} columns")

    # Print table coverage
    print("\nTable coverage:")
    for filename, count in table_counts.items():
        label = filename.replace('.csv', '')
        pct = count / len(folders) * 100
        print(f"  {label:35s} {count:3d}/{len(folders)} QBs ({pct:.0f}%)")

    # ── Step 4: Merge rankings data ──────────────────────────────────────
    print("\n--- Merging Rankings/Elo Data ---")
    rankings = load_rankings()
    if rankings is not None:
        # Check name matching before merge
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
        # Count how many rows got rankings data
        elo_cols = [c for c in master.columns if c.startswith('elo_')]
        if elo_cols:
            has_elo = master[elo_cols[0]].notna().sum()
            print(f"  Rankings data matched: {has_elo}/{len(master)} season-rows ({has_elo/len(master)*100:.1f}%)")

    # ── Step 5: Reorder columns ──────────────────────────────────────────
    # Put identity/key columns first, then group by source
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
    # Catch any missed columns
    missed = [c for c in master.columns if c not in ordered]
    if missed:
        print(f"\n  Note: {len(missed)} uncategorized columns appended: {missed}")
    ordered += missed

    master = master[[c for c in ordered if c in master.columns]]

    # ── Step 6: Sort and save ────────────────────────────────────────────
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

    # Column group summary
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
