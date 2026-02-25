#!/usr/bin/env python3
"""
Combine all WR season CSVs (2015-2025) into:
  1. wr_all_weeks.csv       - Every row from every year (game-level data)
  2. wr_all_seasons.csv     - Aggregated per player per season (1 row = 1 player-season)

Reads from: data/raw/wr/{year}/data/wr_{year}.csv
Writes to:  data/fully combined/wr_all_weeks.csv
            data/fully combined/wr_all_seasons.csv
Also writes data/processed/all_wr_stats.csv (copy of wr_all_weeks for consistency)
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

RAW_DIR = 'data/raw/wr'
OUTPUT_DIR = os.path.join('data', 'fully combined')
PROCESSED_DIR = os.path.join('data', 'processed')
YEARS = range(2015, 2026)


def load_all_years():
    """Load and concatenate all year CSVs."""
    frames = []
    for year in YEARS:
        path = os.path.join(RAW_DIR, str(year), 'data', f'wr_{year}.csv')
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        df = pd.read_csv(path)
        # Extract season year from game_id (e.g. "2025_01_ARI_NO" -> 2025)
        df['season'] = year
        frames.append(df)
        print(f"  {year}: {len(df)} rows, {df['receiver_player_name'].nunique()} players")

    combined = pd.concat(frames, ignore_index=True)
    return combined


def build_season_aggregated(df):
    """Aggregate game-level rows into per-player per-season rows.

    - Counting stats (targets, receptions, yards, tds, etc.) are SUMMED.
    - Rate/average stats (catch_rate, td_rate, Y/R, etc.) are AVERAGED.
    - Categorical/ID columns are kept as first occurrence or mode.
    """

    # Identify column types
    id_cols = ['receiver_player_id', 'receiver_player_name', 'season']

    # Columns to sum (counting/accumulating stats)
    sum_cols = [
        'targets', 'receptions', 'receiving_yards', 'air_yards', 'yac', 'tds',
        'red_zone_targets', 'end_zone_targets', 'third_down_targets',
        'fourth_down_targets', 'high_leverage_targets',
        'explosive_plays', 'first_downs',
        'second_and_long_targets', 'third_and_medium_targets',
        'lost_yards_due_to_penalty',
        # Quarter splits
        'yards_Q1', 'yards_Q2', 'yards_Q3', 'yards_Q4',
        'receptions_Q1', 'receptions_Q2', 'receptions_Q3', 'receptions_Q4',
        'targets_Q1', 'targets_Q2', 'targets_Q3', 'targets_Q4',
        # Win probability splits
        'yards_wp_<25', 'yards_wp_25_45', 'yards_wp_45_55',
        'yards_wp_55_75', 'yards_wp_>75', 'yards_wp_NA',
        'receptions_wp_<25', 'receptions_wp_25_45', 'receptions_wp_45_55',
        'receptions_wp_55_75', 'receptions_wp_>75', 'receptions_wp_NA',
        'targets_wp_<25', 'targets_wp_25_45', 'targets_wp_45_55',
        'targets_wp_55_75', 'targets_wp_>75', 'targets_wp_NA',
    ]

    # Columns to average (rates, per-play metrics, contextual)
    mean_cols = [
        'epa', 'wpa', 'avg_depth', 'catch_rate',
        'team_pass_attempts', 'team_air_yards', 'team_epa',
        'air_yard_share', 'target_share', 'yards_per_target',
        'def_targets_dev', 'def_receptions_dev', 'def_yards_dev',
        'def_tds_dev', 'def_epa_dev',
        'qb_completions', 'qb_attempts', 'qb_air_yards', 'qb_cpoe', 'qb_comp_pct',
        'avg_score_diff', 'avg_quarter', 'adot', 'yac_per_reception',
        'td_rate', 'trailing_pct', 'leading_pct', 'wp_var',
        'target_share_std',
        'temp_f', 'humidity_pct', 'wind_mph',
        'success_rate', 'big_play_rate', 'reception_std',
        'avg_start_yardline', 'avg_target_depth_vs_qb',
        'pregame_spread', 'pregame_total',
    ]

    # Columns to take first/mode (categorical, IDs)
    first_cols = [
        'passer_player_id', 'posteam',
    ]

    # Binary/categorical to average (gives proportion of games)
    prop_cols = [
        'surface', 'is_dome', 'is_rain', 'is_snow', 'is_clear',
    ]

    # Build aggregation dict
    agg_dict = {}

    # Add game count
    agg_dict['game_id'] = 'count'  # becomes number of games

    for col in sum_cols:
        if col in df.columns:
            agg_dict[col] = 'sum'

    for col in mean_cols:
        if col in df.columns:
            agg_dict[col] = 'mean'

    for col in first_cols:
        if col in df.columns:
            agg_dict[col] = 'first'

    for col in prop_cols:
        if col in df.columns:
            agg_dict[col] = 'mean'

    # Any remaining columns not yet assigned - default to mean
    assigned = set(list(agg_dict.keys()) + id_cols)
    for col in df.columns:
        if col not in assigned:
            agg_dict[col] = 'mean'

    grouped = df.groupby(id_cols, as_index=False).agg(agg_dict)

    # Rename game_id count to games_played
    grouped = grouped.rename(columns={'game_id': 'games_played'})

    # Reorder: id cols first, then games_played, then rest
    priority = ['receiver_player_id', 'receiver_player_name', 'season',
                'games_played', 'posteam']
    priority = [c for c in priority if c in grouped.columns]
    other = [c for c in grouped.columns if c not in priority]
    grouped = grouped[priority + other]

    return grouped


def main():
    print("=" * 70)
    print("WR DATA COMBINER")
    print("=" * 70)

    # ── Step 1: Load all years ───────────────────────────────────────────
    print(f"\nLoading WR data from {len(list(YEARS))} seasons...")
    all_data = load_all_years()
    print(f"\nTotal: {len(all_data)} rows, {all_data['receiver_player_name'].nunique()} unique players")

    # ── Step 2: Save big combined CSV (all weeks) ────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_weeks_path = os.path.join(OUTPUT_DIR, 'wr_all_weeks.csv')
    all_data.to_csv(all_weeks_path, index=False)
    size_kb = os.path.getsize(all_weeks_path) / 1024
    print(f"\n[OK] All weeks saved: {all_weeks_path}")
    print(f"     {len(all_data)} rows, {len(all_data.columns)} cols, {size_kb:.1f} KB")

    # Also save to processed for consistency
    processed_path = os.path.join(PROCESSED_DIR, 'all_wr_stats.csv')
    all_data.to_csv(processed_path, index=False)
    print(f"[OK] Copy saved: {processed_path}")

    # ── Step 3: Build season-aggregated CSV ──────────────────────────────
    print(f"\nAggregating to per-player per-season...")
    season_data = build_season_aggregated(all_data)
    print(f"  {len(season_data)} player-seasons from {season_data['receiver_player_name'].nunique()} players")

    season_path = os.path.join(OUTPUT_DIR, 'wr_all_seasons.csv')
    season_data.to_csv(season_path, index=False)
    size_kb = os.path.getsize(season_path) / 1024
    print(f"\n[OK] Season aggregated saved: {season_path}")
    print(f"     {len(season_data)} rows, {len(season_data.columns)} cols, {size_kb:.1f} KB")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"SUMMARY:")
    print(f"  wr_all_weeks.csv:   {len(all_data)} rows (game-level, 2015-2025)")
    print(f"  wr_all_seasons.csv: {len(season_data)} rows (player-season aggregated)")
    print(f"  Seasons: {all_data['season'].min()} - {all_data['season'].max()}")
    print(f"  Players: {all_data['receiver_player_name'].nunique()} unique")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
