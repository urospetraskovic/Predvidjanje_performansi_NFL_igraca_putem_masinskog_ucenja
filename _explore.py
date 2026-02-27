import pandas as pd
import numpy as np

qb = pd.read_csv('data/fully combined/qb_master.csv')
rb = pd.read_csv('data/fully combined/rb_master.csv')
te = pd.read_csv('data/fully combined/te_master.csv')
wr = pd.read_csv('data/fully combined/wr_all_seasons.csv')
elo = pd.read_csv('data/nfl elo data/qb_rankings_career.csv')

print("=== QB COLUMNS ===")
print(list(qb.columns))
print(f"\nShape: {qb.shape}")
print(f"Seasons: {qb.Season.min()}-{qb.Season.max()}")
has_qbrec = 'QBrec' in qb.columns
print(f"Has QBrec: {has_qbrec}")
if has_qbrec:
    print(f"Sample QBrec: {qb.QBrec.dropna().head(3).tolist()}")

# Check numeric ranges for key columns
print("\nQB key stats:")
for c in ['Yds', 'TD', 'Int', 'Rate', 'Cmp%', 'Y/A', 'Sk', 'G', 'GS', 'Age']:
    if c in qb.columns:
        print(f"  {c}: min={qb[c].min()}, max={qb[c].max()}, mean={qb[c].mean():.1f}")

print("\n=== RB COLUMNS ===")
print(list(rb.columns))
print(f"\nShape: {rb.shape}")
for c in ['Rush_Yds', 'Rush_TD', 'Rush_Att', 'Rec_Yds', 'Rec', 'Tgt', 'Rush_Y/A']:
    if c in rb.columns:
        print(f"  {c}: min={rb[c].min()}, max={rb[c].max()}, mean={rb[c].mean():.1f}")

print("\n=== TE COLUMNS ===")
print(list(te.columns))
print(f"\nShape: {te.shape}")
for c in ['Rec_Yds', 'Rec_TD', 'Rec', 'Tgt', 'Rec_Y/R', 'catch_pct']:
    if c in te.columns:
        print(f"  {c}: min={te[c].min()}, max={te[c].max()}, mean={te[c].mean():.1f}")

print("\n=== WR COLUMNS ===")
print(list(wr.columns))
print(f"\nShape: {wr.shape}")
for c in ['receiving_yards', 'tds', 'targets', 'receptions', 'epa', 'target_share', 'adot', 'catch_rate', 'air_yards', 'yac']:
    if c in wr.columns:
        print(f"  {c}: min={wr[c].min():.2f}, max={wr[c].max():.2f}, mean={wr[c].mean():.2f}")

print("\n=== ELO COLUMNS ===")
print(list(elo.columns))
print(f"\nShape: {elo.shape}")

# Check for win-related columns across datasets
print("\n=== WIN-RELATED DATA ===")
for name, df in [('QB', qb), ('RB', rb), ('TE', te), ('WR', wr)]:
    win_cols = [c for c in df.columns if 'win' in c.lower() or 'rec' in c.lower() or 'loss' in c.lower() or 'wpa' in c.lower()]
    print(f"  {name}: {win_cols}")

# ELO win data
elo_win_cols = [c for c in elo.columns if 'win' in c.lower() or 'elo' in c.lower() or 'rank' in c.lower() or 'value' in c.lower()]
print(f"  ELO: {elo_win_cols}")

# Check advanced columns we can use
print("\n=== ADVANCED METRIC COLUMNS ===")
for name, df in [('QB', qb)]:
    adv_cols = [c for c in df.columns if any(p in c for p in ['adj_', 'adv_', 'rr_', 'def_', 'snp_'])]
    print(f"  {name} advanced ({len(adv_cols)}): {adv_cols[:30]}...")

print("\n=== QB sample data (first 3 rows key cols) ===")
key_cols = ['Player', 'Season', 'Age', 'Team', 'G', 'GS', 'QBrec', 'Yds', 'TD', 'Int', 'Rate', 'Cmp%']
print(qb[[c for c in key_cols if c in qb.columns]].head(3).to_string())
