import pandas as pd
import os

# Load the rushing_receiving data (2025 season stats)
csv_path = "data/processed/all_rushing_receiving.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    
    print("=" * 70)
    print("2025 RB SEASON DATA VERIFICATION")
    print("=" * 70)
    print(f"\nFile: {csv_path}")
    print(f"Total rows: {len(df)}")
    print(f"Total players: {df['Player'].nunique()}")
    
    # Game count per player
    print("\n[GAME COUNT PER PLAYER]")
    games_per_player = df.groupby('Player').size().sort_values(ascending=False)
    print(games_per_player)
    
    print(f"\nMax games: {games_per_player.max()}")
    print(f"Min games: {games_per_player.min()}")
    print(f"Players with 17 games: {len(games_per_player[games_per_player == 17])}")
    print(f"Players with <17 games: {len(games_per_player[games_per_player < 17])}")
    
    # Season year check
    if 'Year' in df.columns:
        print(f"\nYears present: {sorted(df['Year'].unique())}")
    
else:
    print(f"File not found: {csv_path}")
