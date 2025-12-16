#!/usr/bin/env python3
"""
NFL RB Stats Scraper - Production Version
Scrapes Pro Football Reference for RB statistics across multiple stat tables.
Extracts: Rushing & Receiving, Advanced Rushing & Receiving, Defense & Fumbles, 
          and Snap Counts.

Usage:
    python nfl_rb_scraper.py                    # Scrape all RBs in the list
    python nfl_rb_scraper.py --player HenrDe00  # Scrape single player
    python nfl_rb_scraper.py --test             # Test with one player
    python nfl_rb_scraper.py --force            # Re-scrape even if exists
    python nfl_rb_scraper.py --combine          # Combine all CSVs
"""

import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random
import argparse
import re
import shutil
from datetime import datetime


# =============================================================================
# RB DATA - Player IDs and Names
# =============================================================================
# Format: (PlayerID, Full Name)
# Find player IDs at: https://www.pro-football-reference.com/players/

NFL_RBS = [
    # Top Rushers This Season
    ('TaylJo01', 'Jonathan Taylor'),
    ('CookJa01', 'James Cook'),
    ('AchaDe00', 'De\'Von Achane'),
    ('RobiBi01', 'Bijan Robinson'),
    ('HenrDe00', 'Derrick Henry'),
    ('WillJa02', 'Javonte Williams'),
    ('GibbJa00', 'Jahmyr Gibbs'),
    ('WillKy00', 'Kyren Williams'),
    ('DowdRi00', 'Rico Dowdle'),
    ('EtieTr00', 'Travis Etienne'),
    ('BarkSa00', 'Saquon Barkley'),
    ('SwifDA00', 'D\'Andre Swift'),
    ('McCaCh01', 'Christian McCaffrey'),
    ('HallBr02', 'Breece Hall'),
    ('JackJo05', 'Josh Jacobs'),
    ('PollTo00', 'Tony Pollard'),
    ('JudkQu00', 'Quinshon Judkins'),
    ('BrowCh01', 'Chase Brown'),
    ('WalkKe00', 'Kenneth Walker III'),
    ('HendTr01', 'TreVeyon Henderson'),
    ('DobbJ.00', 'J.K. Dobbins'),
    ('JeanAs00', 'Ashton Jeanty'),
    ('WarrJa00', 'Jaylen Warren'),
    ('MonaKy00', 'Kyle Monangai'),
    ('MasoJo00', 'Jordan Mason'),
]


# =============================================================================
# TABLE DEFINITIONS - Maps table IDs to filenames
# =============================================================================

TABLES_TO_SCRAPE = {
    'rushing_and_receiving': {
        'filename': 'rushing_receiving.csv',
        'description': 'Rushing and Receiving Stats'
    },
    'adv_rushing_and_receiving': {
        'filename': 'advanced_rushing_receiving.csv',
        'description': 'Advanced Rushing and Receiving Stats'
    },
    'defense': {
        'filename': 'defense_fumbles.csv',
        'description': 'Defense and Fumbles'
    },
    'snap_counts': {
        'filename': 'snap_counts.csv',
        'description': 'Snap Counts'
    },
}

# Common column name mappings for cleaner output
COLUMN_MAPPING = {
    # Basic info
    'year_id': 'Season',
    'age': 'Age',
    'team': 'Team',
    'team_name_abbr': 'Team',
    'tm': 'Team',
    'lg_id': 'Lg',
    'comp_name_abbr': 'Lg',
    'pos': 'Pos',
    'uniform_number': 'No.',
    'number': 'No.',
    
    # Games
    'g': 'G',
    'games': 'G',
    'gs': 'GS',
    'games_started': 'GS',
    
    # Rushing
    'rush_att': 'Att',
    'rush_yds': 'Yds',
    'rush_td': 'TD',
    'rush_first_down': '1D',
    'rush_success': 'Succ%',
    'rush_long': 'Lng',
    'rush_yds_per_att': 'Y/A',
    'rush_yds_per_g': 'Y/G',
    'rush_att_per_g': 'A/G',
    
    # Receiving
    'targets': 'Tgt',
    'rec': 'Rec',
    'rec_yds': 'Yds',
    'rec_yds_per_rec': 'Y/R',
    'rec_td': 'TD',
    'rec_first_down': '1D',
    'rec_per_g': 'R/G',
    'rec_yds_per_g': 'Y/G',
    'rec_catch_pct': 'Ctch%',
    'rec_yds_per_tgt': 'Y/Tgt',
    
    # Scrimmage
    'touches': 'Touch',
    'yds_touch': 'Y/Tch',
    'rush_rec_td': 'RRTD',
    'scrimmage_yds': 'YScm',
    
    # Advanced Rushing and Receiving
    'rush_yds_before_contact': 'YBC',
    'rush_yds_bc_per_rush': 'YBC/Att',
    'rush_yac': 'YAC',
    'rush_yac_per_rush': 'YAC/Att',
    'rush_broken_tackles': 'BrkTkl',
    'rush_broken_tackles_per_rush': 'Att/Br',
    'rec_air_yds': 'YBC',
    'rec_air_yds_per_rec': 'YBC/R',
    'rec_yac': 'YAC',
    'rec_yac_per_rec': 'YAC/R',
    'rec_adot': 'ADOT',
    'rec_broken_tackles': 'BrkTkl',
    'rec_broken_tackles_per_rec': 'Rec/Br',
    'rec_drops': 'Drop',
    'rec_drop_pct': 'Drop%',
    'rec_target_int': 'Int',
    'pass_rating': 'Rat',
    
    # Defense / Fumbles
    'def_int': 'Int',
    'def_int_yds': 'Yds',
    'def_int_td': 'IntTD',
    'def_int_long': 'Lng',
    'pass_defended': 'PD',
    'fumbles_forced': 'FF',
    'fumbles': 'Fmb',
    'fumbles_rec': 'FR',
    'fumbles_rec_yds': 'Yds',
    'fumbles_rec_td': 'FRTD',
    'sacks': 'Sk',
    'tackles_combined': 'Comb',
    'tackles_solo': 'Solo',
    'tackles_assists': 'Ast',
    'tackles_for_loss': 'TFL',
    'qb_hits': 'QBHits',
    'safeties': 'Sfty',
    
    # Snap counts
    'off_pct': 'Off%',
    'def_pct': 'Def%',
    'st_pct': 'ST%',
    'snap_counts_offdef': 'Num',
    'snap_counts': 'Num',
}


class NFLRBScraper:
    """
    Scraper for Pro Football Reference RB statistics.
    Extracts multiple stat tables and saves to organized folders.
    """
    
    def __init__(self, output_dir='data/raw/rb', verbose=True, delay_min=60, delay_max=120):
        self.output_dir = output_dir
        self.verbose = verbose
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.scraper = cloudscraper.create_scraper()
        # Add proper headers to avoid 403 blocks
        self.scraper.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        self.stats = {'success': 0, 'failed': 0, 'skipped': 0}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def log(self, msg, end='\n'):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(msg, end=end, flush=True)
    
    def delay(self):
        """No delay - go fast!"""
        pass
    
    def _get_player_folder_name(self, player_name):
        """Convert player name to folder-safe name."""
        safe_name = re.sub(r'[^\w\s-]', '', player_name)
        safe_name = safe_name.replace(' ', '_')
        return safe_name
    
    def scrape_player(self, player_id, player_name=None, save=True, force=False):
        """
        Scrape all stat tables for a single RB.
        
        Args:
            player_id: PFR player ID (e.g., 'HenrDe00')
            player_name: Display name for the player
            save: Whether to save to CSV files
            force: Whether to overwrite existing player folder
            
        Returns:
            Dictionary with table DataFrames, or None if failed
        """
        first_letter = player_id[0].upper()
        url = f"https://www.pro-football-reference.com/players/{first_letter}/{player_id}.htm"
        
        display_name = player_name or player_id
        self.log(f"\n> Scraping: {display_name}")
        self.log(f"  URL: {url}")
        
        try:
            # Fetch page with retry logic for rate limiting
            self.log("  Fetching...", end=' ')
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = self.scraper.get(url, timeout=15)
                    response.raise_for_status()
                    break
                except Exception as e:
                    if '429' in str(e) and attempt < max_retries - 1:
                        self.log(f"(rate limited, retry {attempt+1})", end=' ')
                        time.sleep(30)  # Wait 30 seconds on 429
                    else:
                        raise
            self.log("[OK]")
            
            # Parse HTML (try lxml first, fallback to html.parser)
            try:
                soup = BeautifulSoup(response.content, 'lxml')
            except Exception:
                soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get player name from page
            h1 = soup.find('h1')
            actual_name = h1.get_text(strip=True) if h1 else display_name
            self.log(f"  Found: {actual_name}")
            
            # Create player folder
            folder_name = self._get_player_folder_name(actual_name)
            player_dir = os.path.join(self.output_dir, folder_name)
            
            # Check if folder exists and handle force flag
            if os.path.exists(player_dir):
                if force:
                    shutil.rmtree(player_dir)
                    os.makedirs(player_dir)
                else:
                    self.log(f"  ! Skipped (exists, use --force to overwrite)")
                    self.stats['skipped'] += 1
                    return None
            else:
                os.makedirs(player_dir, exist_ok=True)
            
            # Extract tables
            results = {}
            tables_found = 0
            
            for table_id, table_info in TABLES_TO_SCRAPE.items():
                # Find table (some are in comments, need to parse those too)
                table = soup.find('table', {'id': table_id})
                
                # If not found, check in comments (PFR hides some tables in comments)
                if not table:
                    for element in soup.find_all(string=True):
                        if isinstance(element, str) and table_id in element:
                            try:
                                comment_soup = BeautifulSoup(element, 'lxml')
                                table = comment_soup.find('table', {'id': table_id})
                                if table:
                                    break
                            except:
                                pass
                
                if not table:
                    continue
                
                # Parse table
                try:
                    df = self._parse_table(table)
                    if df is None:
                        continue
                    
                    # Add player info
                    df['Player'] = actual_name
                    df['PlayerID'] = player_id
                    
                    # Clean data
                    df = self._clean_dataframe(df)
                    if df is None:
                        continue
                except Exception as e:
                    self.log(f"    ! Error parsing table {table_id}: {str(e)[:60]}")
                    continue
                
                # Save to CSV
                if save:
                    filepath = os.path.join(player_dir, table_info['filename'])
                    # Check if we already saved this filename for a different table_id
                    target_filename = table_info['filename']
                    already_saved = False
                    for prev_id, prev_info in TABLES_TO_SCRAPE.items():
                        if prev_id < table_id and prev_info['filename'] == target_filename:
                            if prev_id in results:
                                already_saved = True
                                break
                    
                    if not already_saved:
                        df.to_csv(filepath, index=False)
                        tables_found += 1
                
                results[table_id] = df
            
            if tables_found > 0:
                self.log(f"  [OK] Extracted {tables_found} stat tables")
                self.log(f"  [OK] Saved to: {player_dir}")
                self.stats['success'] += 1
            else:
                self.log(f"  ! No stat tables found")
                self.stats['failed'] += 1
            
            return results
            
        except Exception as e:
            self.log(f"  ! Error: {str(e)[:80]}")
            self.stats['failed'] += 1
            return None
    
    def _parse_table(self, table):
        """
        Parse table using data-stat attributes for reliability.
        """
        rows = []
        tbody = table.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                # Skip rows that are dividers or headers
                if tr.get('class'):
                    classes = tr.get('class', [])
                    if 'thead' in classes or 'partial_table' in classes:
                        continue
                
                row_data = {}
                for cell in tr.find_all(['th', 'td']):
                    data_stat = cell.get('data-stat', '')
                    if data_stat and data_stat != 'ranker':
                        value = cell.get_text(strip=True)
                        row_data[data_stat] = value
                
                if row_data:
                    rows.append(row_data)
        
        if not rows:
            return None
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Apply column mapping
        df = df.rename(columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns})
        
        return df
    
    def _clean_dataframe(self, df):
        """Clean and validate the dataframe."""
        df = df.copy()
        
        # Ensure dataframe is not empty
        if df.empty:
            return None
        
        # Remove summary rows (Career, averages, etc.) - handle both string and non-string Season column
        if 'Season' in df.columns:
            df['Season'] = df['Season'].astype(str)
            df = df[~df['Season'].str.contains('Career|Avg|yr', case=False, na=False)]
        
        # If dataframe is now empty after filtering, return None
        if df.empty:
            return None
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Convert numeric columns (but be careful with Season, Team, Pos, etc.)
        string_cols = {'Season', 'Team', 'Pos', 'Lg', 'Player', 'PlayerID', 'Awards', 'No.', 'Tm'}
        for col in df.columns:
            if col not in string_cols and isinstance(df[col], (pd.Series, list, tuple)):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    # If conversion fails, leave as string
                    pass
        
        return df
    
    def scrape_all(self, force=False):
        """Scrape all RBs in the list."""
        total = len(NFL_RBS)
        for i, (player_id, name) in enumerate(NFL_RBS, 1):
            self.log(f"\n[{i}/{total}]")
            self.scrape_player(player_id, name, force=force)
        
        self.log_summary()
        self.combine_all_csvs()
    
    def log_summary(self):
        """Log scraping summary."""
        self.log("\n" + "=" * 70)
        self.log("RESULTS:")
        self.log(f"  [OK] Success: {self.stats['success']}")
        self.log(f"  >> Skipped: {self.stats['skipped']}")
        self.log(f"  ! Failed:  {self.stats['failed']}")
        self.log(f"  Total:     {sum(self.stats.values())}")
        self.log("=" * 70)
    
    def combine_all_csvs(self):
        """Combine individual player CSV files into master files by table type."""
        self.log("\nCombining CSV files from data/raw/rb...")
        
        # Get all player folders
        player_folders = [d for d in os.listdir(self.output_dir) 
                         if os.path.isdir(os.path.join(self.output_dir, d))]
        
        # Get all table filenames from TABLES_TO_SCRAPE
        table_filenames = set(info['filename'] for info in TABLES_TO_SCRAPE.values())
        
        # Combine each table type
        output_dir = 'data/processed'
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in sorted(table_filenames):
            dfs = []
            
            for player_folder in sorted(player_folders):
                filepath = os.path.join(self.output_dir, player_folder, filename)
                if os.path.exists(filepath):
                    try:
                        df = pd.read_csv(filepath)
                        dfs.append(df)
                    except Exception as e:
                        self.log(f"    ! Error reading {filepath}: {str(e)[:50]}")
            
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                output_file = os.path.join(output_dir, f"all_{filename}")
                combined.to_csv(output_file, index=False)
                self.log(f"  [OK] {filename}: {len(dfs)} players, {len(combined)} rows -> {output_file}")
        
        print(f"\n[OK] Combined {len(player_folders)} player folders")
    
    def list_players(self):
        """List all scraped players and their file counts."""
        player_folders = sorted([d for d in os.listdir(self.output_dir) 
                                if os.path.isdir(os.path.join(self.output_dir, d))])
        
        if not player_folders:
            print("No players scraped yet.")
            return
        
        print("\nScraped Running Backs:")
        print("-" * 50)
        for folder in player_folders:
            player_path = os.path.join(self.output_dir, folder)
            files = os.listdir(player_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            print(f"  {folder}: {len(csv_files)} tables")


def main():
    parser = argparse.ArgumentParser(description='NFL RB Stats Scraper')
    parser.add_argument('--player', help='Scrape specific player by ID (e.g., HenrDe00)')
    parser.add_argument('--test', action='store_true', help='Test scraper with first RB')
    parser.add_argument('--force', action='store_true', help='Force re-scrape even if folder exists')
    parser.add_argument('--combine', action='store_true', help='Combine all CSV files')
    parser.add_argument('--list', action='store_true', help='List all scraped players')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    scraper = NFLRBScraper(verbose=not args.quiet)
    
    print("\n" + "=" * 70)
    print("NFL RB STATS SCRAPER - MULTI-TABLE VERSION")
    print("Scraping %d players..." % len(NFL_RBS))
    print("Tables: Rushing/Receiving, Advanced Rush/Rec, Defense/Fumbles,")
    print("        Snap Counts")
    print("=" * 70)
    
    if args.player:
        # Find player by ID
        player = next((p for p in NFL_RBS if p[0] == args.player), None)
        if player:
            scraper.scrape_player(player[0], player[1], force=args.force)
        else:
            print(f"Player {args.player} not found in list")
    
    elif args.test:
        if NFL_RBS:
            player = NFL_RBS[0]
            scraper.scrape_player(player[0], player[1], force=True)
    
    elif args.list:
        scraper.list_players()
    
    elif args.combine:
        scraper.combine_all_csvs()
    
    else:
        scraper.scrape_all(force=args.force)


if __name__ == '__main__':
    main()
