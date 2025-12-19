#!/usr/bin/env python3
"""
NFL TE Stats Scraper - Production Version
Scrapes Pro Football Reference for TE statistics across multiple stat tables.
Extracts: Receiving & Rushing, Advanced Receiving & Rushing, and Snap Counts.

Usage:
    python nfl_te_scraper.py                    # Scrape all TEs in the list
    python nfl_te_scraper.py --player KellTr00  # Scrape single player
    python nfl_te_scraper.py --test             # Test with one player
    python nfl_te_scraper.py --force            # Re-scrape even if exists
    python nfl_te_scraper.py --combine          # Combine all CSVs
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
# TE DATA - Player IDs and Names
# =============================================================================
# Format: (PlayerID, Full Name)
# Find player IDs at: https://www.pro-football-reference.com/players/

NFL_TES = [
    # Starting Tight Ends in the NFL - 2024-2025 Season
    # Verified IDs from Pro Football Reference
    # AFC EAST
    ('KincDa00', 'Dalton Kincaid'),       # Buffalo Bills
    ('SmitJo01', 'Jonnu Smith'),          # Miami Dolphins
    ('HenrHu00', 'Hunter Henry'),         # New England Patriots
    ('UzomC.00', 'C.J. Uzomah'),          # New York Jets
    
    # AFC NORTH
    ('AndrMa00', 'Mark Andrews'),         # Baltimore Ravens
    ('FreiPa00', 'Pat Freiermuth'),       # Pittsburgh Steelers
    ('HursHa00', 'Hayden Hurst'),         # Cincinnati Bengals
    ('NjokDa00', 'David Njoku'),          # Cleveland Browns
    
    # AFC SOUTH
    ('OkonCh00', 'Chigbo Okoye'),         # Indianapolis Colts
    ('EngrEv00', 'Evan Engram'),          # Jacksonville Jaguars
    ('BreJo00', 'Brevin Jordan'),         # Houston Texans
    ('ThomTe00', 'Tennessee TEs'),        # Tennessee Titans
    
    # AFC WEST
    ('KelcTr00', 'Travis Kelce'),         # Kansas City Chiefs
    ('DissWi00', 'Will Dissly'),          # Los Angeles Chargers
    ('BoweBr01', 'Foster Moreau'),        # Las Vegas Raiders
    ('DulcGr00', 'Greg Dulcich'),         # Denver Broncos
    
    # NFC EAST
    ('SchuDa00', 'Dalton Schultz'),       # Dallas Cowboys
    ('GoedDa00', 'Dallas Goedert'),       # Philadelphia Eagles
    ('ErtzZa00', 'Zach Ertz'),            # Washington Commanders
    ('ManeCh00', 'Chris Manhertz'),       # New York Giants
    
    # NFC NORTH
    ('KmetCo00', 'Cole Kmet'),            # Chicago Bears
    ('LaPoSa01', 'Sam LaPorta'),          # Detroit Lions
    ('HockTJ00', 'T.J. Hockenson'),       # Minnesota Vikings
    ('KrafTu00', 'Tucker Kraft'),         # Green Bay Packers
    
    # NFC SOUTH
    ('PittKy00', 'Kyle Pitts'),           # Atlanta Falcons
    ('OttoCa00', 'Cade Otton'),           # Tampa Bay Buccaneers
    ('ThomIa00', 'Ian Thomas'),           # Carolina Panthers
    ('JalyJa00', 'Jalin Hyatt'),          # New Orleans Saints
    
    # NFC WEST
    ('KittGe00', 'George Kittle'),        # San Francisco 49ers
    ('HigbTy00', 'Tyler Higbee'),         # Los Angeles Rams
    ('WallDa01', 'Darren Waller'),        # Arizona Cardinals
    ('ParcEv00', 'Evan Anderson'),        # Seattle Seahawks
]


# =============================================================================
# TABLE DEFINITIONS - Maps table IDs to filenames
# =============================================================================

TABLES_TO_SCRAPE = {
    'receiving_and_rushing': {
        'filename': 'receiving_rushing.csv',
        'description': 'Receiving and Rushing Stats'
    },
    'adv_receiving_and_rushing': {
        'filename': 'advanced_receiving_rushing.csv',
        'description': 'Advanced Receiving and Rushing Stats'
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
    
    # Scrimmage
    'touches': 'Touch',
    'yds_touch': 'Y/Tch',
    'rush_rec_td': 'RRTD',
    'scrimmage_yds': 'YScm',
    
    # Advanced Receiving and Rushing
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
    
    'rush_yds_before_contact': 'YBC',
    'rush_yds_bc_per_rush': 'YBC/Att',
    'rush_yac': 'YAC',
    'rush_yac_per_rush': 'YAC/Att',
    'rush_broken_tackles': 'BrkTkl',
    'rush_broken_tackles_per_rush': 'Att/Br',
    
    # Snap counts
    'off_pct': 'Off%',
    'def_pct': 'Def%',
    'st_pct': 'ST%',
    'snap_counts_offdef': 'Num',
    'snap_counts': 'Num',
}


class NFLTEScraper:
    """
    Scraper for Pro Football Reference TE statistics.
    Extracts multiple stat tables and saves to organized folders.
    """
    
    def __init__(self, output_dir='data/raw/te', verbose=True, delay_min=180, delay_max=300):
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
        """Add random delay between requests to avoid rate limiting."""
        delay_time = random.uniform(self.delay_min, self.delay_max)
        self.log(f"\n  [Delaying {delay_time:.0f}s to avoid rate limiting...]", end='')
        time.sleep(delay_time)
        self.log(" [OK]")
    
    def _get_player_folder_name(self, player_name):
        """Convert player name to folder-safe name."""
        safe_name = re.sub(r'[^\w\s-]', '', player_name)
        safe_name = safe_name.replace(' ', '_')
        return safe_name
    
    def scrape_player(self, player_id, player_name=None, save=True, force=False):
        """
        Scrape all stat tables for a single TE.
        
        Args:
            player_id: PFR player ID (e.g., 'KellTr00')
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
            max_retries = 15
            for attempt in range(max_retries):
                try:
                    response = self.scraper.get(url, timeout=15)
                    response.raise_for_status()
                    break
                except Exception as e:
                    if '429' in str(e) and attempt < max_retries - 1:
                        wait_time = 120 * (attempt + 1)  # 120, 240, 360, etc seconds (exponential backoff)
                        self.log(f"(rate limited, waiting {wait_time}s, retry {attempt+1})", end=' ')
                        time.sleep(wait_time)
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
        """Scrape all TEs in the list."""
        total = len(NFL_TES)
        for i, (player_id, name) in enumerate(NFL_TES, 1):
            self.log(f"\n[{i}/{total}]")
            self.scrape_player(player_id, name, force=force)
            # Add delay between players to avoid rate limiting
            if i < total:
                self.delay()
        
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
        self.log("\nCombining CSV files from data/raw/te...")
        
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
                output_file = os.path.join(output_dir, f"all_te_{filename}")
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
        
        print("\nScraped Tight Ends:")
        print("-" * 50)
        for folder in player_folders:
            player_path = os.path.join(self.output_dir, folder)
            files = os.listdir(player_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            print(f"  {folder}: {len(csv_files)} tables")


def main():
    parser = argparse.ArgumentParser(description='NFL TE Stats Scraper')
    parser.add_argument('--player', help='Scrape specific player by ID (e.g., KellTr00)')
    parser.add_argument('--test', action='store_true', help='Test scraper with first TE')
    parser.add_argument('--force', action='store_true', help='Force re-scrape even if folder exists')
    parser.add_argument('--combine', action='store_true', help='Combine all CSV files')
    parser.add_argument('--list', action='store_true', help='List all scraped players')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    scraper = NFLTEScraper(verbose=not args.quiet)
    
    print("\n" + "=" * 70)
    print("NFL TE STATS SCRAPER - MULTI-TABLE VERSION")
    print("Scraping %d players..." % len(NFL_TES))
    print("Tables: Receiving/Rushing, Advanced Receiving/Rushing,")
    print("        Snap Counts")
    print("=" * 70)
    
    if args.player:
        # Find player by ID
        player = next((p for p in NFL_TES if p[0] == args.player), None)
        if player:
            scraper.scrape_player(player[0], player[1], force=args.force)
        else:
            print(f"Player {args.player} not found in list")
    
    elif args.test:
        if NFL_TES:
            player = NFL_TES[0]
            scraper.scrape_player(player[0], player[1], force=True)
    
    elif args.list:
        scraper.list_players()
    
    elif args.combine:
        scraper.combine_all_csvs()
    
    else:
        scraper.scrape_all(force=args.force)


if __name__ == '__main__':
    main()
