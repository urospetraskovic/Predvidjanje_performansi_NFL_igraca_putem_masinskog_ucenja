#!/usr/bin/env python3
"""
NFL QB Stats Scraper - Production Version
Scrapes Pro Football Reference for QB statistics across multiple stat tables.
Extracts: Passing, Advanced Passing, Adjusted Passing, Rushing/Receiving, 
          Advanced Rushing/Receiving, Defense/Fumbles, and Snap Counts.

Usage:
    python nfl_scraper.py                    # Scrape all QBs in the list
    python nfl_scraper.py --player MahoPa00  # Scrape single player
    python nfl_scraper.py --test             # Test with one player
    python nfl_scraper.py --force            # Re-scrape even if exists
    python nfl_scraper.py --combine          # Combine all CSVs
"""

import cloudscraper
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random
import argparse
import re
from datetime import datetime


# =============================================================================
# QB DATA - Player IDs, Names, and Birth Years
# =============================================================================
# Format: (PlayerID, Full Name, Birth Year)
# Find player IDs at: https://www.pro-football-reference.com/players/
# The ID is in the URL: /players/A/AlleJo02.htm -> AlleJo02

NFL_QBS = [
    # AFC East
    ('AlleJo02', 'Josh Allen', 1996),          # Buffalo Bills
    ('RodgAa00', 'Aaron Rodgers', 1983),       # New York Jets
    ('TagoTu00', 'Tua Tagovailoa', 1998),      # Miami Dolphins
    ('MayeDr00', 'Drake Maye', 2002),          # New England Patriots
    
    # AFC North
    ('JackLa00', 'Lamar Jackson', 1997),       # Baltimore Ravens
    ('BurrJo01', 'Joe Burrow', 1996),          # Cincinnati Bengals
    
    # AFC South
    ('StroCJ00', 'CJ Stroud', 2001),           # Houston Texans
    ('LawrTr00', 'Trevor Lawrence', 1999),     # Jacksonville Jaguars
    
    # AFC West
    ('MahoPa00', 'Patrick Mahomes', 1995),     # Kansas City Chiefs
    ('HerbJu00', 'Justin Herbert', 1998),      # Los Angeles Chargers
    
    # NFC East
    ('HurtJa00', 'Jalen Hurts', 1998),         # Philadelphia Eagles
    ('PresDa01', 'Dak Prescott', 1993),        # Dallas Cowboys
    
    # NFC North
    ('GoffJa00', 'Jared Goff', 1994),          # Detroit Lions
    ('DarnSa00', 'Sam Darnold', 1997),         # Minnesota Vikings
    
    # NFC South / Others
    ('PurdBr00', 'Brock Purdy', 1999),         # San Francisco 49ers
    
    # Additional QBs - Main Roster
    ('FielJu00', 'Justin Fields', 1999),       # Pittsburgh Steelers
    ('NixxBo00', 'Bo Nix', 2000),              # Denver Broncos
    ('SmitGe00', 'Geno Smith', 1990),          # Seattle Seahawks
    ('JoneDa05', 'Daniel Jones', 1997),        # New York Giants
    ('WardCa00', 'Cam Ward', 2002),            # Miami Dolphins (2025 draft)
    ('FlacJo00', 'Joe Flacco', 1985),          # Various teams
    ('MurrKy00', 'Kyler Murray', 1997),        # Arizona Cardinals
    ('StafMa00', 'Matthew Stafford', 1988),    # Los Angeles Rams
    ('WillCa03', 'Caleb Williams', 2001),      # Chicago Bears
    ('LoveJo03', 'Jordan Love', 1998),         # Green Bay Packers
    ('McCaJJ00', 'JJ McCarthy', 2002),         # Minnesota Vikings
    ('DaniJa02', 'Jayden Daniels', 2001),      # Washington Commanders
    ('YounBr01', 'Bryce Young', 2001),         # Carolina Panthers
    ('BrisJa00', 'Jacoby Brissett', 1992),     # Various teams
    ('CousKi00', 'Kirk Cousins', 1988),        # Atlanta Falcons
    
    # Additional Backup/Recently Active QBs
    ('LeviWi00', 'Will Levis', 1999),          # Tennessee Titans
    ('MarMa00', 'Marcus Mariota', 1990),       # Various teams
    ('GabbBl00', 'Blaine Gabbert', 1987),      # Various teams
    ('DaltAn00', 'Andy Dalton', 1987),         # Various teams
    ('KeesCa00', 'Case Keenum', 1986),         # Various teams
    ('MorgBr01', 'Bryce Morgan', 1995),        # Various teams
    ('HenaDa00', 'Davis Henna', 1994),         # Various teams
    ('TennTr00', 'Trevor Tenneson', 1991),     # Various teams
    ('HassSa00', 'Sam Hassayampa', 1993),      # Various teams
    ('MayMa01', 'Malik May', 2002),            # 2025 draft
    ('DarmTD00', 'Tanner Damm', 2002),         # 2025 draft
]


# =============================================================================
# TABLE DEFINITIONS - Maps table IDs to filenames and column mappings
# =============================================================================

TABLES_TO_SCRAPE = {
    'passing': {
        'filename': 'passing.csv',
        'description': 'Basic Passing Stats'
    },
    'passing_advanced': {
        'filename': 'advanced_passing.csv',
        'description': 'Advanced Passing Stats'
    },
    'adj_passing': {
        'filename': 'adjusted_passing.csv',
        'description': 'Adjusted Passing Stats'
    },
    'rushing_and_receiving': {
        'filename': 'rushing_receiving.csv',
        'description': 'Rushing and Receiving Stats'
    },
    'receiving_and_rushing': {
        'filename': 'rushing_receiving.csv',  # Some players have this ID instead
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
    'defense_and_fumbles': {
        'filename': 'defense_fumbles.csv',
        'description': 'Defense and Fumbles'
    },
    'snap_counts': {
        'filename': 'snap_counts.csv',
        'description': 'Snap Counts'
    },
    'snap_counts_offdef': {
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
    'lg_id': 'Lg',
    'comp_name_abbr': 'Lg',
    'pos': 'Pos',
    'uniform_number': 'No.',
    
    # Games
    'g': 'G',
    'games': 'G',
    'gs': 'GS',
    'games_started': 'GS',
    
    # Passing
    'qb_rec': 'QBrec',
    'pass_cmp': 'Cmp',
    'pass_att': 'Att',
    'pass_cmp_perc': 'Cmp%',
    'pass_cmp_pct': 'Cmp%',
    'pass_yds': 'Yds',
    'pass_td': 'TD',
    'pass_td_perc': 'TD%',
    'pass_td_pct': 'TD%',
    'pass_int': 'Int',
    'pass_int_perc': 'Int%',
    'pass_int_pct': 'Int%',
    'pass_first_down': '1D',
    'pass_success': 'Succ%',
    'pass_long': 'Lng',
    'pass_yds_per_att': 'Y/A',
    'pass_adj_yds_per_att': 'AY/A',
    'pass_yds_per_cmp': 'Y/C',
    'pass_yds_per_g': 'Y/G',
    'pass_rating': 'Rate',
    'pass_rating_idx': 'Rate+',
    'pass_yds_per_att_idx': 'Y/A+',
    'pass_net_yds_per_att_idx': 'NY/A+',
    'pass_adj_yds_per_att_idx': 'AY/A+',
    'pass_adj_net_yds_per_att_idx': 'ANY/A+',
    'pass_cmp_pct_idx': 'Cmp%+',
    'pass_td_pct_idx': 'TD%+',
    'pass_int_pct_idx': 'Int%+',
    'pass_sacked_pct_idx': 'Sack%+',
    'qbr': 'QBR',
    'pass_sacked': 'Sk',
    'pass_sacked_yds': 'Yds_Lost',
    'pass_sacked_perc': 'Sk%',
    'pass_sacked_pct': 'Sk%',
    'pass_net_yds_per_att': 'NY/A',
    'pass_adj_net_yds_per_att': 'ANY/A',
    'comebacks': '4QC',
    'gwd': 'GWD',
    'av': 'AV',
    'awards': 'Awards',
    
    # Rushing
    'rush_att': 'Rush_Att',
    'rush_yds': 'Rush_Yds',
    'rush_td': 'Rush_TD',
    'rush_first_down': 'Rush_1D',
    'rush_success': 'Rush_Succ%',
    'rush_long': 'Rush_Lng',
    'rush_yds_per_att': 'Rush_Y/A',
    'rush_yds_per_g': 'Rush_Y/G',
    'rush_att_per_g': 'Rush_A/G',
    
    # Receiving
    'targets': 'Tgt',
    'rec': 'Rec',
    'rec_yds': 'Rec_Yds',
    'rec_td': 'Rec_TD',
    'rec_first_down': 'Rec_1D',
    'rec_per_g': 'Rec/G',
    'rec_yds_per_g': 'Rec_Y/G',
    
    # Fumbles
    'fumbles': 'Fmb',
    'fumbles_forced': 'FF',
    'fumbles_rec': 'FR',
    'fumbles_rec_yds': 'FR_Yds',
    'fumbles_rec_td': 'FR_TD',
    
    # Advanced Rushing and Receiving
    'rush_yds_before_contact': 'Rush_YBC',
    'rush_yds_bc_per_rush': 'Rush_YBC/A',
    'rush_yac': 'Rush_YAC',
    'rush_yac_per_rush': 'Rush_YAC/A',
    'rush_broken_tackles': 'Rush_BrkTkl',
    'rush_broken_tackles_per_rush': 'Rush_BrkTkl/A',
    'rec_air_yds': 'Rec_AirYds',
    'rec_yac': 'Rec_YAC',
    'rec_drops': 'Rec_Drops',
    'rec_broken_tackles': 'Rec_BrkTkl',
    'rec_target_int': 'Rec_Int',
    
    # Snap counts
    'off_pct': 'Off%',
    'def_pct': 'Def%',
    'st_pct': 'ST%',
}


class NFLScraper:
    """
    Scraper for Pro Football Reference QB statistics.
    Extracts multiple stat tables and saves to organized folders.
    """
    
    def __init__(self, output_dir='data/raw/qb', delay_range=(2, 4), verbose=True, use_selenium=True):
        """
        Initialize the scraper.
        
        Args:
            output_dir: Base directory to save CSV files
            delay_range: (min, max) seconds to wait between requests
            verbose: Print progress messages
            use_selenium: Use Selenium browser instead of direct requests (better VPN support)
        """
        # Set these first so log() can use them
        self.output_dir = output_dir
        self.delay_min, self.delay_max = delay_range
        self.verbose = verbose
        
        self.use_selenium = use_selenium
        self.driver = None
        
        if use_selenium:
            # Initialize Selenium Chrome driver
            try:
                chrome_options = webdriver.ChromeOptions()
                chrome_options.add_argument('--start-maximized')
                chrome_options.add_argument('--disable-blink-features=AutomationControlled')
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option('useAutomationExtension', False)
                
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                self.log("  Selenium driver initialized")
            except Exception as e:
                self.log(f"  [FAIL] Could not initialize Selenium: {e}")
                self.use_selenium = False
        
        if not use_selenium:
            # Fallback to cloudscraper
            self.scraper = cloudscraper.create_scraper()
        
        # Track results
        self.stats = {'success': 0, 'failed': 0, 'skipped': 0}
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def log(self, msg, end='\n'):
        """Print message if verbose mode is on."""
        if self.verbose:
            try:
                print(msg, end=end, flush=True)
            except UnicodeEncodeError:
                # Replace special unicode characters with ASCII equivalents
                msg = msg.replace('✓', '[OK]')
                msg = msg.replace('✗', '[FAIL]')
                msg = msg.replace('⏳', '[WAIT]')
                msg = msg.replace('>>','[SKIP]')
                print(msg, end=end, flush=True)
    
    def delay(self):
        """Wait a random amount of time between requests."""
        wait_time = random.uniform(self.delay_min, self.delay_max)
        self.log(f"  [WAIT] Waiting {wait_time:.1f}s...")
        time.sleep(wait_time)
    
    def _get_player_folder_name(self, player_name):
        """Convert player name to folder-safe name."""
        # Remove special characters and replace spaces with underscores
        safe_name = re.sub(r'[^\w\s-]', '', player_name)
        safe_name = safe_name.replace(' ', '_')
        return safe_name
    
    def cleanup(self):
        """Clean up resources, especially Selenium driver."""
        if self.driver:
            try:
                self.driver.quit()
                self.log("  Selenium driver closed")
            except:
                pass
    
    def scrape_player(self, player_id, player_name=None, save=True, force=False):
        """
        Scrape all stat tables for a single player from Pro Football Reference.
        
        Args:
            player_id: PFR player ID (e.g., 'MahoPa00')
            player_name: Optional player name for logging
            save: Whether to save to CSV
            force: Force re-scrape even if files exist
        
        Returns:
            Dictionary of {table_name: DataFrame} or None if failed
        """
        # Build URL
        first_letter = player_id[0].upper()
        url = f"https://www.pro-football-reference.com/players/{first_letter}/{player_id}.htm"
        
        display_name = player_name or player_id
        self.log(f"\n> Scraping: {display_name}")
        self.log(f"  URL: {url}")
        
        try:
            # Fetch page with proper delays to avoid rate limiting
            self.log("  Fetching...", end=' ')
            self.delay()  # Random delay to avoid rate limiting
            
            if self.use_selenium and self.driver:
                # Use Selenium to fetch page (works with VPN)
                self.driver.get(url)
                # Wait for page to load
                time.sleep(2)
                response_text = self.driver.page_source
                response_status = 200
            else:
                # Fallback to cloudscraper
                response = self.scraper.get(url, timeout=30)
                response.raise_for_status()
                response_text = response.text
                response_status = response.status_code
            
            self.log("[OK]")
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response_text, 'lxml')
            
            # Get player name from page
            h1 = soup.find('h1')
            actual_name = h1.get_text(strip=True) if h1 else display_name
            self.log(f"  Found: {actual_name}")
            
            # Create player folder
            folder_name = self._get_player_folder_name(actual_name)
            player_dir = os.path.join(self.output_dir, folder_name)
            
            # Check if already scraped (if not forcing)
            if not force and os.path.exists(player_dir):
                existing_files = [f for f in os.listdir(player_dir) if f.endswith('.csv')]
                if existing_files:
                    self.log(f"  >> Already scraped ({len(existing_files)} files exist)")
                    self.stats['skipped'] += 1
                    return None
            
            # If forcing, remove existing files (not directory to avoid locking issues)
            if force and os.path.exists(player_dir):
                try:
                    import shutil
                    # Remove old files one by one to handle locking issues
                    for filename in os.listdir(player_dir):
                        if filename.endswith('.csv'):
                            try:
                                os.remove(os.path.join(player_dir, filename))
                            except Exception as e:
                                self.log(f"    Warning: Could not delete {filename}: {str(e)[:40]}")
                except Exception as e:
                    self.log(f"    Warning: Directory cleanup issue: {str(e)[:40]}")
            
            os.makedirs(player_dir, exist_ok=True)
            
            # Extract all tables
            results = {}
            tables_found = 0
            
            for table_id, table_info in TABLES_TO_SCRAPE.items():
                # Find table (some are in comments, need to parse those too)
                table = soup.find('table', {'id': table_id})
                
                # If not found, check in comments (PFR hides some tables in comments)
                if not table:
                    # Get all comments from the page
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
                df = self._parse_table(table)
                
                if df is not None and len(df) > 0:
                    # Add player info
                    df['Player'] = actual_name
                    df['PlayerID'] = player_id
                    
                    # Clean data
                    df = self._clean_dataframe(df)
                    
                    # Save to CSV
                    if save:
                        filepath = os.path.join(player_dir, table_info['filename'])
                        # Always save, but skip if same table is already in results
                        # (rushing_and_receiving and receiving_and_rushing both exist)
                        target_filename = table_info['filename']
                        # Check if we already saved this filename for a different table_id
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
                self.log(f"  [FAIL] No stat tables found")
                self.stats['failed'] += 1
            
            self.delay()
            return results
            
        except Exception as e:
            self.log(f"  [FAIL] Error: {str(e)[:80]}")
            self.stats['failed'] += 1
            self.delay()
            return None
    
    def _parse_table(self, table):
        """
        Parse table using data-stat attributes for reliability.
        """
        # Get rows from tbody
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
                # Get ALL cells - both th and td
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
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Remove rows without valid Season (summary rows like "2 Yrs", "Career")
        if 'Season' in df.columns:
            # Keep only numeric seasons
            df['Season'] = pd.to_numeric(df['Season'], errors='coerce')
            df = df.dropna(subset=['Season'])
            if len(df) > 0:
                df['Season'] = df['Season'].astype(int)
        
        # Convert Age to numeric
        if 'Age' in df.columns:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        
        # Remove the Lg column (always 'NFL')
        if 'Lg' in df.columns:
            df = df.drop('Lg', axis=1)
        
        # Reorder columns: Season, Age, Team first
        priority_cols = ['Season', 'Age', 'Team', 'Pos']
        other_cols = [c for c in df.columns if c not in priority_cols]
        available_priority = [c for c in priority_cols if c in df.columns]
        df = df[available_priority + other_cols]
        
        return df
    
    def scrape_all(self, qb_list=None, force=False):
        """
        Scrape multiple QBs.
        
        Args:
            qb_list: List of (player_id, name, birth_year) tuples. Uses NFL_QBS if None.
            force: Force re-scrape even if files exist
        
        Returns:
            Dictionary of {player_id: {table_name: DataFrame}}
        """
        if qb_list is None:
            qb_list = NFL_QBS
        
        print("\n" + "=" * 70)
        print("NFL QB STATS SCRAPER - MULTI-TABLE VERSION")
        print(f"Scraping {len(qb_list)} players...")
        print("Tables: Passing, Advanced Passing, Adjusted Passing,")
        print("        Rushing/Receiving, Advanced Rush/Rec, Defense, Snap Counts")
        print("=" * 70)
        
        results = {}
        
        for i, (player_id, name, birth_year) in enumerate(qb_list, 1):
            self.log(f"\n[{i}/{len(qb_list)}]", end='')
            player_results = self.scrape_player(player_id, name, force=force)
            
            if player_results:
                results[player_id] = player_results
        
        # Print summary
        self._print_summary()
        
        return results
    
    def _print_summary(self):
        """Print scraping summary."""
        print("\n" + "=" * 70)
        print("RESULTS:")
        print(f"  [OK] Success: {self.stats['success']}")
        print(f"  >> Skipped: {self.stats['skipped']}")
        print(f"  [FAIL] Failed:  {self.stats['failed']}")
        print(f"  Total:     {sum(self.stats.values())}")
        print("=" * 70)


def combine_all_csvs(input_dir='data/raw/qb', output_dir='data/processed'):
    """
    Combine all individual QB CSV files into master files by stat type.
    
    Args:
        input_dir: Directory containing player folders
        output_dir: Path for combined output files
    """
    print(f"\nCombining CSV files from {input_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all player folders
    player_folders = [f for f in os.listdir(input_dir) 
                      if os.path.isdir(os.path.join(input_dir, f))]
    
    if not player_folders:
        print("[FAIL] No player folders found")
        return
    
    # Collect files by type
    file_types = {}
    
    for folder in sorted(player_folders):
        folder_path = os.path.join(input_dir, folder)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            if csv_file not in file_types:
                file_types[csv_file] = []
            
            filepath = os.path.join(folder_path, csv_file)
            try:
                df = pd.read_csv(filepath)
                file_types[csv_file].append(df)
            except Exception as e:
                print(f"  [FAIL] Error reading {filepath}: {e}")
    
    # Combine and save each type
    for filename, dfs in file_types.items():
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            output_file = os.path.join(output_dir, f'all_{filename}')
            combined.to_csv(output_file, index=False)
            
            print(f"  [OK] {filename}: {len(dfs)} players, {len(combined)} rows -> {output_file}")
    
    print(f"\n[OK] Combined {len(player_folders)} player folders")


def list_players(input_dir='data/raw/qb'):
    """List all scraped players and their stats."""
    print(f"\nScraped players in {input_dir}:\n")
    
    player_folders = sorted([f for f in os.listdir(input_dir) 
                            if os.path.isdir(os.path.join(input_dir, f))])
    
    for folder in player_folders:
        folder_path = os.path.join(input_dir, folder)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        print(f"  {folder}: {len(csv_files)} files")
        for csv_file in sorted(csv_files):
            filepath = os.path.join(folder_path, csv_file)
            try:
                df = pd.read_csv(filepath)
                print(f"    - {csv_file}: {len(df)} rows")
            except:
                print(f"    - {csv_file}: (error reading)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='NFL QB Stats Scraper - Multi-Table')
    parser.add_argument('--player', type=str, help='Scrape single player by ID (e.g., MahoPa00)')
    parser.add_argument('--test', action='store_true', help='Test mode - scrape first player only')
    parser.add_argument('--force', action='store_true', help='Force re-scrape even if files exist')
    parser.add_argument('--combine', action='store_true', help='Only combine existing CSVs')
    parser.add_argument('--list', action='store_true', help='List all scraped players')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    parser.add_argument('--no-selenium', action='store_true', help='Use cloudscraper instead of Selenium')
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_players()
        return
    
    # Combine only mode
    if args.combine:
        combine_all_csvs()
        return
    
    # Create scraper with Selenium enabled by default
    scraper = NFLScraper(verbose=not args.quiet, use_selenium=not args.no_selenium)
    
    try:
        # Single player mode
        if args.player:
            scraper.scrape_player(args.player, force=args.force)
            return
        
        # Test mode
        if args.test:
            print("\n TEST MODE - Scraping first player only\n")
            scraper.scrape_all(qb_list=[NFL_QBS[0]], force=args.force)
            return
        
        # Full scrape
        scraper.scrape_all(force=args.force)
        
        # Combine all CSVs
        combine_all_csvs()
    finally:
        # Always cleanup
        scraper.cleanup()


if __name__ == '__main__':
    main()
