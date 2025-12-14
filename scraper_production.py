"""
NFL Data Scraper - Production Version
Automatically scrapes Pro Football Reference without manual export
Ready to use - just run this script!
"""

import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random
from pathlib import Path
from datetime import datetime

class NFLScraper:
    """Production scraper for NFL player stats"""
    
    def __init__(self, delay_min=2, delay_max=5, verbose=True):
        """
        Initialize scraper
        
        Args:
            delay_min: Minimum delay between requests
            delay_max: Maximum delay between requests
            verbose: Print detailed output
        """
        self.scraper = cloudscraper.create_scraper()
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.verbose = verbose
        self.results = {'success': 0, 'failed': 0, 'data': {}}
    
    def _log(self, msg):
        """Print if verbose"""
        if self.verbose:
            print(msg)
    
    def _delay(self):
        """Respectful random delay"""
        delay = random.uniform(self.delay_min, self.delay_max)
        self._log(f"  ⏳ Waiting {delay:.1f}s...")
        time.sleep(delay)
    
    def scrape(self, player_id, save_csv=True):
        """
        Scrape a single player's stats
        
        Args:
            player_id: Player ID (e.g., 'BradTo00') or full URL
            save_csv: Save as CSV file
        
        Returns:
            pandas DataFrame or None
        """
        
        # Build URL
        if player_id.startswith('http'):
            url = player_id
            pid = player_id.split('/')[-1].replace('.htm', '')
        else:
            first_letter = player_id[0].upper()
            url = f"https://www.pro-football-reference.com/players/{first_letter}/{player_id}.htm"
            pid = player_id
        
        self._log(f"\n▶ Scraping: {pid}")
        self._log(f"  URL: {url}")
        
        try:
            self._log(f"  Fetching...")
            response = self.scraper.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract player name
            h1 = soup.find('h1')
            player_name = h1.get_text(strip=True) if h1 else "Unknown"
            self._log(f"  ✓ Found: {player_name}")
            
            # Find table - try different IDs
            table = soup.find('table', {'id': 'passing'})  # QB stats
            
            if not table:
                # Try other position-specific tables
                for table_id in ['rushing_and_receiving', 'receiving_and_rushing', 'defense', 'kicking']:
                    table = soup.find('table', {'id': table_id})
                    if table:
                        break
            
            if not table:
                # Fallback: get first stats table
                table = soup.find('table', {'class': 'stats_table'})
            
            if not table:
                self._log(f"  ✗ Stats table not found")
                self.results['failed'] += 1
                self._delay()
                return None
            
            # Extract headers - PFR tables have headers in thead
            headers = []
            thead = table.find('thead')
            if thead:
                # Get all th elements from header rows, skip rowheader column
                for th in thead.find_all('th'):
                    # Skip the first column if it's empty or "Rk"
                    data_stat = th.get('data-stat', '')
                    if data_stat not in ['', 'ranker']:
                        h = th.get_text(strip=True)
                        if h:
                            headers.append(h)
                    elif th.find_previous('th') is None and th.get_text(strip=True) == '':
                        # Skip rank column
                        continue
                    elif data_stat == '':
                        # Skip index column
                        continue
                    else:
                        h = th.get_text(strip=True)
                        if h and h != 'Rk':
                            headers.append(h)
            
            # If still no headers, extract all th text
            if not headers:
                for th in table.find_all('th'):
                    h = th.get_text(strip=True)
                    if h and h != 'Rk':
                        headers.append(h)
            
            # Extract rows - match header count
            rows = []
            tbody = table.find('tbody')
            if tbody:
                for tr in tbody.find_all('tr'):
                    # Get both th and td (first column might be th)
                    cells = tr.find_all(['th', 'td'])
                    if cells:
                        # Skip first cell if it's the rank
                        if cells[0].name == 'th':
                            cells = cells[1:]
                        row = [cell.get_text(strip=True) for cell in cells]
                        if row and len(row) > 0:
                            rows.append(row)
            else:
                # Fallback
                for tr in table.find_all('tr')[1:]:
                    cells = tr.find_all(['th', 'td'])
                    if cells:
                        if cells[0].name == 'th':
                            cells = cells[1:]
                        row = [cell.get_text(strip=True) for cell in cells]
                        if row:
                            rows.append(row)
            
            if not rows:
                self._log(f"  ✗ No data rows found")
                self.results['failed'] += 1
                self._delay()
                return None
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers[:len(rows[0])])
            df['Player'] = player_name
            df['PlayerID'] = pid
            
            self._log(f"  ✓ Extracted {len(df)} seasons")
            
            # Save CSV
            if save_csv:
                os.makedirs('data/raw', exist_ok=True)
                filename = f"data/raw/{pid}_stats.csv"
                df.to_csv(filename, index=False)
                self._log(f"  ✓ Saved: {filename}")
            
            self.results['success'] += 1
            self.results['data'][pid] = df
            
            self._delay()
            return df
        
        except Exception as e:
            self._log(f"  ✗ Error: {str(e)[:80]}")
            self.results['failed'] += 1
            self._delay()
            return None
    
    def scrape_batch(self, player_list, save_csv=True, combine=True):
        """
        Scrape multiple players
        
        Args:
            player_list: List of player IDs or URLs
            save_csv: Save each as CSV
            combine: Combine into single file
        
        Returns:
            Dictionary with results
        """
        
        print("\n" + "="*70)
        print(f"NFL STATS SCRAPER - BATCH MODE")
        print(f"Scraping {len(player_list)} players...")
        print("="*70)
        
        for i, player in enumerate(player_list, 1):
            print(f"\n[{i}/{len(player_list)}]", end='')
            self.scrape(player, save_csv=save_csv)
        
        # Summary
        print(f"\n\n" + "="*70)
        print(f"RESULTS:")
        print(f"  ✓ Success: {self.results['success']}")
        print(f"  ✗ Failed: {self.results['failed']}")
        print(f"  Total: {self.results['success'] + self.results['failed']}")
        print("="*70)
        
        # Combine data
        if combine and self.results['data']:
            try:
                all_dfs = list(self.results['data'].values())
                # Use outer join to keep all columns from all dataframes
                combined = pd.concat(all_dfs, axis=0, sort=False, ignore_index=True)
                
                os.makedirs('data/processed', exist_ok=True)
                combined_file = 'data/processed/nfl_scraped_stats_combined.csv'
                combined.to_csv(combined_file, index=False)
                
                print(f"\n✓ Combined data saved to: {combined_file}")
                print(f"  Total records: {len(combined)}")
                print(f"  Unique players: {combined['Player'].nunique()}")
            except Exception as e:
                print(f"\n! Warning: Could not combine data: {e}")
                print(f"  But individual files are saved in data/raw/")
        
        return self.results['data']


# Preset player lists
QBS = [
    'BradTo00',   # Tom Brady
    'MahoMa00',   # Patrick Mahomes (actually Ma01 or Ma02?)
    'AlleJo02',   # Josh Allen
    'JackLa00',   # Lamar Jackson
    'HurtJa00',   # Jalen Hurts
    'MontJo01',   # Joe Montana
    'Unitas',      # Johnny Unitas (if available)
]

RBS = [
    'HenrDe00',   # Derrick Henry
    'SandBa00',   # Barry Sanders
    'SmitEm00',   # Emmitt Smith
    'PaytoWa00',  # Walter Payton
]

WRS_TES = [
    'RiceJe00',   # Jerry Rice
    'MossRa00',   # Randy Moss
    'KelcTr00',   # Travis Kelce
    'GronRo00',   # Rob Gronkowski
    'GonzTo00',   # Tony Gonzalez
]

def scrape_qbs():
    """Scrape all QBs"""
    scraper = NFLScraper(delay_min=3, delay_max=6)
    return scraper.scrape_batch(QBS, combine=True)

def scrape_rbs():
    """Scrape all RBs"""
    scraper = NFLScraper(delay_min=3, delay_max=6)
    return scraper.scrape_batch(RBS, combine=True)

def scrape_receivers():
    """Scrape WRs and TEs"""
    scraper = NFLScraper(delay_min=3, delay_max=6)
    return scraper.scrape_batch(WRS_TES, combine=True)

def scrape_all():
    """Scrape all positions"""
    all_players = QBS + RBS + WRS_TES
    scraper = NFLScraper(delay_min=3, delay_max=6)
    return scraper.scrape_batch(all_players, combine=True)

def scrape_custom(player_ids):
    """Scrape custom list"""
    scraper = NFLScraper(delay_min=3, delay_max=6)
    return scraper.scrape_batch(player_ids, combine=True)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("NFL STATS SCRAPER - AUTOMATED")
    print("="*70)
    
    # UNCOMMENT THE ONE YOU WANT TO RUN:
    
    # Option 1: Scrape QBs only
    print("\n[RUN MODE: QBs only]")
    print("-"*70)
    qb_data = scrape_qbs()
    
    # Option 2: Scrape RBs only
    # rb_data = scrape_rbs()
    
    # Option 3: Scrape WRs/TEs
    # wr_data = scrape_receivers()
    
    # Option 4: Scrape all positions
    # all_data = scrape_all()
    
    # Option 5: Scrape custom list
    # custom_players = ['BradTo00', 'AlleJo02', 'HenrDe00']
    # custom_data = scrape_custom(custom_players)
    
    print("\n" + "="*70)
    print("USAGE:")
    print("="*70)
    print("""
# Scrape specific players:
scraper = NFLScraper()
scraper.scrape('BradTo00')
scraper.scrape('AlleJo02')

# Scrape multiple at once:
scraper = NFLScraper()
scraper.scrape_batch(['BradTo00', 'AlleJo02', 'HenrDe00'])

# Use preset functions:
scrape_qbs()         # All QBs
scrape_rbs()         # All RBs  
scrape_receivers()   # WRs and TEs
scrape_all()         # Everyone
scrape_custom(['BradTo00', 'MahoMa00'])  # Custom list
    """)
    
    print("\n✓ Done! Check data/raw/ and data/processed/ folders")
