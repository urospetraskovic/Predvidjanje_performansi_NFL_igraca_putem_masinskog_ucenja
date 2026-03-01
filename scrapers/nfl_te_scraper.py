from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random
import argparse
import re

# "C:\Users\Win10\AppData\Local\Programs\Opera GX\opera.exe";
OPERA_BINARY = os.path.expanduser(r"~\AppData\Local\Programs\Opera GX\opera.exe")
CHROMEDRIVER_PATH = os.path.expanduser(r"~\.cache\selenium\chromedriver\win64\143.0.7499.42\chromedriver.exe")

NFL_TES = [
    ('KincDa00', 'Dalton Kincaid'),       
    ('SmitJo01', 'Jonnu Smith'),         
    ('HenrHu00', 'Hunter Henry'),         
    ('UzomC.00', 'C.J. Uzomah'),          
    ('AndrMa00', 'Mark Andrews'),         
    ('FreiPa00', 'Pat Freiermuth'),       
    ('HursHa00', 'Hayden Hurst'),         
    ('NjokDa00', 'David Njoku'),          
    ('OkonCh00', 'Chigoziem Okonkwo'),    
    ('EngrEv00', 'Evan Engram'),          
    ('SchuDa00', 'Dalton Schultz'),       
    ('KelcTr00', 'Travis Kelce'),         
    ('DissWi00', 'Will Dissly'),          
    ('MoreFo00', 'Foster Moreau'),        
    ('DulcGr00', 'Greg Dulcich'),         
    ('GoedDa00', 'Dallas Goedert'),
    ('ErtzZa00', 'Zach Ertz'),
    ('KmetCo00', 'Cole Kmet'),            
    ('LaPoSa01', 'Sam LaPorta'),          
    ('HockTJ00', 'T.J. Hockenson'),      
    ('KrafTu00', 'Tucker Kraft'),         
    ('PittKy00', 'Kyle Pitts'),           
    ('OttoCa00', 'Cade Otton'),           
    ('ThomIa00', 'Ian Thomas'),           
    ('KittGe00', 'George Kittle'),        
    ('HigbTy00', 'Tyler Higbee'),         
    ('WallDa01', 'Darren Waller'),        
    ('BoweBr00', 'Brock Bowers'),         
    ('GonzTo00', 'Tony Gonzalez'),
    ('GateAn00', 'Antonio Gates'),
    ('SharSh00', 'Shannon Sharpe'),
    ('GronRo00', 'Rob Gronkowski'),
    ('OlseGr00', 'Greg Olsen'),
    ('WittJa00', 'Jason Witten'),
    ('GrahJi00', 'Jimmy Graham'),
    ('CookJa02', 'Jared Cook'),
    ('WatsBe00', 'Ben Watson'),
    ('LewiMa00', 'Marcedes Lewis'),
]

TABLES_TO_SCRAPE = {
    'receiving_and_rushing': {
        'filename': 'receiving_rushing.csv',
        'description': 'Receiving and Rushing Stats'
    },
    'rushing_and_receiving': {
        'filename': 'receiving_rushing.csv',
        'description': 'Receiving and Rushing Stats (alt ID)'
    },
    'adv_receiving_and_rushing': {
        'filename': 'advanced_receiving_rushing.csv',
        'description': 'Advanced Receiving and Rushing Stats'
    },
    'adv_rushing_and_receiving': {
        'filename': 'advanced_receiving_rushing.csv',
        'description': 'Advanced Receiving and Rushing Stats (alt ID)'
    },
    'snap_counts': {
        'filename': 'snap_counts.csv',
        'description': 'Snap Counts'
    },
    'snap_counts_offdef': {
        'filename': 'snap_counts.csv',
        'description': 'Snap Counts (alt ID)'
    },
}

# column name mappings for cleaner output
COLUMN_MAPPING = {
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
    # games
    'g': 'G',
    'games': 'G',
    'gs': 'GS',
    'games_started': 'GS',
    # receiving
    'targets': 'Tgt',
    'rec': 'Rec',
    'rec_yds': 'Rec_Yds',
    'rec_yds_per_rec': 'Rec_Y/R',
    'rec_td': 'Rec_TD',
    'rec_first_down': 'Rec_1D',
    'rec_per_g': 'Rec/G',
    'rec_yds_per_g': 'Rec_Y/G',
    'rec_catch_pct': 'Catch%',
    'catch_pct': 'Catch%',
    'rec_yds_per_tgt': 'Rec_Y/Tgt',
    'rec_long': 'Rec_Lng',
    'rec_success': 'Rec_Succ%',
    # rushing
    'rush_att': 'Rush_Att',
    'rush_yds': 'Rush_Yds',
    'rush_td': 'Rush_TD',
    'rush_first_down': 'Rush_1D',
    'rush_success': 'Rush_Succ%',
    'rush_long': 'Rush_Lng',
    'rush_yds_per_att': 'Rush_Y/A',
    'rush_yds_per_g': 'Rush_Y/G',
    'rush_att_per_g': 'Rush_A/G',
    # scrimmage / combined
    'touches': 'Touches',
    'yds_touch': 'Y/Touch',
    'yds_per_touch': 'Y/Touch',
    'rush_receive_td': 'Rush_Rec_TD',
    'rush_rec_td': 'Rush_Rec_TD',
    'yds_from_scrimmage': 'Scrimmage_Yds',
    'scrimmage_yds': 'Scrimmage_Yds',
    # advanced Receiving
    'rec_air_yds': 'Rec_AirYds',
    'rec_air_yds_per_rec': 'Rec_AirYds/R',
    'rec_yac': 'Rec_YAC',
    'rec_yac_per_rec': 'Rec_YAC/R',
    'rec_adot': 'Rec_aDOT',
    'rec_broken_tackles': 'Rec_BrkTkl',
    'rec_broken_tackles_per_rec': 'Rec_BrkTkl/R',
    'rec_drops': 'Rec_Drops',
    'rec_drop_pct': 'Rec_Drop%',
    'rec_target_int': 'Rec_Int',
    'rec_pass_rating': 'Rec_PassRtg',
    # advanced Rushing
    'rush_yds_before_contact': 'Rush_YBC',
    'rush_yds_bc_per_rush': 'Rush_YBC/A',
    'rush_yac': 'Rush_YAC',
    'rush_yac_per_rush': 'Rush_YAC/A',
    'rush_broken_tackles': 'Rush_BrkTkl',
    'rush_broken_tackles_per_rush': 'Rush_BrkTkl/A',
    # snap counts
    'offense': 'Off_Snaps',
    'off_pct': 'Off%',
    'defense': 'Def_Snaps',
    'def_pct': 'Def%',
    'special_teams': 'ST_Snaps',
    'st_pct': 'ST%',
    # other
    'av': 'AV',
    'awards': 'Awards',
}

OUTPUT_DIR = 'data/raw/te'

def get_player_folder_name(player_name):
    """Convert player name to folder-safe name."""
    safe_name = re.sub(r'[^\w\s-]', '', player_name)
    safe_name = safe_name.replace(' ', '_')
    return safe_name


def parse_table(table):
    """Parse an HTML table using data-stat attributes."""
    rows = []
    tbody = table.find('tbody')
    if tbody:
        for tr in tbody.find_all('tr'):
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
    df = pd.DataFrame(rows)
    df = df.rename(columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns})
    return df


def clean_dataframe(df):
    """Clean and validate the dataframe."""
    df = df.copy()
    if 'Season' in df.columns:
        df['Season'] = pd.to_numeric(df['Season'], errors='coerce')
        df = df.dropna(subset=['Season'])
        if len(df) > 0:
            df['Season'] = df['Season'].astype(int)
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    if 'Lg' in df.columns:
        df = df.drop('Lg', axis=1)
    priority_cols = ['Season', 'Age', 'Team', 'Pos']
    other_cols = [c for c in df.columns if c not in priority_cols]
    available_priority = [c for c in priority_cols if c in df.columns]
    df = df[available_priority + other_cols]
    return df


def create_driver():
    """Create and return an Opera GX Selenium driver."""
    options = webdriver.ChromeOptions()
    options.binary_location = OPERA_BINARY

    # use a unique temp profile each time so we don't conflict with running Opera
    import tempfile
    temp_profile = tempfile.mkdtemp(prefix='opera_scraper_')
    options.add_argument(f"user-data-dir={temp_profile}")

    # anti-detection
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--no-first-run')
    options.add_argument('--no-default-browser-check')

    print(f"Using ChromeDriver: {CHROMEDRIVER_PATH}")
    service = Service(executable_path=CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    print("Opera GX browser launched successfully!")
    return driver


def scrape_player(driver, player_id, player_name, force=False):
    """Scrape all stat tables for a single TE player."""
    first_letter = player_id[0].upper()
    url = f"https://www.pro-football-reference.com/players/{first_letter}/{player_id}.htm"

    print(f"\n> Scraping: {player_name} ({player_id})")
    print(f"  URL: {url}")

    # check if already scraped (with all expected tables)
    folder_name = get_player_folder_name(player_name)
    player_dir = os.path.join(OUTPUT_DIR, folder_name)
    if not force and os.path.exists(player_dir):
        existing_files = [f for f in os.listdir(player_dir) if f.endswith('.csv')]
        # need at least receiving_rushing (the universal TE table)
        has_rr = 'receiving_rushing.csv' in existing_files
        #if has_rr and len(existing_files) >= 2:
        #    print(f"  >> Already scraped ({len(existing_files)} files exist) - SKIPPING")
        #    return 'skipped'
        #elif existing_files:
        #    print(f"  >> Incomplete ({len(existing_files)} files) - RE-SCRAPING")

    # longer delay between requests to avoid PFR rate-limiting
    wait_time = random.uniform(8, 12)
    print(f"  [WAIT] Waiting {wait_time:.1f}s...")
    time.sleep(wait_time)

    try:
        # try up to 3 attempts to load the page with tables
        page_source = None
        for attempt in range(3):
            if attempt > 0:
                extra_wait = random.uniform(20, 35)
                print(f"  [RETRY {attempt+1}/3] Waiting {extra_wait:.1f}s before retry...")
                time.sleep(extra_wait)

            driver.get(url)
            time.sleep(6)

            # opera GX sometimes stays on start page - retry navigation
            if "pro-football-reference" not in driver.current_url:
                print(f"  Redirected to {driver.current_url[:60]}, retrying...")
                time.sleep(3)
                driver.get(url)
                time.sleep(8)

            # check for Cloudflare
            page_source = driver.page_source
            if 'Just a moment' in page_source or 'Checking your browser' in page_source:
                print("  [WAIT] Cloudflare challenge detected, waiting 20s...")
                time.sleep(20)
                page_source = driver.page_source

            if 'Just a moment' in page_source:
                print("  [WAIT] Still on Cloudflare, waiting 30s more...")
                time.sleep(30)
                page_source = driver.page_source

            if 'Just a moment' in page_source:
                print("  [WAIT] Last Cloudflare wait, 30s...")
                time.sleep(30)
                page_source = driver.page_source

            if 'Just a moment' in page_source:
                print("  [FAIL] Could not bypass Cloudflare")
                continue

            # check if tables exist before accepting this attempt
            if 'receiving_and_rushing' in page_source or 'rushing_and_receiving' in page_source:
                break  # tables present, good to go
            elif 'pro-football-reference' in driver.current_url:
                # page loaded but no tables visible - might be hidden in comments
                from bs4 import Comment
                _soup = BeautifulSoup(page_source, 'lxml')
                _comments = _soup.find_all(string=lambda t: isinstance(t, Comment))
                has_table_in_comments = any(
                    'receiving_and_rushing' in str(c) or 'rushing_and_receiving' in str(c)
                    for c in _comments
                )
                if has_table_in_comments:
                    break  # tables in comments, good to go
                else:
                    print(f"  Page loaded but no tables found (attempt {attempt+1}/3)")
                    # continue to retry

        if page_source is None or 'Just a moment' in page_source:
            print("  [FAIL] Could not bypass Cloudflare after 3 attempts")
            return 'failed'

        soup = BeautifulSoup(page_source, 'lxml')

        # get actual player name from page
        h1 = soup.find('h1')
        actual_name = h1.get_text(strip=True) if h1 else player_name
        print(f"  Found: {actual_name}")

        # create player folder with actual name
        folder_name = get_player_folder_name(actual_name)
        player_dir = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(player_dir, exist_ok=True)

        # extract all tables
        tables_found = 0
        saved_filenames = set()

        for table_id, table_info in TABLES_TO_SCRAPE.items():
            target_filename = table_info['filename']

            # skip if we already saved this filename from another table id
            if target_filename in saved_filenames:
                continue

            # find table directly
            table = soup.find('table', {'id': table_id})

            # if not found, check in HTML comments (PFR hides some tables in comments)
            if not table:
                from bs4 import Comment
                comments = soup.find_all(string=lambda text: isinstance(text, Comment))
                for comment in comments:
                    if table_id in str(comment):
                        try:
                            comment_soup = BeautifulSoup(str(comment), 'lxml')
                            table = comment_soup.find('table', {'id': table_id})
                            if table:
                                break
                        except:
                            pass

            if not table:
                continue

            df = parse_table(table)
            if df is not None and len(df) > 0:
                df['Player'] = actual_name
                df['PlayerID'] = player_id
                df = clean_dataframe(df)

                filepath = os.path.join(player_dir, target_filename)
                df.to_csv(filepath, index=False)
                saved_filenames.add(target_filename)
                tables_found += 1

        if tables_found > 0:
            print(f"  [OK] Extracted {tables_found} stat tables -> {player_dir}")
            return 'success'
        else:
            print(f"  [FAIL] No stat tables found on page")
            return 'failed'

    except Exception as e:
        print(f"  [FAIL] Error: {str(e)[:100]}")
        return 'failed'


def combine_all_csvs(input_dir='data/raw/te', output_dir='data/processed'):
    """Combine all individual TE CSV files into master files by stat type."""
    print(f"\nCombining TE CSV files from {input_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    player_folders = [f for f in os.listdir(input_dir)
                      if os.path.isdir(os.path.join(input_dir, f))]

    if not player_folders:
        print("[FAIL] No player folders found")
        return

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

    for filename, dfs in file_types.items():
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            output_file = os.path.join(output_dir, f'all_te_{filename}')
            combined.to_csv(output_file, index=False)
            print(f"  [OK] {filename}: {len(dfs)} players, {len(combined)} rows -> {output_file}")

    print(f"\n[OK] Combined {len(player_folders)} player folders")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='NFL TE Stats Scraper - Opera GX')
    parser.add_argument('--player', type=str, help='Scrape single player by ID (e.g., KelcTr00)')
    parser.add_argument('--test', action='store_true', help='Test mode - scrape first player only')
    parser.add_argument('--force', action='store_true', help='Force re-scrape even if files exist')
    parser.add_argument('--combine', action='store_true', help='Only combine existing CSVs')

    args = parser.parse_args()

    if args.combine:
        combine_all_csvs()
        return

    # determine which TEs to scrape
    if args.player:
        te_list = [(args.player, args.player)]
    elif args.test:
        te_list = [NFL_TES[0]]
        print("\n TEST MODE - Scraping first player only\n")
    else:
        # deduplicate by player ID (keep first occurrence)
        seen_ids = set()
        te_list = []
        for entry in NFL_TES:
            pid = entry[0]
            if pid not in seen_ids:
                seen_ids.add(pid)
                te_list.append(entry)

    print(f"\n{'='*70}")
    print(f"NFL TE STATS SCRAPER - Opera GX with VPN")
    print(f"{len(te_list)} players queued")
    print(f"Tables: Receiving/Rushing, Advanced Rec/Rush, Snap Counts")
    print(f"{'='*70}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    driver = create_driver()

    # warm-up page load to get past Opera GX start page
    print("\nWarm-up: loading PFR homepage...")
    try:
        driver.get("https://www.pro-football-reference.com/")
        time.sleep(5)
        if 'Just a moment' in driver.page_source:
            print("  Cloudflare on warm-up, waiting 15s...")
            time.sleep(15)
        print("  Warm-up done.")
    except:
        print("  Warm-up failed, continuing anyway...")

    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    failed_players = []
    consecutive_fails = 0

    try:
        for i, (player_id, name) in enumerate(te_list, 1):
            # every 10 players, take a long cooldown to avoid rate-limiting
            if i > 1 and (i - 1) % 10 == 0:
                cooldown = random.uniform(45, 70)
                print(f"\n  [COOLDOWN] Pausing {cooldown:.0f}s after {i-1} players to avoid rate-limit...")
                time.sleep(cooldown)

            print(f"\n[{i}/{len(te_list)}]", end='')
            result = scrape_player(driver, player_id, name, force=args.force)
            stats[result] += 1

            if result == 'failed':
                failed_players.append((player_id, name))
                consecutive_fails += 1
                # if we get 3 consecutive fails, restart browser (Cloudflare is sticky)
                if consecutive_fails >= 3:
                    print("\n  [RESTART] 3 consecutive failures - restarting browser...")
                    try:
                        driver.quit()
                    except:
                        pass
                    time.sleep(random.uniform(30, 45))
                    driver = create_driver()
                    # warm up again
                    try:
                        driver.get("https://www.pro-football-reference.com/")
                        time.sleep(8)
                        if 'Just a moment' in driver.page_source:
                            time.sleep(20)
                    except:
                        pass
                    consecutive_fails = 0
            else:
                consecutive_fails = 0

        # retry failed players once with fresh browser and longer delays
        if failed_players:
            print(f"\n{'='*70}")
            print(f"RETRYING {len(failed_players)} failed players with fresh browser...")
            print(f"{'='*70}")
            try:
                driver.quit()
            except:
                pass
            time.sleep(random.uniform(60, 90))
            driver = create_driver()
            # warm up
            try:
                driver.get("https://www.pro-football-reference.com/")
                time.sleep(8)
                if 'Just a moment' in driver.page_source:
                    time.sleep(20)
            except:
                pass

            retry_success = 0
            for j, (player_id, name) in enumerate(failed_players, 1):
                if j > 1 and (j - 1) % 5 == 0:
                    cooldown = random.uniform(45, 70)
                    print(f"\n  [COOLDOWN] Retry pause {cooldown:.0f}s...")
                    time.sleep(cooldown)

                print(f"\n[RETRY {j}/{len(failed_players)}]", end='')
                result = scrape_player(driver, player_id, name, force=True)
                if result == 'success':
                    stats['success'] += 1
                    stats['failed'] -= 1
                    retry_success += 1
            print(f"\n  Retry recovered {retry_success}/{len(failed_players)} players")

        print(f"\n{'='*70}")
        print(f"RESULTS:")
        print(f"  [OK] Success: {stats['success']}")
        print(f"  >> Skipped:   {stats['skipped']}")
        print(f"  [FAIL] Failed: {stats['failed']}")
        print(f"  Total:        {sum(stats.values())}")
        print(f"{'='*70}")

        if stats['success'] > 0:
            print("\nCombining all TE CSVs...")
            combine_all_csvs()

    finally:
        driver.quit()
        print("\nBrowser closed. Done!")


if __name__ == '__main__':
    main()
