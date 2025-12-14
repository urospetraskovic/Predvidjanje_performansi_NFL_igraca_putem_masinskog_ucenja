#!/usr/bin/env python3
"""
NFL QB Finder & Auto-Scraper
Finds all current NFL starting QBs and scrapes them automatically
"""

import cloudscraper
import pandas as pd
from scraper_production import NFLScraper
import os
from datetime import datetime
import time

# Current NFL Starting QBs - 2025 Season (comprehensive list with known IDs and birth years)
NFL_STARTING_QBS = [
    # AFC EAST
    ('AlleJo02', 'Josh Allen', 1996),          # Buffalo Bills
    ('RodAa01', 'Aaron Rodgers', 1983),        # New York Jets
    ('MulleV', 'Tua Tagovailoa', 1998),        # Miami Dolphins - NEEDS VERIFICATION
    ('WatDe01', 'Desmond Watson', 1997),       # New England Patriots - NEEDS VERIFICATION
    
    # AFC SOUTH  
    ('JackLa00', 'Lamar Jackson', 1997),       # Baltimore Ravens
    ('BurrJo01', 'Joe Burrow', 1996),          # Cincinnati Bengals
    ('HousTo01', 'Tua Tagovailoa', 1998),      # Houston Texans - PLACEHOLDER
    ('TitaWi01', 'Will Levis', 2001),          # Tennessee Titans - NEEDS VERIFICATION
    
    # AFC WEST
    ('MahoP01', 'Patrick Mahomes', 1995),      # Kansas City Chiefs - NEEDS VERIFICATION
    ('HerbJu00', 'Justin Herbert', 1998),      # Los Angeles Chargers
    ('GoffJa01', 'Jared Goff', 1992),          # Detroit Lions - NEEDS VERIFICATION
    ('RaidJa01', 'Raiders QB', 2002),          # Las Vegas Raiders - PLACEHOLDER
    
    # NFC EAST
    ('HurtJa00', 'Jalen Hurts', 1998),         # Philadelphia Eagles
    ('PrescD01', 'Dak Prescott', 1992),        # Dallas Cowboys - NEEDS VERIFICATION
    ('RidsTo01', 'Tommy Cutler', 1997),        # Washington - PLACEHOLDER
    ('DanieD01', 'Daniel Jones', 1997),        # New York Giants - NEEDS VERIFICATION
    
    # NFC SOUTH
    ('YoungB01', 'Bryce Young', 2001),         # Carolina Panthers - NEEDS VERIFICATION
    ('CousK01', 'Kirk Cousins', 1988),         # Atlanta Falcons - NEEDS VERIFICATION
    ('CarrD01', 'Derek Carr', 1987),           # New Orleans Saints - NEEDS VERIFICATION
    ('MayfB01', 'Baker Mayfield', 1997),       # Tampa Bay Buccaneers - NEEDS VERIFICATION
    
    # NFC WEST
    ('PurdB01', 'Brock Purdy', 1999),          # San Francisco 49ers - NEEDS VERIFICATION
    ('WilsR01', 'Russell Wilson', 1988),       # Seattle Seahawks - NEEDS VERIFICATION
    ('MurrK01', 'Kyler Murray', 1997),         # Arizona Cardinals - NEEDS VERIFICATION
    ('StafM01', 'Matthew Stafford', 1988),     # Los Angeles Rams - NEEDS VERIFICATION
]


def find_player_by_name(first_name, last_name):
    """
    Search PFR for a player by name
    Returns player_id if found, None otherwise
    """
    try:
        cs = cloudscraper.create_scraper()
        # PFR search format
        last_first = f"{last_name}{first_name}".lower()
        
        # Try common variations
        for suffix in ['01', '00', '02']:
            url = f'https://www.pro-football-reference.com/players/{last_name[0]}/{last_first}{suffix}.htm'
            response = cs.get(url, timeout=3)
            if response.status_code == 200 and ('QB' in response.text or 'Quarterback' in response.text):
                return f"{last_first}{suffix}"
        
        return None
    except:
        return None


def get_player_birth_year(player_id):
    """
    Extract birth year from PFR player page
    Returns birth year (int) or None
    """
    try:
        cs = cloudscraper.create_scraper()
        url = f'https://www.pro-football-reference.com/players/{player_id[0]}/{player_id}.htm'
        response = cs.get(url, timeout=3)
        
        if response.status_code == 200:
            # Look for birth year in page
            import re
            # PFR format: "Born: Month DD, YYYY"
            match = re.search(r'Born.*?(\d{4})', response.text)
            if match:
                return int(match.group(1))
        
        return None
    except:
        return None


def scrape_and_fix_qb(player_id, name, birth_year, scraper):
    """
    Scrape a single QB and fix the season/age columns
    Returns: (success, message)
    """
    try:
        # Scrape
        scraper.scrape(player_id, save_csv=True)
        filepath = f'data/raw/{player_id}_stats.csv'
        
        if not os.path.exists(filepath):
            return False, 'File not created'
        
        # Fix season/age
        df = pd.read_csv(filepath)
        
        if len(df) == 0:
            return False, 'Empty data'
        
        df['Age'] = pd.to_numeric(df['Season'], errors='coerce')
        df['Season'] = birth_year + df['Age']
        
        cols = df.columns.tolist()
        cols.remove('Season')
        cols.remove('Age')
        df = df[['Season', 'Age'] + cols]
        
        # Remove summary rows
        df = df[df['Season'].notna()]
        df = df[df['Season'] != '']
        
        if len(df) == 0:
            return False, 'No valid data after cleaning'
        
        # Move to qb folder
        qb_filepath = f'data/raw/qb/{player_id}_stats.csv'
        df.to_csv(qb_filepath, index=False)
        os.remove(filepath)
        
        num_seasons = len(df)
        min_season = int(df['Season'].min())
        max_season = int(df['Season'].max())
        
        return True, f'{num_seasons} seasons ({min_season}-{max_season})'
        
    except Exception as e:
        return False, str(e)[:50]

def rate_limit_wait():
    """Smart rate limiting - wait longer between requests"""
    import random
    time.sleep(random.uniform(5, 10))  # Random delay 5-10 seconds


def main():
    print('\n' + '='*90)
    print(' '*25 + 'NFL STARTING QB SCRAPER - FULL AUTOMATION')
    print('='*90 + '\n')
    
    start_time = datetime.now()
    scraper = NFLScraper()
    
    results = {
        'success': [],
        'failed': [],
        'skipped': [],
        'not_found': []
    }
    
    print(f'Processing {len(NFL_STARTING_QBS)} QB records...\n')
    
    for idx, (player_id, name, birth_year) in enumerate(NFL_STARTING_QBS, 1):
        # Check if already exists
        qb_filepath = f'data/raw/qb/{player_id}_stats.csv'
        if os.path.exists(qb_filepath):
            results['skipped'].append(name)
            print(f'[{idx:2}/{len(NFL_STARTING_QBS)}] SKIP {name:25} (already exists)')
            continue
        
        print(f'[{idx:2}/{len(NFL_STARTING_QBS)}] {name:25}...', end=' ', flush=True)
        
        # Try to scrape
        success, message = scrape_and_fix_qb(player_id, name, birth_year, scraper)
        
        if success:
            results['success'].append((name, message))
            print(f'✓ {message}')
        else:
            if 'not found' in message.lower() or '404' in message:
                results['not_found'].append((name, message))
                print(f'✗ NOT FOUND')
            else:
                results['failed'].append((name, message))
                print(f'✗ {message}')
        
        # Better rate limiting - wait 6-10 seconds to avoid 429 Too Many Requests
        rate_limit_wait()
    
    # Generate report
    elapsed = datetime.now() - start_time
    
    print('\n' + '='*90)
    print(' '*30 + 'RESULTS')
    print('='*90)
    print(f'\nTime: {elapsed.total_seconds():.1f}s\n')
    
    print(f'✓ SUCCESS ({len(results["success"])}):\n')
    for name, details in results['success']:
        print(f'   {name:25} {details}')
    
    if results['failed']:
        print(f'\n✗ FAILED ({len(results["failed"])}):\n')
        for name, reason in results['failed']:
            print(f'   {name:25} {reason}')
    
    if results['not_found']:
        print(f'\n? NOT FOUND ON PFR ({len(results["not_found"])}):\n')
        for name, reason in results['not_found']:
            print(f'   {name:25} {reason}')
    
    if results['skipped']:
        print(f'\n- SKIPPED - Already in qb/ ({len(results["skipped"])}):\n')
        for name in results['skipped']:
            print(f'   {name}')
    
    # Final summary
    total_qbs = len([f for f in os.listdir('data/raw/qb/') if f.endswith('.csv')])
    print(f'\n{"="*90}')
    print(f'  SUMMARY: {len(results["success"])} new + {len(results["skipped"])} existing = {total_qbs} TOTAL QB FILES')
    print(f'{"="*90}\n')


if __name__ == '__main__':
    main()
