#!/usr/bin/env python3
"""
NFL QB Scraper - Current Working Version
Scrapes verified starting QBs and can be easily extended
"""

import pandas as pd
from scraper_production import NFLScraper
import os
from datetime import datetime
import time
import random

# VERIFIED WORKING QBs (Confirmed IDs that exist on PFR)
VERIFIED_QBS = [
    ('AlleJo02', 'Josh Allen', 1996),          # Buffalo Bills
    ('JackLa00', 'Lamar Jackson', 1997),       # Baltimore Ravens
    ('BurrJo01', 'Joe Burrow', 1996),          # Cincinnati Bengals
    ('HurtJa00', 'Jalen Hurts', 1998),         # Philadelphia Eagles
    ('HerbJu00', 'Justin Herbert', 1998),      # Los Angeles Chargers
]

# ADD MORE QBs HERE AS YOU FIND THEIR CORRECT IDs
# To find a QB's ID:
# 1. Go to https://www.pro-football-reference.com/
# 2. Search for the player
# 3. Copy their player ID from the URL (e.g., "AlleJo02" from /players/A/AlleJo02.htm)
# 4. Add to list below with format: ('PLAYERID', 'Full Name', birth_year)

ADDITIONAL_QBS = [
    ('RodgAa00', 'Aaron Rodgers', 1983),     # Green Bay Packers
    ('MahoPa00', 'Patrick Mahomes', 1995),   # Kansas City Chiefs
    ('PresDa01', 'Dak Prescott', 1992),      # Dallas Cowboys
    ('GoffJa00', 'Jared Goff', 1992),        # Detroit Lions
    ('PurdBr00', 'Brock Purdy', 2001),       # San Francisco 49ers
    ('YoungBr02', 'Bryce Young', 2001),      # New England Patriots
    ('MayeDr00', 'Drake Maye', 2004),        # New England Patriots
    ('TagoTu00', 'Tua Tagovailoa', 1998),    # Miami Dolphins
    ('StroCJ00', 'CJ Stroud', 2003),         # Houston Texans
    ('WillCa00', 'Caleb Williams', 2002),    # Chicago Bears
    ('DarnSa00', 'Sam Darnold', 1997),       # Tennessee Titans
    ('JonesDa01', 'Daniel Jones', 1997),     # New York Giants
    ('LawrTr00', 'Trevor Lawrence', 1999),   # Jacksonville Jaguars
]

# Combine all QBs
ALL_QBS = VERIFIED_QBS + ADDITIONAL_QBS


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
        
        # Clean up data
        df = pd.read_csv(filepath)
        
        if len(df) == 0:
            return False, 'Empty data'
        
        # Season and Age are already in the table from PFR - just clean them up
        df['Season'] = pd.to_numeric(df['Season'], errors='coerce')
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        
        # Remove the Lg column (always 'NFL')
        if 'Lg' in df.columns:
            df = df.drop('Lg', axis=1)
        
        # Reorder columns to have Season, Age, Team first
        cols = df.columns.tolist()
        if 'Season' in cols:
            cols.remove('Season')
        if 'Age' in cols:
            cols.remove('Age')
        if 'Team' in cols:
            cols.remove('Team')
        
        # Put Season, Age, Team at the front
        if 'Team' in df.columns:
            df = df[['Season', 'Age', 'Team'] + cols]
        else:
            df = df[['Season', 'Age'] + cols]
        
        # Remove summary rows
        df = df[df['Season'].notna()]
        df = df[df['Season'] != '']
        
        if len(df) == 0:
            return False, 'No valid data'
        
        # Move to qb folder
        qb_filepath = f'data/raw/qb/{player_id}_stats.csv'
        df.to_csv(qb_filepath, index=False)
        # Keep raw file for debugging
        # os.remove(filepath)
        
        num_seasons = len(df)
        min_season = int(df['Season'].min())
        max_season = int(df['Season'].max())
        
        return True, f'{num_seasons} seasons ({min_season}-{max_season})'
        
    except Exception as e:
        return False, str(e)[:40]


def main():
    print('\n' + '='*90)
    print(' '*20 + 'NFL QB DATA SCRAPER - AUTOMATED BATCH PROCESSING')
    print('='*90 + '\n')
    
    if not ALL_QBS:
        print('ERROR: No QBs configured. Add QBs to VERIFIED_QBS or ADDITIONAL_QBS.')
        print('\nTo add a QB:')
        print('1. Go to https://www.pro-football-reference.com/')
        print('2. Search for the player')
        print('3. Copy the player ID from the URL')
        print('4. Add to ADDITIONAL_QBS in this script\n')
        return
    
    start_time = datetime.now()
    scraper = NFLScraper()
    
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    print(f'Processing {len(ALL_QBS)} QBs...\n')
    
    for idx, (player_id, name, birth_year) in enumerate(ALL_QBS, 1):
        # Check if already exists
        qb_filepath = f'data/raw/qb/{player_id}_stats.csv'
        if os.path.exists(qb_filepath):
            results['skipped'].append(name)
            print(f'[{idx:2}/{len(ALL_QBS)}] SKIP {name:25} (already in qb/)')
            continue
        
        print(f'[{idx:2}/{len(ALL_QBS)}] {name:25}...', end=' ', flush=True)
        success, message = scrape_and_fix_qb(player_id, name, birth_year, scraper)
        
        if success:
            results['success'].append((name, message))
            print(f'✓ {message}')
        else:
            results['failed'].append((name, message))
            print(f'✗ {message}')
        
        # Smart rate limiting - random 5-8 seconds
        time.sleep(random.uniform(5, 8))
    
    # Generate report
    elapsed = datetime.now() - start_time
    
    print('\n' + '='*90)
    print(' '*30 + 'BATCH COMPLETE')
    print('='*90)
    print(f'\nTime: {elapsed.total_seconds():.1f}s')
    
    if results['success']:
        print(f'\n✓ SUCCESS ({len(results["success"])}):\n')
        for name, details in results['success']:
            print(f'   {name:25} {details}')
    
    if results['failed']:
        print(f'\n✗ FAILED ({len(results["failed"])}):\n')
        for name, reason in results['failed']:
            print(f'   {name:25} {reason}')
    
    if results['skipped']:
        print(f'\n- SKIPPED ({len(results["skipped"])}):\n')
        for name in results['skipped']:
            print(f'   {name}')
    
    # Final count
    total_qbs = len([f for f in os.listdir('data/raw/qb/') if f.endswith('.csv')])
    print(f'\n{"="*90}')
    print(f'  TOTAL QB FILES: {total_qbs}')
    print(f'{"="*90}\n')
    
    # Instructions for adding more
    if len(results['failed']) > 0 or not ALL_QBS:
        print('HOW TO ADD MORE QBs:')
        print('1. Visit: https://www.pro-football-reference.com/players/')
        print('2. Search for a QB name')
        print('3. Copy player ID from URL (e.g., RodAa01 from /players/R/RodAa01.htm)')
        print('4. Add to ADDITIONAL_QBS in this script with birth year')
        print('5. Run this script again\n')


if __name__ == '__main__':
    main()
