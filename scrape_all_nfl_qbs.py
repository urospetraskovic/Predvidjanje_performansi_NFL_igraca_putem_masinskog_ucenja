#!/usr/bin/env python3
"""
Automated NFL QB Data Scraper - COMPLETE VERSION
Scrapes all current starting QBs, fixes data, and organizes into qb folder
Updated for 2025 NFL Season
"""

import pandas as pd
from scraper_production import NFLScraper
import os
from datetime import datetime
import time

# Verified working QBs (confirmed to exist on PFR)
VERIFIED_QBS = [
    ('AlleJo02', 'Josh Allen', 1996),
    ('JackLa00', 'Lamar Jackson', 1997),
    ('BurrJo01', 'Joe Burrow', 1996),
    ('HurtJa00', 'Jalen Hurts', 1998),
    ('HerbJu00', 'Justin Herbert', 1998),
]

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
        
        # Move to qb folder
        qb_filepath = f'data/raw/qb/{player_id}_stats.csv'
        df.to_csv(qb_filepath, index=False)
        os.remove(filepath)
        
        num_seasons = len(df)
        min_season = int(df['Season'].min())
        max_season = int(df['Season'].max())
        
        return True, f'{num_seasons} seasons ({min_season}-{max_season})'
        
    except Exception as e:
        return False, str(e)[:40]  # Truncate error message


def main():
    print('\n' + '='*80)
    print(' '*20 + 'NFL QB DATA SCRAPER - AUTOMATED BATCH')
    print('='*80 + '\n')
    
    start_time = datetime.now()
    scraper = NFLScraper()
    
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    print('Processing verified NFL QBs...\n')
    
    for idx, (player_id, name, birth_year) in enumerate(VERIFIED_QBS, 1):
        # Check if already exists
        qb_filepath = f'data/raw/qb/{player_id}_stats.csv'
        if os.path.exists(qb_filepath):
            results['skipped'].append((name, 'Already in qb/'))
            print(f'[{idx}/{len(VERIFIED_QBS)}] SKIP {name:20} - Already exists')
            continue
        
        print(f'[{idx}/{len(VERIFIED_QBS)}] Scraping {name:20}...', end=' ', flush=True)
        success, message = scrape_and_fix_qb(player_id, name, birth_year, scraper)
        
        if success:
            results['success'].append((name, message))
            print(f'✓ {message}')
        else:
            results['failed'].append((name, message))
            print(f'✗ {message}')
        
        time.sleep(0.5)  # Be respectful to PFR
    
    # Generate report
    elapsed = datetime.now() - start_time
    
    print('\n' + '='*80)
    print(' '*25 + 'BATCH COMPLETE')
    print('='*80)
    print(f'\nTime: {elapsed.total_seconds():.1f}s | Success: {len(results["success"])} | Failed: {len(results["failed"])}')
    
    if results['success']:
        print(f'\n✓ Successfully processed ({len(results["success"])}):\n')
        for name, details in results['success']:
            print(f'   {name:20} {details}')
    
    if results['failed']:
        print(f'\n✗ Failed ({len(results["failed"])}):\n')
        for name, reason in results['failed']:
            print(f'   {name:20} {reason}')
    
    if results['skipped']:
        print(f'\n- Skipped ({len(results["skipped"])}):\n')
        for name, reason in results['skipped']:
            print(f'   {name:20} {reason}')
    
    # Final summary
    total_qbs = len([f for f in os.listdir('data/raw/qb/') if f.endswith('.csv')])
    print(f'\n{"="*80}')
    print(f'  TOTAL QB FILES IN data/raw/qb/: {total_qbs}')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    main()
