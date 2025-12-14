#!/usr/bin/env python3
"""
Smart NFL QB ID Finder
Systematically searches for correct ProFootballReference player IDs
"""

import cloudscraper
import time

# Current NFL Starting QBs with known correct info
NFL_QBS_2025 = {
    'Josh Allen': ('J', 'AlleJo02', 1996),           # Buffalo Bills
    'Aaron Rodgers': ('A', 'RodAa01', 1983),         # New York Jets
    'Tua Tagovailoa': ('T', 'TagoTa01', 1998),       # Miami Dolphins
    'Jacoby Brissett': ('B', 'BrisJa01', 1988),      # New England Patriots
    
    'Lamar Jackson': ('L', 'JackLa00', 1997),        # Baltimore Ravens
    'Joe Burrow': ('B', 'BurrJo01', 1996),           # Cincinnati Bengals
    'C.J. Stroud': ('S', 'StroC.01', 2003),          # Houston Texans
    'Will Levis': ('L', 'LeviWi01', 2001),           # Tennessee Titans
    
    'Patrick Mahomes': ('M', 'MahoP01', 1995),       # Kansas City Chiefs
    'Justin Herbert': ('H', 'HerbJu00', 1998),       # Los Angeles Chargers
    'Jared Goff': ('G', 'GoffJa01', 1992),           # Detroit Lions
    'Gardner Minshew': ('M', 'MinshGa01', 1996),     # Las Vegas Raiders
    
    'Jalen Hurts': ('H', 'HurtJa00', 1998),          # Philadelphia Eagles
    'Dak Prescott': ('P', 'PrescD01', 1992),         # Dallas Cowboys
    'Jayden Daniels': ('D', 'DanieJa01', 2003),      # Washington
    'Daniel Jones': ('J', 'JonesDa01', 1997),        # New York Giants
    
    'Bryce Young': ('Y', 'YoungBr01', 2001),         # Carolina Panthers
    'Kirk Cousins': ('C', 'CousK01', 1988),          # Atlanta Falcons
    'Derek Carr': ('C', 'CarrDe01', 1987),           # New Orleans Saints
    'Baker Mayfield': ('M', 'MayfB01', 1997),        # Tampa Bay Buccaneers
    
    'Brock Purdy': ('P', 'PurdB01', 1999),           # San Francisco 49ers
    'Geno Smith': ('S', 'SmiGeNo01', 1990),          # Seattle Seahawks
    'Kyler Murray': ('M', 'MurrK01', 1997),          # Arizona Cardinals
    'Matthew Stafford': ('S', 'StafM01', 1988),      # Los Angeles Rams
}


def test_player_id(player_id, first_letter):
    """Test if a player ID exists on PFR"""
    try:
        cs = cloudscraper.create_scraper()
        url = f'https://www.pro-football-reference.com/players/{first_letter}/{player_id}.htm'
        response = cs.get(url, timeout=2)
        return response.status_code == 200
    except:
        return False


def find_player_id_variations(name, first_letter):
    """Try common variations of a player ID"""
    # Split name
    parts = name.split()
    
    variations = [
        # Already provided format
        # Try different combinations
        f"{parts[-1][:3]}{parts[0][:2]}00",
        f"{parts[-1][:3]}{parts[0][:2]}01",
        f"{parts[-1][:3]}{parts[0][:2]}02",
        f"{parts[-1]}{parts[0][:2]}00",
        f"{parts[-1]}{parts[0][:2]}01",
        f"{parts[0][:3]}{parts[-1][:2]}00",
        f"{parts[0][:3]}{parts[-1][:2]}01",
    ]
    
    for var in variations:
        if test_player_id(var, first_letter):
            return var
    
    return None


def main():
    print('\n' + '='*90)
    print(' '*30 + 'NFL QB ID FINDER - SYSTEMATIC SEARCH')
    print('='*90 + '\n')
    
    found = []
    not_found = []
    
    for idx, (name, (letter, suggested_id, birth_year)) in enumerate(NFL_QBS_2025.items(), 1):
        print(f'[{idx:2}/{len(NFL_QBS_2025)}] {name:25}...', end=' ', flush=True)
        
        # First try suggested ID
        if test_player_id(suggested_id, letter):
            print(f'✓ {suggested_id}')
            found.append((suggested_id, name, birth_year))
            time.sleep(0.3)
            continue
        
        # Try variations
        alt_id = find_player_id_variations(name, letter)
        if alt_id:
            print(f'✓ {alt_id} (variant)')
            found.append((alt_id, name, birth_year))
        else:
            print(f'✗ NOT FOUND')
            not_found.append((name, suggested_id))
        
        time.sleep(0.3)
    
    # Output
    print('\n' + '='*90)
    print('VERIFIED PLAYER IDs FOR scrape_all_nfl_complete.py:')
    print('='*90 + '\n')
    
    print('NFL_STARTING_QBS = [')
    for player_id, name, birth_year in found:
        print(f"    ('{player_id}', '{name}', {birth_year}),")
    print(']\n')
    
    print('='*90)
    print(f'RESULTS: {len(found)} found / {len(not_found)} not found = {len(found)} TOTAL')
    print('='*90 + '\n')
    
    if not_found:
        print('Not found (need manual ID lookup):')
        for name, suggested in not_found:
            print(f'  - {name} (suggested: {suggested})')
        print()


if __name__ == '__main__':
    main()
