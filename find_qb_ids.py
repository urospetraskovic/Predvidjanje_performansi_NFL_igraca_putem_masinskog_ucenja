#!/usr/bin/env python3
"""
NFL QB Finder - Locates correct ProFootballReference player IDs for current QBs
"""

import cloudscraper
import time

# Current NFL Starting QBs - 2025 Season (first/last names)
NFL_STARTING_QBS_2025 = [
    ('Josh', 'Allen'),           # Buffalo Bills
    ('Aaron', 'Rodgers'),        # New York Jets
    ('Tua', 'Tagovailoa'),       # Miami Dolphins
    ('Jacoby', 'Brissett'),      # New England Patriots
    
    ('Lamar', 'Jackson'),        # Baltimore Ravens
    ('Joe', 'Burrow'),           # Cincinnati Bengals
    ('C.J.', 'Stroud'),          # Houston Texans
    ('Will', 'Levis'),           # Tennessee Titans
    
    ('Patrick', 'Mahomes'),      # Kansas City Chiefs
    ('Justin', 'Herbert'),       # Los Angeles Chargers
    ('Jared', 'Goff'),           # Detroit Lions
    ('Gardner', 'Minshew'),      # Las Vegas Raiders
    
    ('Jalen', 'Hurts'),          # Philadelphia Eagles
    ('Dak', 'Prescott'),         # Dallas Cowboys
    ('Jayden', 'Daniels'),       # Washington
    ('Daniel', 'Jones'),         # New York Giants
    
    ('Bryce', 'Young'),          # Carolina Panthers
    ('Kirk', 'Cousins'),         # Atlanta Falcons
    ('Derek', 'Carr'),           # New Orleans Saints
    ('Baker', 'Mayfield'),       # Tampa Bay Buccaneers
    
    ('Brock', 'Purdy'),          # San Francisco 49ers
    ('Geno', 'Smith'),           # Seattle Seahawks
    ('Kyler', 'Murray'),         # Arizona Cardinals
    ('Matthew', 'Stafford'),     # Los Angeles Rams
]


def find_player_id(first_name, last_name):
    """
    Search PFR for a player by trying common ID formats
    Returns: (player_id, full_name) or (None, None)
    """
    try:
        cs = cloudscraper.create_scraper()
        
        # Clean up names
        last = last_name.replace('.', '').lower()
        first = first_name.replace('.', '').lower()
        
        # Try variations (PFR format: LastFirst00, LastFirst01, etc.)
        for suffix in ['00', '01', '02', '03']:
            player_id = f"{last}{first[:2]}{suffix}"
            url = f'https://www.pro-football-reference.com/players/{last[0]}/{player_id}.htm'
            
            response = cs.get(url, timeout=3)
            if response.status_code == 200 and ('QB' in response.text or 'Quarterback' in response.text):
                # Extract name from page
                import re
                name_match = re.search(r'<h1[^>]*>([^<]+)', response.text)
                if name_match:
                    full_name = name_match.group(1).strip()
                    return player_id, full_name
        
        return None, None
        
    except Exception as e:
        return None, None


def main():
    print('\n' + '='*80)
    print(' '*25 + 'FINDING CORRECT NFL QB PLAYER IDs')
    print('='*80 + '\n')
    
    found_qbs = []
    not_found = []
    
    for idx, (first, last) in enumerate(NFL_STARTING_QBS_2025, 1):
        print(f'[{idx:2}/{len(NFL_STARTING_QBS_2025)}] {first} {last:20}...', end=' ', flush=True)
        
        player_id, full_name = find_player_id(first, last)
        
        if player_id:
            found_qbs.append((player_id, full_name, first, last))
            print(f'✓ {player_id}')
        else:
            not_found.append((first, last))
            print(f'✗ Not found')
        
        time.sleep(0.5)
    
    # Generate Python code for correct QBs
    print('\n' + '='*80)
    print('VERIFIED PLAYER IDs - Copy into scrape_all_nfl_complete.py:')
    print('='*80 + '\n')
    
    print('NFL_STARTING_QBS = [')
    for player_id, full_name, first, last in found_qbs:
        print(f"    ('{player_id}', '{full_name}', XXXX),  # {first} {last}")
    print(']\n')
    
    print('='*80)
    print(f'FOUND: {len(found_qbs)} / {len(NFL_STARTING_QBS_2025)}')
    print('='*80 + '\n')
    
    if not_found:
        print('Could not find on PFR:')
        for first, last in not_found:
            print(f'  - {first} {last}')
        print()


if __name__ == '__main__':
    main()
