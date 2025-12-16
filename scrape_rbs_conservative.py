#!/usr/bin/env python3
"""
Conservative RB Scraper - Use this after IP rate limiting cooldown (2-4 hours)
This script scrapes RBs in small batches with LONG delays to avoid re-triggering rate limits.
"""

import time
from nfl_rb_scraper import NFLRBScraper

# Divide 32 RBs into 4 batches of 8
BATCHES = [
    # Batch 1: AFC East
    [
        ('CookJa01', 'James Cook'),
        ('AchaDe00', 'De\'Von Achane'),
        ('SteveRh00', 'Rhamondre Stevenson'),
        ('HallBr02', 'Breece Hall'),
        ('HenrDe00', 'Derrick Henry'),
        ('BrowCh01', 'Chase Brown'),
        ('HarriNa00', 'Najee Harris'),
        ('WarrJa00', 'Jaylen Warren'),
    ],
    # Batch 2: AFC South + West
    [
        ('MixoJo00', 'Joe Mixon'),
        ('BagsTA00', 'Tank Bigsby'),
        ('PollTo00', 'Tony Pollard'),
        ('PaceIs00', 'Isiah Pacheco'),
        ('JackJo05', 'Josh Jacobs'),
        ('WillJa02', 'Javonte Williams'),
        ('BarkSa00', 'Saquon Barkley'),
        ('SinlDe00', 'Devin Singletary'),
    ],
    # Batch 3: NFC East + North Start
    [
        ('EllkZe00', 'Ezekiel Elliott'),
        ('RobiBr02', 'Brian Robinson Jr.'),
        ('MontDa02', 'David Montgomery'),
        ('JoneAa02', 'Aaron Jones'),
        ('MattAl00', 'Alexander Mattison'),
        ('HubbCh00', 'Chuba Hubbard'),
        ('KamaAl00', 'Alvin Kamara'),
        ('WhitRa00', 'Rachaad White'),
    ],
    # Batch 4: NFC South + West
    [
        ('EtieTr00', 'Travis Etienne'),
        ('MccaCh00', 'Christian McCaffrey'),
        ('WalkKe03', 'Kenneth Walker III'),
        ('WillKy00', 'Kyren Williams'),
        ('ConnJa00', 'James Conner'),
    ],
]

def main():
    print("=" * 70)
    print("CONSERVATIVE RB SCRAPER - For use after IP cooldown")
    print("=" * 70)
    print("\nThis script:")
    print("  • Uses 60-120 second delays between requests (conservative)")
    print("  • Scrapes in batches of 8 players")
    print("  • Waits 30 minutes between batches")
    print("  • Minimizes risk of re-triggering rate limits\n")
    
    # Create scraper with very conservative delays
    scraper = NFLRBScraper(delay_min=60, delay_max=120, verbose=True)
    
    total_batches = len(BATCHES)
    
    for batch_num, batch in enumerate(BATCHES, 1):
        print(f"\n{'='*70}")
        print(f"BATCH {batch_num}/{total_batches}: Scraping {len(batch)} players")
        print(f"{'='*70}\n")
        
        for player_id, name in batch:
            try:
                scraper.scrape_player(player_id, name, save=True)
            except Exception as e:
                print(f"  ! Error scraping {name}: {str(e)[:100]}")
                if '429' in str(e) or '403' in str(e):
                    print("  ! Rate limit triggered again - aborting batch")
                    print("  ! The IP is still blocked. Wait another 2-4 hours and retry.")
                    return
        
        # Wait between batches
        if batch_num < total_batches:
            wait_seconds = 1800  # 30 minutes
            print(f"\n{'='*70}")
            print(f"Batch {batch_num} complete!")
            print(f"Waiting 30 minutes before next batch...")
            print(f"{'='*70}\n")
            
            for i in range(wait_seconds, 0, -300):  # Count down every 5 minutes
                if i > 0:
                    mins = i // 60
                    print(f"  Waiting... {mins} minutes remaining", end='\r')
                    time.sleep(min(300, i))
    
    # Combine all CSVs at the end
    print(f"\n{'='*70}")
    print("All batches complete! Combining CSV files...")
    print(f"{'='*70}\n")
    scraper.combine_all_csvs()
    
    print(f"\n{'='*70}")
    print("SUCCESS! All RBs scraped and combined.")
    print(f"{'='*70}")
    print(f"\nData saved to:")
    print(f"  • Individual files: data/raw/rb/[RB_Name]/")
    print(f"  • Combined files:   data/processed/all_*.csv")
    print(f"\nStatistics:")
    print(f"  • Success: {scraper.stats['success']}")
    print(f"  • Failed:  {scraper.stats['failed']}")
    print(f"  • Skipped: {scraper.stats['skipped']}")

if __name__ == '__main__':
    main()
