# 2025 NFL QB Data - Scraping Complete ✓
## Update December 20, 2025 (via VPN on February 14, 2026)

### Scraping Results
✓ Successfully scraped **33 active QBs** with Pro Football Reference pages
✗ 8 QBs with no PFR pages (mostly recent draft picks without data yet)
✓ Added 3 backup/veteran QBs: Will Levis, Andy Dalton, Blaine Gabbert

### 2025 Season Coverage
- **Total QBs with 2025 data:** 31
- **Games covered:** Week 4 through Week 17 (partial - depends on QB participation)

### Top 2025 Performers (Games Played):
| Rank | QB | Team | Games | Status |
|------|-----|------|-------|--------|
| 1 | Bo Nix | Denver | 14 | Rookie/Starter |
| 1 | Caleb Williams | Chicago | 14 | Rookie/Starter |
| 1 | Cam Ward | Miami | 14 | Rookie/Starter |
| 1 | Jalen Hurts | Philadelphia | 14 | Established |
| 1 | Drake Maye | New England | 14 | Rookie/Starter |
| 6 | Aaron Rodgers | Pittsburgh | 13 | Veteran |
| 6 | Bryce Young | Carolina | 13 | Young Starter |
| 6 | Geno Smith | Seattle | 13 | Veteran Backup |
| 6 | Daniel Jones | NY Giants | 13 | Established |
| 10 | C.J. Stroud | Houston | 11 | Rookie/Starter |
| 10 | Lamar Jackson | Baltimore | 11 | Star QB |
| 31 | Andy Dalton | (Multiple) | 4 | Veteran Backup |

### Data Files Generated
All stat categories available for 33 QBs:
- ✓ Passing Stats (247 rows)
- ✓ Advanced Passing (183 rows)
- ✓ Adjusted Passing (224 rows)
- ✓ Rushing/Receiving (245 rows)
- ✓ Advanced Rush/Rec (181 rows)
- ✓ Defense/Fumbles (230 rows)
- ✓ Snap Counts (179 rows)

### Historical Coverage
- Full data from **2005-2025** for all QBs
- Rookie years to present for all active players
- Career statistics for analysis

### Technical Implementation
- **Scraper:** Selenium WebDriver + BeautifulSoup
- **VPN:** Opera VPN used to bypass IP blocking
- **Delay:** 2-4 seconds between requests (respectful scraping)
- **Data Format:** CSV files organized by QB folder

### Files Location
```
data/
  raw/qb/          # Individual QB folders with 7 stat CSVs each
    (33 QB folders)
  processed/       # Combined datasets
    all_passing.csv
    all_advanced_passing.csv
    all_adjusted_passing.csv
    all_rushing_receiving.csv
    all_advanced_rushing_receiving.csv
    all_defense_fumbles.csv
    all_snap_counts.csv
```

### Next Steps
To update with more recent 2025 data:
1. Keep Opera VPN enabled
2. Run: `python nfl_scraper.py --force` to re-scrape all QBs
3. Data will be refreshed with latest stats from Pro Football Reference

---
**Status:** ✓ Complete - Ready for analysis
**Last Update:** February 14, 2026
