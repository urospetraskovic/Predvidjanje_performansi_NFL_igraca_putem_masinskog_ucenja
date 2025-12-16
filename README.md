# NFL QB Stats Analyzer

Scrapes and analyzes NFL quarterback statistics from Pro Football Reference.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Scrape all QBs (skips existing files)
python nfl_scraper.py

# Scrape specific player
python nfl_scraper.py --player MahoPa00

# Force re-scrape all (overwrites existing)
python nfl_scraper.py --force

# Test with one player
python nfl_scraper.py --test

# Only combine existing CSVs
python nfl_scraper.py --combine
```

## Project Structure

```
├── nfl_scraper.py          # Main scraper script
├── requirements.txt        # Python dependencies
├── data/
│   ├── raw/qb/            # Individual QB stat CSVs
│   └── processed/         # Combined/analyzed data
├── notebooks/             # Jupyter notebooks for analysis
├── scripts/               # Additional utility scripts
└── output/                # Analysis results
```

## Adding New QBs

Edit `nfl_scraper.py` and add to the `NFL_QBS` list:

```python
NFL_QBS = [
    ('AlleJo02', 'Josh Allen', 1996),      # Example
    ('NewQB00', 'New Player', 1998),       # Add here
]
```

To find a player's ID:
1. Go to https://www.pro-football-reference.com/
2. Search for the player
3. Copy the ID from URL: `/players/A/AlleJo02.htm` → `AlleJo02`

## Data Columns

| Column | Description |
|--------|-------------|
| Season | Year of the season |
| Age | Player's age that season |
| Team | Team abbreviation (changes tracked!) |
| G/GS | Games played / started |
| Cmp/Att | Completions / Attempts |
| Yds | Passing yards |
| TD/Int | Touchdowns / Interceptions |
| Rate | Passer rating |
| QBR | ESPN's QBR |

## Data Sources

- [Pro Football Reference](https://www.pro-football-reference.com/)

