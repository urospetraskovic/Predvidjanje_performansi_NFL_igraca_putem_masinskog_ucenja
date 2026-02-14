# NFL Player Performance Prediction

**Predviđanje Performansi NFL Igrača korišćenjem Mašinskog Učenja**

**Authors:** Milan Jovkić R2 10/2025, Uroš Petrašković R2 9/2025

## 📋 Project Overview

This project develops machine learning models to predict future NFL player performance using historical statistical data. The system provides position-specific predictions for:

- **Quarterbacks (QB):** Passing Yards
- **Running Backs (RB):** Rushing Yards
- **Wide Receivers (WR):** Receiving Yards
- **Tight Ends (TE):** Receiving Yards

## 🎯 Motivation

NFL teams operate under a salary cap, making efficient budget management crucial for team success. This project helps:
- NFL teams identify high-potential players
- Fantasy football participants make better decisions
- Sports analysts understand performance predictors

## 🔬 Methodology

Based on research from:
1. "Advancing NFL win prediction: from Pythagorean formulas to machine learning algorithms" - Frontiers in Sports and Active Living (2025)
2. Elimam et al. (2025) - "Multi-Output Regression for the Prediction of World-Class Performances"
3. Abadzic et al. (2024) - "Data Analysis on Predicting the Top 12 Fantasy Football Players"

### Algorithms Implemented:
- Linear Regression (baseline)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- ElasticNet (L1 + L2)
- K-Nearest Neighbors Regressor
- **Random Forest Regressor** (primary model)
- Gradient Boosting Regressor
- Multi-Output Regression

### Evaluation Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full analysis
python nfl_performance_prediction.py

# Or use the Jupyter notebook
jupyter notebook NFL_Performance_Prediction.ipynb
```

## 📁 Project Structure

```
├── nfl_performance_prediction.py   # Main analysis script
├── NFL_Performance_Prediction.ipynb # Interactive Jupyter notebook
├── requirements.txt                # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data preprocessing utilities
│   ├── models.py                   # ML model implementations
│   └── evaluation.py               # Evaluation and visualization
├── data/
│   ├── raw/                        # Raw scraped data
│   │   ├── qb/                     # Quarterback stats
│   │   ├── rb/                     # Running back stats
│   │   ├── te/                     # Tight end stats
│   │   └── wr/                     # Wide receiver stats (HuggingFace)
│   └── processed/                  # Combined/processed CSV files
├── models/                         # Saved trained models
├── results/                        # Analysis results and plots
└── README.md
```

## 📊 Data Sources

- **Pro Football Reference** - QB, RB, TE statistics (scraped)
- **HuggingFace** - `SebastianAndreu/24679_NFL_WR_Dataset_2025` (WR data 2015-2025)

## 🔧 Scraping Data

```bash
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

