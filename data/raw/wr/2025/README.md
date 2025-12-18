---
dataset_info:
  features:
  - name: game_id
    dtype: string
  - name: receiver_player_id
    dtype: string
  - name: receiver_player_name
    dtype: string
  - name: passer_player_id
    dtype: string
  - name: defteam
    dtype: int64
  - name: targets
    dtype: float64
  - name: receptions
    dtype: float64
  - name: receiving_yards
    dtype: float64
  - name: air_yards
    dtype: float64
  - name: yac
    dtype: float64
  - name: tds
    dtype: float64
  - name: epa
    dtype: float64
  - name: wpa
    dtype: float64
  - name: avg_depth
    dtype: float64
  - name: catch_rate
    dtype: float64
  - name: posteam
    dtype: int64
  - name: team_pass_attempts
    dtype: float64
  - name: team_air_yards
    dtype: float64
  - name: team_epa
    dtype: float64
  - name: air_yard_share
    dtype: float64
  - name: target_share
    dtype: float64
  - name: yards_per_target
    dtype: float64
  - name: def_targets_dev
    dtype: float64
  - name: def_receptions_dev
    dtype: float64
  - name: def_yards_dev
    dtype: float64
  - name: def_tds_dev
    dtype: float64
  - name: def_epa_dev
    dtype: float64
  - name: red_zone_targets
    dtype: int64
  - name: end_zone_targets
    dtype: int64
  - name: third_down_targets
    dtype: int64
  - name: fourth_down_targets
    dtype: int64
  - name: high_leverage_targets
    dtype: int64
  - name: qb_completions
    dtype: float64
  - name: qb_attempts
    dtype: float64
  - name: qb_air_yards
    dtype: float64
  - name: qb_cpoe
    dtype: float64
  - name: qb_comp_pct
    dtype: float64
  - name: avg_score_diff
    dtype: float64
  - name: avg_quarter
    dtype: float64
  - name: adot
    dtype: float64
  - name: yac_per_reception
    dtype: float64
  - name: td_rate
    dtype: float64
  - name: explosive_plays
    dtype: float64
  - name: first_downs
    dtype: float64
  - name: trailing_pct
    dtype: float64
  - name: leading_pct
    dtype: float64
  - name: wp_var
    dtype: float64
  - name: target_share_std
    dtype: float64
  - name: surface
    dtype: int64
  - name: is_dome
    dtype: int64
  - name: is_rain
    dtype: int64
  - name: is_snow
    dtype: int64
  - name: is_clear
    dtype: int64
  - name: temp_f
    dtype: float64
  - name: humidity_pct
    dtype: float64
  - name: wind_mph
    dtype: float64
  - name: success_rate
    dtype: float64
  - name: big_play_rate
    dtype: float64
  - name: reception_std
    dtype: float64
  - name: second_and_long_targets
    dtype: float64
  - name: third_and_medium_targets
    dtype: float64
  - name: avg_start_yardline
    dtype: float64
  - name: avg_target_depth_vs_qb
    dtype: float64
  - name: yards_Q1
    dtype: float64
  - name: yards_Q2
    dtype: float64
  - name: yards_Q3
    dtype: float64
  - name: yards_Q4
    dtype: float64
  - name: receptions_Q1
    dtype: float64
  - name: receptions_Q2
    dtype: float64
  - name: receptions_Q3
    dtype: float64
  - name: receptions_Q4
    dtype: float64
  - name: targets_Q1
    dtype: float64
  - name: targets_Q2
    dtype: float64
  - name: targets_Q3
    dtype: float64
  - name: targets_Q4
    dtype: float64
  - name: lost_yards_due_to_penalty
    dtype: float64
  - name: yards_wp_<25
    dtype: float64
  - name: yards_wp_25_45
    dtype: float64
  - name: yards_wp_45_55
    dtype: float64
  - name: yards_wp_55_75
    dtype: float64
  - name: yards_wp_>75
    dtype: float64
  - name: yards_wp_NA
    dtype: int64
  - name: receptions_wp_<25
    dtype: float64
  - name: receptions_wp_25_45
    dtype: float64
  - name: receptions_wp_45_55
    dtype: float64
  - name: receptions_wp_55_75
    dtype: float64
  - name: receptions_wp_>75
    dtype: float64
  - name: receptions_wp_NA
    dtype: int64
  - name: targets_wp_<25
    dtype: int64
  - name: targets_wp_25_45
    dtype: int64
  - name: targets_wp_45_55
    dtype: int64
  - name: targets_wp_55_75
    dtype: int64
  - name: targets_wp_>75
    dtype: int64
  - name: targets_wp_NA
    dtype: int64
  - name: home_team
    dtype: int64
  - name: away_team
    dtype: int64
  - name: pregame_spread
    dtype: float64
  - name: pregame_total
    dtype: float64
  splits:
  - name: full
    num_bytes: 1555639
    num_examples: 1918
  download_size: 294512
  dataset_size: 1555639
configs:
- config_name: default
  data_files:
  - split: full
    path: data/full-*
license: mit
language:
- en
---

# NFL Wide Receiver Performance Dataset (2025)

## Dataset Description

This dataset contains comprehensive wide receiver performance statistics derived from NFL play-by-play data for the **2025**. It includes game-level metrics, situational targeting patterns, defensive adjustments, and advanced efficiency calculations.

### Dataset Summary

- **Season**: 2025
- **Records**: Player-game observations
- **Features**: 100+ columns including:
  - Core statistics (targets, receptions, yards, touchdowns)
  - Quarter-by-quarter breakdowns
  - Win probability bucketed performance
  - Defensive strength adjustments
  - Situational metrics (red zone, high leverage, down-and-distance)
  - Team share metrics (target share, air yards share, WOPR)
  - Efficiency metrics (aDOT, yards per target, catch rate)
  - Weather and venue conditions

### Supported Tasks

- **Receiving Yards Prediction**: Predict receiving yards for upcoming games
- **Target Share Analysis**: Model player opportunity distribution
- **Performance Forecasting**: Project future player performance
- **Matchup Analysis**: Evaluate player-defense matchups

## Dataset Structure

### Data Fields

**Key Identifiers:**
- `game_id`: Unique game identifier
- `receiver_player_id`: NFL GSIS player ID
- `receiver_player_name`: Player display name
- `passer_player_id`: Quarterback player ID
- `season`: NFL season year
- `week`: Week number

**Core Statistics:**
- `targets`: Total pass attempts targeting the receiver
- `receptions`: Completed receptions
- `receiving_yards`: Total receiving yards
- `tds`: Receiving touchdowns
- `air_yards`: Total air yards on targets
- `yac`: Yards after catch

**Quarter Breakdowns:**
- `yards_Q1`, `yards_Q2`, `yards_Q3`, `yards_Q4`: Yards by quarter
- `receptions_Q1-4`: Receptions by quarter
- `targets_Q1-4`: Targets by quarter

**Win Probability Buckets:**
- `yards_wp_<25`, `yards_wp_25_45`, etc.: Performance in different game situations
- Similar breakdowns for receptions and targets

**Share Metrics:**
- `target_share`: Player's share of team targets
- `air_yards_share`: Player's share of team air yards
- `yard_share`: Player's share of team receiving yards
- `reception_share`: Player's share of team receptions
- `wopr`: Weighted Opportunity Rating (0.7 × target_share + 0.3 × air_yards_share)

**Efficiency Metrics:**
- `aDOT`: Average depth of target
- `yards_per_target`: Receiving yards per target
- `catch_rate`: Reception percentage
- `yac_per_rec`: Yards after catch per reception
- `explosive_rec_pct`: Percentage of receptions ≥15 yards
- `first_down_pct`: Percentage of receptions resulting in first downs

**Defensive Adjustments:**
- `def_targets_dev`: Defense targets allowed vs league average
- `def_receptions_dev`: Defense receptions allowed vs league average
- `def_yards_dev`: Defense yards allowed vs league average
- `def_tds_dev`: Defense TDs allowed vs league average
- `def_epa_dev`: Defense EPA allowed vs league average
- `adj_epa`: Defense-adjusted Expected Points Added
- `adj_epa_per_target`: Defense-adjusted EPA per target

**Situational Metrics:**
- `red_zone_targets`: Targets inside the 20-yard line
- `end_zone_targets`: Targets in the end zone
- `third_down_targets`: Targets on 3rd down
- `fourth_down_targets`: Targets on 4th down
- `high_leverage_targets`: Targets in high-leverage situations (WP < 0.25 or > 0.75)
- `red_zone_share`: Player's share of team red zone targets
- `third_down_share`: Player's share of team 3rd down targets

**Game Context:**
- `posteam`: Player's team (encoded 1-32)
- `defteam`: Opposing defense (encoded 1-32)
- `home_team`: Home team (encoded 1-32)
- `away_team`: Away team (encoded 1-32)
- `home_flag`: 0 if home, 1 if away
- `pregame_spread`: Betting line point spread
- `pregame_total`: Betting line total points
- `avg_score_diff`: Average score differential when targeted
- `avg_quarter`: Average quarter when targeted
- `trailing_pct`: Percentage of targets while trailing
- `leading_pct`: Percentage of targets while leading

**Weather & Venue:**
- `surface`: Playing surface type (encoded 0-6)
- `is_dome`: 1 if indoor, 0 if outdoor
- `is_rain`: 1 if rainy conditions
- `is_snow`: 1 if snowy conditions
- `is_clear`: 1 if clear conditions
- `temp_f`: Temperature in Fahrenheit
- `humidity_pct`: Humidity percentage
- `wind_mph`: Wind speed in miles per hour

**QB Context:**
- `qb_completions`: Quarterback's completions that game
- `qb_attempts`: Quarterback's attempts that game
- `qb_comp_pct`: Quarterback's completion percentage
- `qb_air_yards`: Quarterback's average air yards
- `qb_cpoe`: Quarterback's completion percentage over expected

**Advanced Metrics:**
- `epa`: Expected Points Added
- `wpa`: Win Probability Added
- `success_rate`: Percentage of successful plays (EPA > 0 or YPT > 0.5)
- `big_play_rate`: Percentage of plays ≥20 yards
- `explosive_plays`: Count of plays ≥20 yards
- `first_downs`: First downs generated
- `consistency_score`: mean_adj_epa / std_adj_epa
- `inverse_volatility`: 1 / std_adj_epa
- `season_adj_epa_per_target`: Season-level defense-adjusted EPA per target
- `wp_var`: Variance in win probability across targets
- `target_share_std`: Standard deviation of target share across games

### Data Splits

This dataset does not include pre-defined splits. Users should create their own train/validation/test splits based on their use case:
- **Time-based split**: Use early weeks for training, later weeks for validation/testing
- **Cross-validation**: K-fold cross-validation across games
- **Season holdout**: Train on this season, test on future seasons

### Dataset Creation

#### Source Data

Raw play-by-play data sourced from [nflverse](https://github.com/nflverse/nflverse-data/releases/tag/pbp), which aggregates official NFL data with additional features.

#### Data Processing

The dataset was created through two complementary processing pipelines:

1. **Pipeline A (Defensive Adjustments)**: 
   - Calculates defense-adjusted performance metrics
   - Adds situational targeting patterns
   - Includes QB context and team-level statistics
   - Incorporates weather and venue conditions

2. **Pipeline B (Temporal & Situational)**: 
   - Generates quarter-by-quarter breakdowns
   - Creates win probability bucketed statistics
   - Computes team share metrics and WOPR
   - Calculates season-level consistency metrics

Both pipelines were merged to create a comprehensive feature set.

## Considerations for Using the Data

### Social Impact

This dataset is intended for:
- Sports analytics and research
- Fantasy football decision-making
- Educational purposes in machine learning and sports statistics

**Not intended for:**
- Real-money gambling (use responsibly)
- Player evaluation for contract negotiations
- Any decision-making that could impact player careers

### Discussion of Biases

- **Opportunity bias**: Statistics heavily dependent on team offensive scheme and QB quality
- **Injury data**: Dataset does not account for injuries that may affect performance
- **Sample size**: Players with limited playing time have less reliable statistics
- **Game script**: Performance metrics influenced by whether team is winning/losing
- **Strength of schedule**: Not all defensive matchups are equal, though some adjustment is included

### Limitations

- **Historical data only**: Does not predict future performance definitively
- **Missing context**: Does not include play design, route running, or other qualitative factors
- **Weather parsing**: Temperature/wind/humidity may be missing or inaccurate for some games
- **Roster changes**: Does not account for mid-season team changes or trades
- **Playoff games**: May or may not include playoff data depending on the year

## Additional Information

### Acknowledgments

- **Data Source**: [nflverse/nflverse-data](https://github.com/nflverse/nflverse-data/releases/tag/pbp)
- **AI Assistance**: Code development assisted by Claude (Anthropic)
- **Course**: CMU 24-679: Designing and Deploying AI/ML Systems