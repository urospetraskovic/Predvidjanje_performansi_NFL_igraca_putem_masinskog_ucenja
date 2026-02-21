"""
NFL Player Performance Prediction - Main Analysis Script

This script performs comprehensive analysis for predicting NFL player performance:
1. Data Loading and Preprocessing
2. Position-Specific Model Training (QB, RB, WR, TE)
3. Multiple Algorithm Comparison (Linear, Ridge, Lasso, ElasticNet, KNN, RF, GB)
4. Feature Importance Analysis (SHAP values)
5. Multi-Output Regression
6. Full Evaluation and Visualization (MAE, RMSE, R²)

Authors: Milan Jovkić R2 10/2025, Uroš Petrašković R2 9/2025
Based on methodology from:
- Frontiers in Sports and Active Living (2025) - NFL Win Prediction
- Elimam et al. (2025) - Multi-Output Regression for Performance Prediction
- Abadzic et al. (2024) - Fantasy Football Prediction
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import NFLDataPreprocessor, create_time_based_split
from src.models import (NFLPerformanceModel, MultiOutputNFLModel, 
                         compare_models, train_position_specific_model)
from src.evaluation import (ModelEvaluator, plot_feature_importance, 
                             plot_model_comparison, plot_learning_curve,
                             create_position_comparison_dashboard,
                             generate_full_report, compute_shap_values)

# Configuration
DATA_DIR = 'data/processed'
RAW_WR_DIR = 'data/raw/wr'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'
RANDOM_STATE = 42

# Create output directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data():
    """Load all position-specific datasets."""
    print("\n" + "="*60)
    print("LOADING NFL PLAYER STATISTICS")
    print("="*60)
    
    data = {}
    
    # --- Load QB data ---
    try:
        data['qb'] = pd.read_csv(os.path.join(DATA_DIR, 'all_qb_stats_full.csv'))
        print(f"  QB: {len(data['qb'])} player-season records, "
              f"{data['qb']['Season'].nunique()} seasons "
              f"({data['qb']['Season'].min()}-{data['qb']['Season'].max()})")
    except FileNotFoundError:
        print("  QB data not found.")
        data['qb'] = None
    
    # --- Load RB data (using all_rb_ prefix to avoid QB collision) ---
    try:
        rb_basic = pd.read_csv(os.path.join(DATA_DIR, 'all_rb_rushing_receiving.csv'))
        rb_advanced = pd.read_csv(os.path.join(DATA_DIR, 'all_rb_advanced_rushing_receiving.csv'))
        
        # Merge basic + advanced on (Season, Player, PlayerID)
        merge_cols = ['Season', 'Player', 'PlayerID']
        # Drop columns from advanced that already exist in basic (except merge keys)
        adv_only_cols = [c for c in rb_advanced.columns if c not in rb_basic.columns or c in merge_cols]
        rb_advanced_clean = rb_advanced[adv_only_cols]
        data['rb'] = rb_basic.merge(rb_advanced_clean, on=merge_cols, how='left')
        print(f"  RB: {len(data['rb'])} player-season records, "
              f"{data['rb']['Season'].nunique()} seasons "
              f"({data['rb']['Season'].min()}-{data['rb']['Season'].max()})")
    except FileNotFoundError:
        print("  RB data not found. Run the RB data combiner first.")
        data['rb'] = None
    
    # --- Load TE data ---
    try:
        te_basic = pd.read_csv(os.path.join(DATA_DIR, 'all_te_receiving_rushing.csv'))
        te_advanced = pd.read_csv(os.path.join(DATA_DIR, 'all_te_advanced_receiving_rushing.csv'))
        
        merge_cols = ['Season', 'Player', 'PlayerID']
        adv_only_cols = [c for c in te_advanced.columns if c not in te_basic.columns or c in merge_cols]
        te_advanced_clean = te_advanced[adv_only_cols]
        data['te'] = te_basic.merge(te_advanced_clean, on=merge_cols, how='left')
        print(f"  TE: {len(data['te'])} player-season records, "
              f"{data['te']['Season'].nunique()} seasons "
              f"({data['te']['Season'].min()}-{data['te']['Season'].max()})")
    except FileNotFoundError:
        print("  TE data not found.")
        data['te'] = None
    
    # --- Load WR data (game-level -> aggregate to season-level) ---
    try:
        wr_games = []
        for year in range(2015, 2026):
            year_file = os.path.join(RAW_WR_DIR, str(year), 'data', f'wr_{year}.csv')
            if os.path.exists(year_file):
                df = pd.read_csv(year_file)
                df['Season'] = year
                wr_games.append(df)
        
        if wr_games:
            wr_all_games = pd.concat(wr_games, ignore_index=True)
            data['wr'] = _aggregate_wr_to_season(wr_all_games)
            print(f"  WR: {len(data['wr'])} player-season records from {len(wr_games)} seasons "
                  f"({data['wr']['Season'].min()}-{data['wr']['Season'].max()})")
        else:
            data['wr'] = None
            print("  WR: No data files found in data/raw/wr/")
    except Exception as e:
        print(f"  WR data error: {e}")
        data['wr'] = None
    
    return data


def _aggregate_wr_to_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate WR game-level data to season-level per player.
    
    The WR dataset has one row per receiver per game (98 columns);
    we aggregate to season totals/averages for modeling.
    
    Note: receiving_yards, air_yards, yac, first_downs are summed for the
    DataFrame (target + EDA) but are NOT used as prediction features to
    prevent data leakage.
    """
    # --- Columns to SUM (season totals) ---
    sum_cols = [
        # Target variable + EDA-only columns (excluded from features later)
        'receiving_yards', 'air_yards', 'yac', 'first_downs',
        # Core counting stats (used as features)
        'targets', 'receptions', 'tds', 'explosive_plays',
        # Situational targets
        'red_zone_targets', 'end_zone_targets', 'third_down_targets',
        'fourth_down_targets', 'high_leverage_targets',
        'second_and_long_targets', 'third_and_medium_targets',
        # Quarter breakdowns — receptions & targets only (NOT yards_Q*)
        'receptions_Q1', 'receptions_Q2', 'receptions_Q3', 'receptions_Q4',
        'targets_Q1', 'targets_Q2', 'targets_Q3', 'targets_Q4',
        # Win-probability breakdowns — receptions & targets only (NOT yards_wp_*)
        'receptions_wp_<25', 'receptions_wp_25_45', 'receptions_wp_45_55',
        'receptions_wp_55_75', 'receptions_wp_>75',
        'targets_wp_<25', 'targets_wp_25_45', 'targets_wp_45_55',
        'targets_wp_55_75', 'targets_wp_>75',
    ]
    
    # --- Columns to AVERAGE (season means) ---
    mean_cols = [
        # Efficiency / rate metrics
        'catch_rate', 'success_rate', 'big_play_rate', 'td_rate',
        'epa', 'wpa',
        # Team share & usage
        'target_share', 'air_yard_share',
        # QB context
        'qb_comp_pct', 'qb_cpoe', 'qb_air_yards',
        'qb_completions', 'qb_attempts',
        # Route / depth (play design, not outcome)
        'adot', 'avg_depth',
        # Game situation
        'avg_score_diff', 'avg_quarter', 'leading_pct',
        'wp_var', 'target_share_std', 'reception_std',
        'avg_start_yardline', 'avg_target_depth_vs_qb',
        # Team passing context
        'team_pass_attempts', 'team_air_yards', 'team_epa',
        # Defensive matchup quality
        'def_targets_dev', 'def_receptions_dev', 'def_yards_dev',
        'def_tds_dev', 'def_epa_dev',
        # Weather & venue
        'temp_f', 'humidity_pct', 'wind_mph',
        'is_dome', 'is_rain', 'is_clear', 'surface',
        # Betting lines (game context)
        'pregame_spread', 'pregame_total',
    ]
    
    agg_dict = {}
    for col in sum_cols:
        if col in df.columns:
            agg_dict[col] = 'sum'
    for col in mean_cols:
        if col in df.columns:
            agg_dict[col] = 'mean'
    
    if 'receiver_player_name' not in df.columns:
        print("  Warning: WR data missing 'receiver_player_name' column.")
        return df
    
    season_df = df.groupby(['receiver_player_name', 'Season']).agg(agg_dict).reset_index()
    games = df.groupby(['receiver_player_name', 'Season']).size().reset_index(name='games')
    season_df = season_df.merge(games, on=['receiver_player_name', 'Season'])
    
    # Non-leaky per-game derived stats
    if 'receptions' in season_df.columns:
        season_df['receptions_per_game'] = season_df['receptions'] / season_df['games']
    if 'targets' in season_df.columns:
        season_df['targets_per_game'] = season_df['targets'] / season_df['games']
    if 'tds' in season_df.columns:
        season_df['tds_per_game'] = season_df['tds'] / season_df['games']
    
    return season_df


# =============================================================================
# COMMON PREPROCESSING PIPELINE
# =============================================================================

def _preprocess_position_data(df: pd.DataFrame, feature_cols: list, target_col: str,
                               min_games: int = 4, test_seasons: int = 2):
    """
    Common preprocessing pipeline for any position.
    
    Creates a fresh preprocessor per position (separate scaler/encoders),
    does chronological train/test split, and applies consistent preprocessing.
    
    Returns:
        X_train, y_train, X_test, y_test, preprocessor
    """
    # Create fresh preprocessor for this position
    preprocessor = NFLDataPreprocessor(scaler_type='standard')
    
    # Filter: minimum games + valid target
    df_filtered = df[(df['G'] >= min_games) & (df[target_col].notna())].copy()
    print(f"Records after filtering (G>={min_games}, valid {target_col}): {len(df_filtered)}")
    
    if len(df_filtered) < 10:
        print("Insufficient data for analysis.")
        return None, None, None, None, None
    
    # Chronological train/test split (prevents data leakage)
    train_df, test_df = create_time_based_split(df_filtered, 'Season', test_seasons=test_seasons)
    
    if len(train_df) < 5 or len(test_df) < 3:
        print("Insufficient data after chronological split.")
        return None, None, None, None, None
    
    # Get features available in the data
    available_features = [col for col in feature_cols if col in train_df.columns]
    print(f"Using {len(available_features)}/{len(feature_cols)} specified features")
    
    # Preprocess training data (fit=True: fits scaler, encoders, imputer)
    X_train, y_train = preprocessor.full_preprocessing_pipeline(
        train_df, available_features, target_col, 
        treat_outliers=True, scale=True, fit=True
    )
    
    # Preprocess test data (fit=False: reuses fitted transformers)
    X_test, y_test = preprocessor.full_preprocessing_pipeline(
        test_df, available_features, target_col,
        treat_outliers=True, scale=True, fit=False
    )
    
    # Fill any remaining NaN with training medians
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)
    
    # Align features (same columns in same order)
    common_features = sorted(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set:     {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_train, y_train, X_test, y_test, preprocessor


def _run_position_analysis(position: str, df: pd.DataFrame,
                            feature_cols: list, target_col: str,
                            target_description: str):
    """
    Complete analysis pipeline for one position:
    1. Preprocess data with chronological split
    2. Compare all 7 algorithms
    3. Train Random Forest with hyperparameter tuning
    4. Tree-based feature importance
    5. SHAP feature importance analysis
    6. Full evaluation report (plots + metrics)
    7. Learning curve analysis
    8. Save trained model
    
    Returns:
        best_model, test_metrics
    """
    print("\n" + "="*60)
    print(f"{position.upper()} PERFORMANCE ANALYSIS")
    print(f"Target Variable: {target_description}")
    print("="*60)
    
    if df is None or len(df) == 0:
        print(f"No {position} data available.")
        return None, None
    
    result = _preprocess_position_data(df, feature_cols, target_col)
    X_train, y_train, X_test, y_test, pos_preprocessor = result
    
    if X_train is None:
        return None, None
    
    # ----------------------------------------------------------------
    # Step 1: Compare all algorithms
    # ----------------------------------------------------------------
    print("\n--- Step 1/6: Comparing All Algorithms ---")
    comparison_results = compare_models(X_train, y_train, X_test, y_test,
                                        tune_hyperparameters=False)
    
    print(f"\n{'Model':<20} {'Test MAE':>10} {'Test RMSE':>10} {'Test R²':>10} {'CV RMSE':>10}")
    print("-" * 62)
    for _, row in comparison_results.iterrows():
        print(f"{row['model']:<20} {row['test_mae']:>10.2f} {row['test_rmse']:>10.2f} "
              f"{row['test_r2']:>10.4f} {row['cv_rmse']:>10.2f}")
    
    comparison_results.to_csv(
        os.path.join(RESULTS_DIR, f'{position}_model_comparison.csv'), index=False
    )
    plot_model_comparison(
        comparison_results,
        save_path=os.path.join(RESULTS_DIR, f'{position}_model_comparison.png')
    )
    
    # ----------------------------------------------------------------
    # Step 2: Train Random Forest with hyperparameter tuning
    # ----------------------------------------------------------------
    print("\n--- Step 2/6: Training Random Forest (Hyperparameter Tuning) ---")
    best_model = NFLPerformanceModel(model_type='random_forest')
    train_metrics = best_model.train(X_train, y_train, tune_hyperparameters=True)
    test_metrics = best_model.evaluate(X_test, y_test)
    
    print(f"\n{position} Random Forest Results:")
    print(f"  Training:  MAE={train_metrics['train_mae']:.2f}, "
          f"RMSE={train_metrics['train_rmse']:.2f}, R²={train_metrics['train_r2']:.4f}")
    print(f"  Test:      MAE={test_metrics['mae']:.2f}, "
          f"RMSE={test_metrics['rmse']:.2f}, R²={test_metrics['r2']:.4f}")
    print(f"  CV RMSE:   {train_metrics['cv_rmse_mean']:.2f} (±{train_metrics['cv_rmse_std']:.2f})")
    if best_model.best_params:
        print(f"  Best params: {best_model.best_params}")
    
    # ----------------------------------------------------------------
    # Step 3: Tree-based Feature Importance
    # ----------------------------------------------------------------
    print("\n--- Step 3/6: Feature Importance Analysis ---")
    if best_model.feature_importance is not None:
        print(f"\nTop 15 Features for {position} ({target_description}):")
        fi = best_model.feature_importance.head(15)
        for _, row in fi.iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"  {row['feature']:<25} {row['importance']:.4f} {bar}")
        
        plot_feature_importance(
            best_model.feature_importance,
            top_n=15,
            title=f"{position} - Feature Importance (Random Forest)",
            save_path=os.path.join(RESULTS_DIR, f'{position}_feature_importance.png')
        )
    
    # ----------------------------------------------------------------
    # Step 4: SHAP Analysis
    # ----------------------------------------------------------------
    print("\n--- Step 4/6: SHAP Feature Importance ---")
    shap_importance = compute_shap_values(
        best_model, X_train, X_test,
        model_name=f"{position}_{target_description.replace(' ', '_')}",
        save_dir=RESULTS_DIR
    )
    
    # ----------------------------------------------------------------
    # Step 5: Full Evaluation Report (visualizations)
    # ----------------------------------------------------------------
    print("\n--- Step 5/6: Generating Evaluation Report ---")
    generate_full_report(
        best_model, X_test, y_test,
        f"{position}_{target_description.replace(' ', '_')}",
        best_model.feature_importance, RESULTS_DIR
    )
    
    # ----------------------------------------------------------------
    # Step 6: Learning Curve
    # ----------------------------------------------------------------
    print("\n--- Step 6/6: Learning Curve Analysis ---")
    try:
        plot_learning_curve(
            best_model, X_train, y_train,
            title=f"{position} - Learning Curve (Random Forest)",
            save_path=os.path.join(RESULTS_DIR, f'{position}_learning_curve.png')
        )
    except Exception as e:
        print(f"  Learning curve failed: {e}")
    
    # Save model
    best_model.save_model(os.path.join(MODELS_DIR, f'{position}_model.joblib'))
    
    return best_model, test_metrics


# =============================================================================
# POSITION-SPECIFIC ANALYSIS FUNCTIONS
# =============================================================================

def analyze_qb_performance(df: pd.DataFrame):
    """
    Analyze and predict QB passing yards.
    
    Target: Passing Yards (Yds)
    Features: Standard passing stats + advanced passing metrics + rushing
    """
    # NOTE: Removed leaky features that are derived from passing yards (the target):
    #   1D (too correlated), Y/A, AY/A, Y/C, NY/A, ANY/A (all = f(Yds)),
    #   Rate, QBR (formulas include yards),
    #   pass_air_yds, pass_yac (air_yds + yac ≈ Yds),
    #   pass_rpo_yds, pass_rpo_pass_yds, pass_play_action_yds (subsets of Yds)
    feature_cols = [
        # Standard passing (non-leaky)
        'Age', 'G', 'GS', 'Cmp', 'Att', 'Cmp%', 'TD', 'TD%', 'Int', 'Int%',
        'Succ%', 'Lng', 'Sk', 'Yds_Lost', 'Sk%', '4QC', 'GWD',
        # Rushing (different stat from passing yards)
        'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rush_1D', 'Rush_Y/A',
        # Advanced passing — accuracy & pressure (not derived from total yards)
        'pass_drops', 'pass_drop_pct',
        'pass_poor_throws', 'pass_poor_throw_pct', 'pass_on_target',
        'pass_on_target_pct', 'pocket_time', 'pass_blitzed', 'pass_hurried',
        'pass_hits', 'pass_pressured', 'pass_pressured_pct', 'rush_scrambles',
        # Advanced rushing
        'Rush_YBC', 'Rush_YAC', 'Rush_BrkTkl',
        # RPO and play action (counts only, not yards subsets)
        'pass_rpo_plays', 'pass_rpo_pass_att',
        'pass_rpo_rush_att', 'pass_rpo_rush_yds',
        'pass_play_action_att',
        # Snap counts
        'offense', 'Off%'
    ]
    
    return _run_position_analysis('QB', df, feature_cols, 'Yds', 'Passing Yards')


def analyze_rb_performance(df: pd.DataFrame):
    """
    Analyze and predict RB rushing yards.
    
    Target: Rushing Yards (Yds)
    Features: Standard rushing/receiving + advanced contact/broken tackle metrics
    """
    # NOTE: Removed leaky features derived from rushing yards (the target):
    #   1D (too correlated), Y/A = Yds/Att, Y/G = Yds/G,
    #   yds_per_touch, yds_from_scrimmage (both include Yds),
    #   YBC, YBC/Att, YAC, YAC/Att (YBC + YAC = rushing Yds)
    feature_cols = [
        # Standard rushing (non-leaky)
        'Age', 'G', 'GS', 'Att', 'TD', 'Succ%', 'Lng', 'A/G',
        # Standard receiving (different stat from target)
        'Tgt', 'Rec', 'Yds.1', 'Y/R', 'TD.1', '1D.1', 'rec_success', 'rec_long',
        'R/G', 'Y/G.1', 'catch_pct', 'Y/Tgt',
        # Combined (without leaky yards derivatives)
        'Touch', 'rush_receive_td', 'Fmb',
        # Advanced rushing — broken tackles (not yards-derived)
        'BrkTkl', 'Att/Br',
        # Advanced receiving (yards before/after catch for RECEIVING, not rushing)
        'YBC.1', 'YBC/R', 'YAC.1', 'YAC/R', 'ADOT', 'BrkTkl.1', 'Rec/Br',
        'Drop', 'Drop%', 'rec_pass_rating'
    ]
    
    return _run_position_analysis('RB', df, feature_cols, 'Yds', 'Rushing Yards')


def analyze_wr_performance(df: pd.DataFrame):
    """
    Analyze and predict WR receiving yards.
    
    Target: Receiving Yards (receiving_yards)
    Features: 70+ non-leaky features from the full 98-column game-level dataset,
              aggregated to season level. Excludes yards-derived features
              (air_yards, yac, yards_per_target, etc.) to prevent data leakage.
    """
    if df is None or len(df) == 0:
        print("\n" + "="*60)
        print("WIDE RECEIVER (WR) PERFORMANCE ANALYSIS")
        print("No WR data available.")
        print("="*60)
        return None, None
    
    # Filter minimum 4 games before passing to pipeline
    if 'games' in df.columns:
        df = df[df['games'] >= 4].copy()
    if 'G' not in df.columns and 'games' in df.columns:
        df['G'] = df['games']
    
    if len(df) < 20:
        print("Insufficient WR data (< 20 player-seasons after filtering).")
        return None, None
    
    # NOTE: Excluded leaky features derived from receiving_yards (the target):
    #   air_yards, yac (air_yards + yac ≈ receiving_yards),
    #   yards_per_target = receiving_yards / targets,
    #   yac_per_reception = yac / receptions,
    #   first_downs (too correlated with yards),
    #   yards_per_game = receiving_yards / games,
    #   yards_Q1-Q4 (sum to receiving_yards),
    #   yards_wp_* (sum to receiving_yards)
    feature_cols = [
        # Volume (counting stats)
        'targets', 'receptions', 'tds', 'games',
        # Efficiency (rates — not derived from total receiving_yards)
        'catch_rate', 'success_rate', 'big_play_rate', 'td_rate',
        'epa', 'wpa',
        # Team share & usage
        'target_share', 'air_yard_share',
        # Route / depth (play design, not outcome)
        'adot', 'avg_depth',
        # QB context
        'qb_comp_pct', 'qb_cpoe', 'qb_air_yards',
        'qb_completions', 'qb_attempts',
        # Situational targets
        'red_zone_targets', 'end_zone_targets', 'third_down_targets',
        'fourth_down_targets', 'high_leverage_targets',
        'second_and_long_targets', 'third_and_medium_targets',
        # Impact
        'explosive_plays',
        # Game situation
        'avg_score_diff', 'avg_quarter', 'leading_pct',
        'wp_var', 'target_share_std', 'reception_std',
        'avg_start_yardline', 'avg_target_depth_vs_qb',
        # Team passing context
        'team_pass_attempts', 'team_air_yards', 'team_epa',
        # Defensive matchup quality
        'def_targets_dev', 'def_receptions_dev', 'def_yards_dev',
        'def_tds_dev', 'def_epa_dev',
        # Weather & venue
        'temp_f', 'humidity_pct', 'wind_mph',
        'is_dome', 'is_rain', 'is_clear', 'surface',
        # Betting lines (game context)
        'pregame_spread', 'pregame_total',
        # Quarter breakdowns — receptions & targets (NOT yards)
        'receptions_Q1', 'receptions_Q2', 'receptions_Q3', 'receptions_Q4',
        'targets_Q1', 'targets_Q2', 'targets_Q3', 'targets_Q4',
        # Win-probability breakdowns — receptions & targets (NOT yards)
        'receptions_wp_<25', 'receptions_wp_25_45', 'receptions_wp_45_55',
        'receptions_wp_55_75', 'receptions_wp_>75',
        'targets_wp_<25', 'targets_wp_25_45', 'targets_wp_45_55',
        'targets_wp_55_75', 'targets_wp_>75',
        # Per-game rates (non-leaky)
        'receptions_per_game', 'targets_per_game', 'tds_per_game',
    ]
    
    return _run_position_analysis('WR', df, feature_cols, 'receiving_yards', 'Receiving Yards')


def analyze_te_performance(df: pd.DataFrame):
    """
    Analyze and predict TE receiving yards.
    
    Target: Receiving Yards (Yds)
    Features: Receiving + rushing + advanced metrics (YBC, YAC, drops, etc.)
    """
    # NOTE: Removed leaky features derived from receiving yards (the target):
    #   1D (too correlated), Y/R = Yds/Rec, Y/G = Yds/G, Y/Tgt = Yds/Tgt,
    #   yds_per_touch, yds_from_scrimmage (both include Yds),
    #   YBC, YBC/R, YAC, YAC/R (receiving YBC + YAC = receiving Yds),
    #   rec_pass_rating (formula includes yards)
    feature_cols = [
        # Standard receiving (non-leaky)
        'Age', 'G', 'GS', 'Tgt', 'Rec', 'TD', 'rec_success',
        'rec_long', 'R/G', 'catch_pct',
        # Combined (without leaky yards derivatives)
        'Touch',
        # Rushing (different stat from receiving yards target)
        'Att', 'Yds.1', 'TD.1', '1D.1', 'Succ%', 'Lng', 'Y/A',
        # Advanced receiving — route depth & skill (not total-yards-derived)
        'ADOT', 'BrkTkl', 'Rec/Br',
        'Drop', 'Drop%',
        # Advanced rushing (rushing yards before/after contact — different stat)
        'YBC.1', 'YBC/Att', 'YAC.1', 'YAC/Att', 'BrkTkl.1', 'Att/Br'
    ]
    
    return _run_position_analysis('TE', df, feature_cols, 'Yds', 'Receiving Yards')


# =============================================================================
# MULTI-OUTPUT REGRESSION
# =============================================================================

def run_multi_output_analysis(df: pd.DataFrame, position: str):
    """
    Run multi-output regression to predict multiple metrics simultaneously.
    Based on Elimam et al. (2025) methodology.
    
    Multi-output regression predicts correlated targets together,
    which can improve accuracy over separate single-output models.
    Compares kNN and Random Forest in multi-output setting.
    """
    print("\n" + "="*60)
    print(f"MULTI-OUTPUT REGRESSION: {position}")
    print("Predicting multiple performance metrics simultaneously")
    print("="*60)
    
    if df is None or len(df) == 0:
        print(f"No {position} data available.")
        return None, None
    
    # Define targets and features by position (leaky features removed)
    configs = {
        'QB': {
            'targets': ['Yds', 'TD', 'Rate'],
            'target_names': ['Passing Yards', 'Touchdowns', 'Passer Rating'],
            'features': ['Age', 'G', 'GS', 'Cmp', 'Att', 'Cmp%', 'Int', 'Int%',
                         'Succ%', 'Sk', 'Sk%', 'Rush_Att', 'Rush_Yds']
        },
        'RB': {
            'targets': ['Yds', 'TD', 'Yds.1'],
            'target_names': ['Rushing Yards', 'Rushing TDs', 'Receiving Yards'],
            'features': ['Age', 'G', 'GS', 'Att', 'Succ%',
                         'Tgt', 'Rec', 'Touch', 'BrkTkl', 'Fmb']
        },
        'TE': {
            'targets': ['Yds', 'TD', 'Rec'],
            'target_names': ['Receiving Yards', 'Touchdowns', 'Receptions'],
            'features': ['Age', 'G', 'GS', 'Tgt', 'catch_pct',
                         'Touch', 'ADOT', 'Drop', 'Drop%', 'BrkTkl']
        }
    }
    
    if position not in configs:
        print(f"Multi-output not configured for {position}.")
        return None, None
    
    config = configs[position]
    available_targets = [c for c in config['targets'] if c in df.columns]
    available_features = [c for c in config['features'] if c in df.columns]
    
    if len(available_targets) < 2:
        print("Not enough target columns for multi-output regression.")
        return None, None
    
    target_name_map = dict(zip(config['targets'], config['target_names']))
    print(f"Targets: {[target_name_map.get(t, t) for t in available_targets]}")
    print(f"Features ({len(available_features)}): {available_features}")
    
    # Prepare data
    df_filtered = df.dropna(subset=available_targets + available_features)
    df_filtered = df_filtered[df_filtered['G'] >= 4]
    
    if len(df_filtered) < 10:
        print("Insufficient data for multi-output analysis.")
        return None, None
    
    # Chronological split
    train_df, test_df = create_time_based_split(df_filtered, 'Season', test_seasons=2)
    
    train_medians = train_df[available_features].median()
    X_train = train_df[available_features].fillna(train_medians)
    y_train = train_df[available_targets]
    X_test = test_df[available_features].fillna(train_medians)
    y_test = test_df[available_targets]
    
    print(f"Training: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Compare multi-output models (kNN vs Random Forest, as in Elimam et al.)
    model_types = ['knn', 'random_forest']
    results_comparison = []
    
    for model_type in model_types:
        print(f"\n--- Multi-Output {model_type.replace('_', ' ').title()} ---")
        mo_model = MultiOutputNFLModel(base_model_type=model_type)
        mo_model.train(X_train, y_train)
        mo_test_metrics = mo_model.evaluate(X_test, y_test)
        
        results_comparison.append({
            'approach': f'Multi-Output {model_type}',
            'avg_rmse': mo_test_metrics['average']['rmse'],
            'avg_r2': mo_test_metrics['average']['r2'],
            'avg_mae': mo_test_metrics['average']['mae']
        })
        
        for target in available_targets:
            if target in mo_test_metrics:
                m = mo_test_metrics[target]
                name = target_name_map.get(target, target)
                print(f"  {name:<20} RMSE={m['rmse']:.2f}, R²={m['r2']:.4f}, MAE={m['mae']:.2f}")
        print(f"  {'Average':<20} RMSE={mo_test_metrics['average']['rmse']:.2f}, "
              f"R²={mo_test_metrics['average']['r2']:.4f}")
    
    # Save comparison
    comp_df = pd.DataFrame(results_comparison).sort_values('avg_rmse')
    print(f"\n--- Multi-Output Model Comparison ({position}) ---")
    print(comp_df.to_string(index=False))
    comp_df.to_csv(os.path.join(RESULTS_DIR, f'{position}_multi_output_comparison.csv'), 
                   index=False)
    
    # Return best (Random Forest)
    best_mo = MultiOutputNFLModel(base_model_type='random_forest')
    best_mo.train(X_train, y_train)
    best_metrics = best_mo.evaluate(X_test, y_test)
    
    return best_mo, best_metrics


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main entry point for NFL Performance Prediction analysis."""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#     NFL PLAYER PERFORMANCE PREDICTION SYSTEM           #")
    print("#     Using Machine Learning                              #")
    print("#" + " "*58 + "#")
    print("#"*60)
    print("\nAuthors: Milan Jovkić R2 10/2025, Uroš Petrašković R2 9/2025")
    print("\nMethodology:")
    print("  - Position-specific modeling (QB, RB, WR, TE)")
    print("  - Algorithm comparison: Linear, Ridge, Lasso, ElasticNet, KNN, RF, GB")
    print("  - Primary model: Random Forest Regression")
    print("  - Feature importance: SHAP values + Tree-based importance")
    print("  - Multi-output regression (Elimam et al. 2025)")
    print("  - Evaluation metrics: MAE, RMSE, R²")
    print("  - Chronological train/test split (no data leakage)")
    
    # Load all position data
    data = load_all_data()
    
    # Store results for cross-position comparison
    all_results = {}
    
    # =====================================================================
    # 1. QUARTERBACK - Target: Passing Yards
    # =====================================================================
    qb_model, qb_metrics = analyze_qb_performance(data['qb'])
    if qb_metrics:
        all_results['QB'] = qb_metrics
    
    # =====================================================================
    # 2. RUNNING BACK - Target: Rushing Yards
    # =====================================================================
    rb_model, rb_metrics = analyze_rb_performance(data['rb'])
    if rb_metrics:
        all_results['RB'] = rb_metrics
    
    # =====================================================================
    # 3. WIDE RECEIVER - Target: Receiving Yards
    # =====================================================================
    wr_model, wr_metrics = analyze_wr_performance(data['wr'])
    if wr_metrics:
        all_results['WR'] = wr_metrics
    
    # =====================================================================
    # 4. TIGHT END - Target: Receiving Yards
    # =====================================================================
    te_model, te_metrics = analyze_te_performance(data['te'])
    if te_metrics:
        all_results['TE'] = te_metrics
    
    # =====================================================================
    # 5. MULTI-OUTPUT REGRESSION (Elimam et al. 2025)
    # =====================================================================
    print("\n\n" + "#"*60)
    print("MULTI-OUTPUT REGRESSION ANALYSIS")
    print("#"*60)
    
    for position in ['QB', 'RB', 'TE']:
        if data.get(position.lower()) is not None:
            run_multi_output_analysis(data[position.lower()], position)
    
    # =====================================================================
    # 6. CROSS-POSITION COMPARISON DASHBOARD
    # =====================================================================
    if all_results:
        print("\n" + "="*60)
        print("OVERALL POSITION COMPARISON")
        print("="*60)
        
        print(f"\n{'Position':<10} {'Target':<20} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
        print("-" * 62)
        target_map = {'QB': 'Passing Yards', 'RB': 'Rushing Yards', 
                      'WR': 'Receiving Yards', 'TE': 'Receiving Yards'}
        for pos, metrics in all_results.items():
            mae = metrics.get('mae', metrics.get('MAE', 0))
            rmse = metrics.get('rmse', metrics.get('RMSE', 0))
            r2 = metrics.get('r2', metrics.get('R2', 0))
            print(f"{pos:<10} {target_map.get(pos, 'Yards'):<20} "
                  f"{mae:>10.2f} {rmse:>10.2f} {r2:>10.4f}")
        
        create_position_comparison_dashboard(
            all_results,
            save_path=os.path.join(RESULTS_DIR, 'position_comparison.png')
        )
        
        summary_df = pd.DataFrame([
            {'Position': pos, 'Target': target_map.get(pos, 'Yards'), **metrics}
            for pos, metrics in all_results.items()
        ])
        summary_df.to_csv(os.path.join(RESULTS_DIR, 'overall_summary.csv'), index=False)
        print("\nSummary saved to results/overall_summary.csv")
    
    # =====================================================================
    # COMPLETION
    # =====================================================================
    print("\n" + "#"*60)
    print("#     ANALYSIS COMPLETE                                   #")
    print("#"*60)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Models saved to:  {MODELS_DIR}/")
    print("\nKey outputs generated:")
    print("  Per position (QB, RB, WR, TE):")
    print("    *_model_comparison.csv/png   - 7-algorithm comparison")
    print("    *_feature_importance.png     - Tree-based feature importance")
    print("    *_shap_summary.png           - SHAP beeswarm plot")
    print("    *_shap_bar.png               - SHAP mean |value| bar plot")
    print("    *_predictions.png            - Actual vs Predicted scatter")
    print("    *_residuals.png              - Residual analysis (3 panels)")
    print("    *_errors.png                 - Error distribution analysis")
    print("    *_learning_curve.png         - Bias/variance diagnosis")
    print("    *_model.joblib               - Saved trained model")
    print("  Cross-position:")
    print("    position_comparison.png      - Dashboard comparing all positions")
    print("    overall_summary.csv          - Summary statistics table")
    print("    *_multi_output_comparison.csv - Multi vs single output results")


if __name__ == "__main__":
    main()
