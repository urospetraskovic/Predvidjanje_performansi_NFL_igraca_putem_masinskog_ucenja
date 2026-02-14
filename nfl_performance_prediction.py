"""
NFL Player Performance Prediction - Main Analysis Script

This script performs comprehensive analysis for predicting NFL player performance:
1. Data Loading and Preprocessing
2. Position-Specific Model Training (QB, RB, WR, TE)
3. Multiple Algorithm Comparison
4. Feature Importance Analysis (SHAP-like)
5. Multi-Output Regression
6. Full Evaluation and Visualization

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
                             generate_full_report)

# Configuration
DATA_DIR = 'data/processed'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'
RANDOM_STATE = 42

# Create output directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_all_data():
    """Load all position-specific datasets."""
    print("\n" + "="*60)
    print("LOADING NFL PLAYER STATISTICS")
    print("="*60)
    
    preprocessor = NFLDataPreprocessor()
    
    data = {}
    
    # Load QB data
    try:
        data['qb'] = preprocessor.load_qb_data(os.path.join(DATA_DIR, 'all_qb_stats_full.csv'))
        print(f"  QB: {len(data['qb'])} records")
    except FileNotFoundError as e:
        print(f"  QB data not found: {e}")
        data['qb'] = None
    
    # Load RB data  
    try:
        rb_basic = pd.read_csv(os.path.join(DATA_DIR, 'all_rushing_receiving.csv'))
        rb_advanced = pd.read_csv(os.path.join(DATA_DIR, 'all_advanced_rushing_receiving.csv'))
        # Merge on common columns
        merge_cols = ['Season', 'Player', 'PlayerID']
        common_cols = [c for c in merge_cols if c in rb_basic.columns and c in rb_advanced.columns]
        if common_cols:
            data['rb'] = rb_basic.merge(rb_advanced, on=common_cols, how='left', suffixes=('', '_adv'))
        else:
            data['rb'] = rb_basic
        print(f"  RB: {len(data['rb'])} records")
    except FileNotFoundError as e:
        print(f"  RB data not found: {e}")
        data['rb'] = None
    
    # Load TE data
    try:
        te_basic = pd.read_csv(os.path.join(DATA_DIR, 'all_te_receiving_rushing.csv'))
        te_advanced = pd.read_csv(os.path.join(DATA_DIR, 'all_te_advanced_receiving_rushing.csv'))
        merge_cols = ['Season', 'Player', 'PlayerID']
        common_cols = [c for c in merge_cols if c in te_basic.columns and c in te_advanced.columns]
        if common_cols:
            data['te'] = te_basic.merge(te_advanced, on=common_cols, how='left', suffixes=('', '_adv'))
        else:
            data['te'] = te_basic
        print(f"  TE: {len(data['te'])} records")
    except FileNotFoundError as e:
        print(f"  TE data not found: {e}")
        data['te'] = None
    
    # Load WR data (from HuggingFace dataset structure)
    try:
        wr_files = []
        wr_base_dir = 'data/raw/wr'
        for year in range(2015, 2026):
            year_file = os.path.join(wr_base_dir, str(year), 'data', f'wr_{year}.csv')
            if os.path.exists(year_file):
                df = pd.read_csv(year_file)
                df['Season'] = year
                wr_files.append(df)
        
        if wr_files:
            data['wr'] = pd.concat(wr_files, ignore_index=True)
            print(f"  WR: {len(data['wr'])} records")
        else:
            data['wr'] = None
            print("  WR: No data files found")
    except Exception as e:
        print(f"  WR data error: {e}")
        data['wr'] = None
    
    return data


def analyze_qb_performance(df: pd.DataFrame, preprocessor: NFLDataPreprocessor):
    """
    Analyze and predict QB passing yards.
    
    Target: Passing Yards (Yds)
    """
    print("\n" + "="*60)
    print("QUARTERBACK (QB) PERFORMANCE ANALYSIS")
    print("Target Variable: Passing Yards")
    print("="*60)
    
    if df is None or len(df) == 0:
        print("No QB data available.")
        return None, None
    
    # Filter valid records (players with meaningful playing time)
    df_filtered = df[df['G'] >= 4].copy()  # At least 4 games
    print(f"Records after filtering (G>=4): {len(df_filtered)}")
    
    # Create time-based split
    train_df, test_df = create_time_based_split(df_filtered, 'Season', test_seasons=2)
    
    # Prepare features
    feature_cols = [
        'Age', 'G', 'GS', 'Cmp', 'Att', 'Cmp%', 'TD', 'TD%', 'Int', 'Int%',
        '1D', 'Succ%', 'Lng', 'Y/A', 'AY/A', 'Y/C', 'Rate', 'Sk',
        'Yds_Lost', 'Sk%', 'NY/A', 'ANY/A', 
        'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rush_1D', 'Rush_Y/A',
        # Advanced features
        'pass_air_yds', 'pass_yac', 'pass_drops', 'pass_drop_pct',
        'pass_poor_throws', 'pass_on_target_pct', 'pocket_time',
        'pass_blitzed', 'pass_hurried', 'pass_pressured', 'pass_pressured_pct'
    ]
    
    target_col = 'Yds'
    
    # Get available features
    available_features = [col for col in feature_cols if col in train_df.columns]
    print(f"Using {len(available_features)} features")
    
    # Preprocess
    X_train, y_train = preprocessor.full_preprocessing_pipeline(
        train_df, available_features, target_col, treat_outliers=True, scale=True
    )
    
    # Use same scaler for test data (set fit=False)
    test_df_processed = preprocessor.handle_missing_values(test_df)
    test_df_processed = preprocessor.treat_outliers(test_df_processed, available_features, method='clip')
    available_test_features = [col for col in available_features if col in test_df_processed.columns]
    X_test = test_df_processed[available_test_features].copy()
    y_test = test_df_processed[target_col].copy()
    
    # Handle any remaining NaN
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    
    # Align features
    common_features = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Compare models
    print("\n--- Comparing Multiple Algorithms ---")
    comparison_results = compare_models(X_train, y_train, X_test, y_test, 
                                        tune_hyperparameters=False)
    print("\nModel Comparison Results:")
    print(comparison_results.to_string(index=False))
    
    # Save comparison
    comparison_results.to_csv(os.path.join(RESULTS_DIR, 'qb_model_comparison.csv'), index=False)
    
    # Plot comparison
    plot_model_comparison(comparison_results, 
                         save_path=os.path.join(RESULTS_DIR, 'qb_model_comparison.png'))
    
    # Train best model (Random Forest with tuning)
    print("\n--- Training Random Forest with Hyperparameter Tuning ---")
    best_model = NFLPerformanceModel(model_type='random_forest')
    train_metrics = best_model.train(X_train, y_train, tune_hyperparameters=True)
    test_metrics = best_model.evaluate(X_test, y_test)
    
    # Feature importance
    if best_model.feature_importance is not None:
        print("\nTop 15 Most Important Features for QB Passing Yards:")
        print(best_model.feature_importance.head(15).to_string(index=False))
        
        plot_feature_importance(
            best_model.feature_importance,
            top_n=15,
            title="QB Model - Feature Importance (SHAP-like)",
            save_path=os.path.join(RESULTS_DIR, 'qb_feature_importance.png')
        )
    
    # Full evaluation
    generate_full_report(best_model, X_test, y_test, "QB_PassingYards",
                        best_model.feature_importance, RESULTS_DIR)
    
    # Save model
    best_model.save_model(os.path.join(MODELS_DIR, 'qb_model.joblib'))
    
    return best_model, test_metrics


def analyze_rb_performance(df: pd.DataFrame, preprocessor: NFLDataPreprocessor):
    """
    Analyze and predict RB rushing yards.
    
    Target: Rushing Yards (Yds)
    """
    print("\n" + "="*60)
    print("RUNNING BACK (RB) PERFORMANCE ANALYSIS")
    print("Target Variable: Rushing Yards")
    print("="*60)
    
    if df is None or len(df) == 0:
        print("No RB data available.")
        return None, None
    
    # Filter valid records
    df_filtered = df[df['G'] >= 4].copy()
    print(f"Records after filtering (G>=4): {len(df_filtered)}")
    
    # Create time-based split
    train_df, test_df = create_time_based_split(df_filtered, 'Season', test_seasons=2)
    
    # Features for RB
    feature_cols = [
        'Age', 'G', 'GS', 'Att', 'TD', '1D', 'Succ%', 'Lng', 'Y/A', 'Y/G', 'A/G',
        'Tgt', 'Rec', 'Yds.1', 'Y/R', 'TD.1', '1D.1', 'rec_success', 'rec_long',
        'R/G', 'Y/G.1', 'catch_pct', 'Y/Tgt', 'Touch', 'yds_per_touch',
        # Advanced features
        'YBC', 'YBC/Att', 'YAC', 'YAC/Att', 'BrkTkl', 'Att/Br'
    ]
    
    target_col = 'Yds'
    
    # Get available features
    available_features = [col for col in feature_cols if col in train_df.columns]
    print(f"Using {len(available_features)} features")
    
    # Preprocess
    X_train, y_train = preprocessor.full_preprocessing_pipeline(
        train_df, available_features, target_col, treat_outliers=True, scale=True
    )
    
    # Process test data
    test_df_processed = preprocessor.handle_missing_values(test_df)
    test_df_processed = preprocessor.treat_outliers(test_df_processed, available_features, method='clip')
    available_test_features = [col for col in available_features if col in test_df_processed.columns]
    X_test = test_df_processed[available_test_features].copy()
    y_test = test_df_processed[target_col].copy()
    
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    
    common_features = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Compare models
    print("\n--- Comparing Multiple Algorithms ---")
    comparison_results = compare_models(X_train, y_train, X_test, y_test)
    print("\nModel Comparison Results:")
    print(comparison_results.to_string(index=False))
    comparison_results.to_csv(os.path.join(RESULTS_DIR, 'rb_model_comparison.csv'), index=False)
    
    plot_model_comparison(comparison_results,
                         save_path=os.path.join(RESULTS_DIR, 'rb_model_comparison.png'))
    
    # Train best model
    print("\n--- Training Random Forest with Hyperparameter Tuning ---")
    best_model = NFLPerformanceModel(model_type='random_forest')
    train_metrics = best_model.train(X_train, y_train, tune_hyperparameters=True)
    test_metrics = best_model.evaluate(X_test, y_test)
    
    if best_model.feature_importance is not None:
        print("\nTop 15 Most Important Features for RB Rushing Yards:")
        print(best_model.feature_importance.head(15).to_string(index=False))
        
        plot_feature_importance(
            best_model.feature_importance,
            top_n=15,
            title="RB Model - Feature Importance",
            save_path=os.path.join(RESULTS_DIR, 'rb_feature_importance.png')
        )
    
    generate_full_report(best_model, X_test, y_test, "RB_RushingYards",
                        best_model.feature_importance, RESULTS_DIR)
    
    best_model.save_model(os.path.join(MODELS_DIR, 'rb_model.joblib'))
    
    return best_model, test_metrics


def analyze_wr_performance(df: pd.DataFrame, preprocessor: NFLDataPreprocessor):
    """
    Analyze and predict WR receiving yards.
    
    Target: Receiving Yards
    """
    print("\n" + "="*60)
    print("WIDE RECEIVER (WR) PERFORMANCE ANALYSIS")
    print("Target Variable: Receiving Yards")
    print("="*60)
    
    if df is None or len(df) == 0:
        print("No WR data available.")
        return None, None
    
    # Aggregate per-game stats to season stats by player
    agg_cols = {
        'targets': 'sum',
        'receptions': 'sum',
        'receiving_yards': 'sum',
        'air_yards': 'sum',
        'yac': 'sum',
        'tds': 'sum',
        'epa': 'sum',
        'avg_depth': 'mean',
        'catch_rate': 'mean',
        'yards_per_target': 'mean',
        'target_share': 'mean',
        'air_yard_share': 'mean',
        'red_zone_targets': 'sum',
        'end_zone_targets': 'sum',
        'third_down_targets': 'sum',
        'adot': 'mean',
        'yac_per_reception': 'mean',
        'success_rate': 'mean',
        'first_downs': 'sum',
        'explosive_plays': 'sum'
    }
    
    # Get available aggregation columns
    available_agg = {k: v for k, v in agg_cols.items() if k in df.columns}
    
    if 'receiver_player_name' in df.columns and 'Season' in df.columns:
        df_season = df.groupby(['receiver_player_name', 'Season']).agg(available_agg).reset_index()
        df_season['games'] = df.groupby(['receiver_player_name', 'Season']).size().values
    else:
        df_season = df.copy()
    
    # Filter meaningful records
    if 'games' in df_season.columns:
        df_filtered = df_season[df_season['games'] >= 4].copy()
    else:
        df_filtered = df_season.copy()
    
    print(f"Records for analysis: {len(df_filtered)}")
    
    if len(df_filtered) < 50:
        print("Insufficient WR data for meaningful analysis.")
        return None, None
    
    # Create time-based split
    train_df, test_df = create_time_based_split(df_filtered, 'Season', test_seasons=2)
    
    # Features for WR
    feature_cols = [
        'targets', 'receptions', 'air_yards', 'yac', 'tds', 'epa',
        'avg_depth', 'catch_rate', 'yards_per_target', 'target_share',
        'air_yard_share', 'red_zone_targets', 'end_zone_targets',
        'third_down_targets', 'adot', 'yac_per_reception', 'success_rate',
        'first_downs', 'explosive_plays', 'games'
    ]
    
    target_col = 'receiving_yards'
    
    available_features = [col for col in feature_cols if col in train_df.columns]
    print(f"Using {len(available_features)} features")
    
    if len(available_features) < 3:
        print("Not enough features available for WR analysis.")
        return None, None
    
    # Preprocess
    X_train, y_train = preprocessor.full_preprocessing_pipeline(
        train_df, available_features, target_col, treat_outliers=True, scale=True
    )
    
    test_df_processed = preprocessor.handle_missing_values(test_df)
    X_test = test_df_processed[[c for c in available_features if c in test_df_processed.columns]].copy()
    y_test = test_df_processed[target_col].copy()
    
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    
    common_features = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Compare models
    print("\n--- Comparing Multiple Algorithms ---")
    comparison_results = compare_models(X_train, y_train, X_test, y_test)
    print("\nModel Comparison Results:")
    print(comparison_results.to_string(index=False))
    comparison_results.to_csv(os.path.join(RESULTS_DIR, 'wr_model_comparison.csv'), index=False)
    
    plot_model_comparison(comparison_results,
                         save_path=os.path.join(RESULTS_DIR, 'wr_model_comparison.png'))
    
    # Train best model
    print("\n--- Training Random Forest with Hyperparameter Tuning ---")
    best_model = NFLPerformanceModel(model_type='random_forest')
    train_metrics = best_model.train(X_train, y_train, tune_hyperparameters=True)
    test_metrics = best_model.evaluate(X_test, y_test)
    
    if best_model.feature_importance is not None:
        print("\nTop 15 Most Important Features for WR Receiving Yards:")
        print(best_model.feature_importance.head(15).to_string(index=False))
        
        plot_feature_importance(
            best_model.feature_importance,
            top_n=15,
            title="WR Model - Feature Importance",
            save_path=os.path.join(RESULTS_DIR, 'wr_feature_importance.png')
        )
    
    generate_full_report(best_model, X_test, y_test, "WR_ReceivingYards",
                        best_model.feature_importance, RESULTS_DIR)
    
    best_model.save_model(os.path.join(MODELS_DIR, 'wr_model.joblib'))
    
    return best_model, test_metrics


def analyze_te_performance(df: pd.DataFrame, preprocessor: NFLDataPreprocessor):
    """
    Analyze and predict TE receiving yards.
    
    Target: Receiving Yards (Yds)
    """
    print("\n" + "="*60)
    print("TIGHT END (TE) PERFORMANCE ANALYSIS")
    print("Target Variable: Receiving Yards")
    print("="*60)
    
    if df is None or len(df) == 0:
        print("No TE data available.")
        return None, None
    
    # Filter valid records
    df_filtered = df[df['G'] >= 4].copy()
    print(f"Records after filtering (G>=4): {len(df_filtered)}")
    
    # Create time-based split
    train_df, test_df = create_time_based_split(df_filtered, 'Season', test_seasons=2)
    
    # Features for TE
    feature_cols = [
        'Age', 'G', 'GS', 'Tgt', 'Rec', 'Y/R', 'TD', '1D', 'rec_success',
        'rec_long', 'R/G', 'Y/G', 'catch_pct', 'Y/Tgt', 'Touch', 'yds_per_touch'
    ]
    
    target_col = 'Yds'
    
    available_features = [col for col in feature_cols if col in train_df.columns]
    print(f"Using {len(available_features)} features")
    
    # Preprocess
    X_train, y_train = preprocessor.full_preprocessing_pipeline(
        train_df, available_features, target_col, treat_outliers=True, scale=True
    )
    
    test_df_processed = preprocessor.handle_missing_values(test_df)
    X_test = test_df_processed[[c for c in available_features if c in test_df_processed.columns]].copy()
    y_test = test_df_processed[target_col].copy()
    
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    
    common_features = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Compare models
    print("\n--- Comparing Multiple Algorithms ---")
    comparison_results = compare_models(X_train, y_train, X_test, y_test)
    print("\nModel Comparison Results:")
    print(comparison_results.to_string(index=False))
    comparison_results.to_csv(os.path.join(RESULTS_DIR, 'te_model_comparison.csv'), index=False)
    
    plot_model_comparison(comparison_results,
                         save_path=os.path.join(RESULTS_DIR, 'te_model_comparison.png'))
    
    # Train best model
    print("\n--- Training Random Forest with Hyperparameter Tuning ---")
    best_model = NFLPerformanceModel(model_type='random_forest')
    train_metrics = best_model.train(X_train, y_train, tune_hyperparameters=True)
    test_metrics = best_model.evaluate(X_test, y_test)
    
    if best_model.feature_importance is not None:
        print("\nTop 15 Most Important Features for TE Receiving Yards:")
        print(best_model.feature_importance.head(15).to_string(index=False))
        
        plot_feature_importance(
            best_model.feature_importance,
            top_n=15,
            title="TE Model - Feature Importance",
            save_path=os.path.join(RESULTS_DIR, 'te_feature_importance.png')
        )
    
    generate_full_report(best_model, X_test, y_test, "TE_ReceivingYards",
                        best_model.feature_importance, RESULTS_DIR)
    
    best_model.save_model(os.path.join(MODELS_DIR, 'te_model.joblib'))
    
    return best_model, test_metrics


def run_multi_output_analysis(df: pd.DataFrame, position: str):
    """
    Run multi-output regression to predict multiple metrics simultaneously.
    Based on Elimam et al. (2025) methodology.
    """
    print("\n" + "="*60)
    print(f"MULTI-OUTPUT REGRESSION: {position}")
    print("Predicting multiple performance metrics simultaneously")
    print("="*60)
    
    if df is None or len(df) == 0:
        print(f"No {position} data available for multi-output analysis.")
        return None
    
    # Define targets based on position
    if position == 'QB':
        target_cols = ['Yds', 'TD', 'Rate']  # Passing Yards, TDs, Passer Rating
        feature_cols = ['Age', 'G', 'GS', 'Cmp', 'Att', 'Cmp%', 'Int', '1D', 'Sk']
    elif position == 'RB':
        target_cols = ['Yds', 'TD', 'Yds.1']  # Rushing Yards, Rushing TDs, Receiving Yards
        feature_cols = ['Age', 'G', 'GS', 'Att', '1D', 'Succ%', 'Tgt', 'Rec']
    else:
        print(f"Multi-output not configured for {position}")
        return None
    
    # Check available columns
    available_targets = [c for c in target_cols if c in df.columns]
    available_features = [c for c in feature_cols if c in df.columns]
    
    if len(available_targets) < 2:
        print(f"Not enough target columns available for multi-output regression.")
        return None
    
    print(f"Predicting: {available_targets}")
    print(f"Using features: {available_features}")
    
    # Prepare data
    df_filtered = df.dropna(subset=available_targets + available_features)
    
    # Split
    train_df, test_df = create_time_based_split(df_filtered, 'Season', test_seasons=2)
    
    X_train = train_df[available_features].fillna(train_df[available_features].median())
    y_train = train_df[available_targets]
    X_test = test_df[available_features].fillna(train_df[available_features].median())
    y_test = test_df[available_targets]
    
    # Train multi-output model
    print("\n--- Training Multi-Output Random Forest ---")
    mo_model = MultiOutputNFLModel(base_model_type='random_forest')
    train_metrics = mo_model.train(X_train, y_train)
    
    print("\nTraining Metrics per Target:")
    for target, metrics in train_metrics.items():
        if isinstance(metrics, dict):
            print(f"  {target}: RMSE={metrics['train_rmse']:.2f}, R²={metrics['train_r2']:.4f}")
    
    # Evaluate
    test_metrics = mo_model.evaluate(X_test, y_test)
    
    print("\nTest Metrics per Target:")
    for target, metrics in test_metrics.items():
        if isinstance(metrics, dict):
            print(f"  {target}: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.4f}")
    
    print(f"\nAverage RMSE (aRMSE): {test_metrics['average']['rmse']:.2f}")
    print(f"Average R²: {test_metrics['average']['r2']:.4f}")
    
    return mo_model, test_metrics


def main():
    """Main entry point for NFL Performance Prediction analysis."""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#     NFL PLAYER PERFORMANCE PREDICTION SYSTEM           #")
    print("#     Using Machine Learning                              #")
    print("#" + " "*58 + "#")
    print("#"*60)
    print("\nAuthors: Milan Jovkić R2 10/2025, Uroš Petrašković R2 9/2025")
    print("\nMethodology based on:")
    print("  - Random Forest Regression (primary model)")
    print("  - Multiple algorithm comparison")
    print("  - Position-specific modeling")
    print("  - Feature importance analysis")
    print("  - Multi-output regression")
    
    # Initialize preprocessor
    preprocessor = NFLDataPreprocessor(scaler_type='standard')
    
    # Load all data
    data = load_all_data()
    
    # Store results for comparison
    all_results = {}
    
    # Analyze each position
    
    # 1. Quarterbacks
    qb_model, qb_metrics = analyze_qb_performance(data['qb'], preprocessor)
    if qb_metrics:
        all_results['QB'] = qb_metrics
    
    # 2. Running Backs
    rb_model, rb_metrics = analyze_rb_performance(data['rb'], preprocessor)
    if rb_metrics:
        all_results['RB'] = rb_metrics
    
    # 3. Wide Receivers
    wr_model, wr_metrics = analyze_wr_performance(data['wr'], preprocessor)
    if wr_metrics:
        all_results['WR'] = wr_metrics
    
    # 4. Tight Ends
    te_model, te_metrics = analyze_te_performance(data['te'], preprocessor)
    if te_metrics:
        all_results['TE'] = te_metrics
    
    # Multi-output regression
    if data['qb'] is not None:
        run_multi_output_analysis(data['qb'], 'QB')
    
    if data['rb'] is not None:
        run_multi_output_analysis(data['rb'], 'RB')
    
    # Create position comparison dashboard
    if all_results:
        print("\n" + "="*60)
        print("OVERALL POSITION COMPARISON")
        print("="*60)
        
        create_position_comparison_dashboard(
            all_results,
            save_path=os.path.join(RESULTS_DIR, 'position_comparison.png')
        )
        
        # Save summary
        summary_df = pd.DataFrame([
            {'Position': pos, **metrics}
            for pos, metrics in all_results.items()
        ])
        summary_df.to_csv(os.path.join(RESULTS_DIR, 'overall_summary.csv'), index=False)
        print("\nSummary saved to results/overall_summary.csv")
    
    print("\n" + "#"*60)
    print("#     ANALYSIS COMPLETE                                   #")
    print("#"*60)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Models saved to: {MODELS_DIR}/")
    print("\nKey files generated:")
    print("  - position_comparison.png: Visual comparison of all positions")
    print("  - *_model_comparison.csv: Algorithm comparison for each position")
    print("  - *_feature_importance.png: Feature importance plots")
    print("  - *_predictions.png: Actual vs Predicted plots")
    print("  - overall_summary.csv: Summary statistics")


if __name__ == "__main__":
    main()
