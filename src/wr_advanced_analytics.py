"""
NFL Wide Receiver Advanced Analytics Suite
==========================================

This comprehensive module provides advanced analytics for NFL Wide Receivers:

1. Game-by-Game Performance Prediction
2. Player Style Clustering (K-Means, Hierarchical)
3. Breakout Player Identification
4. Matchup Analysis vs Defenses
5. Red Zone Efficiency Analysis
6. Consistency/Volatility Scoring
7. Weather Impact Analysis
8. Situational Performance (Quarter, Win Probability)
9. Target Share Prediction
10. Fantasy Points Projection

Authors: Milan Jovkić R2 10/2025, Uroš Petrašković R2 9/2025
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score


class WRDataLoader:
    """Load and combine WR data from multiple seasons."""
    
    def __init__(self, base_dir: str = 'data/raw/wr'):
        self.base_dir = base_dir
        self.data = None
        
    def load_all_seasons(self, start_year: int = 2015, end_year: int = 2025) -> pd.DataFrame:
        """Load WR data for all available seasons."""
        all_data = []
        
        for year in range(start_year, end_year + 1):
            filepath = os.path.join(self.base_dir, str(year), 'data', f'wr_{year}.csv')
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['season'] = year
                all_data.append(df)
                print(f"Loaded {year}: {len(df)} game records")
        
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal records: {len(self.data)}")
            print(f"Unique players: {self.data['receiver_player_name'].nunique()}")
            print(f"Columns: {len(self.data.columns)}")
            return self.data
        else:
            raise FileNotFoundError("No WR data files found")
    
    def aggregate_to_season(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Aggregate game-by-game data to season totals."""
        if df is None:
            df = self.data
        
        # Define aggregation rules
        sum_cols = ['targets', 'receptions', 'receiving_yards', 'air_yards', 'yac', 
                    'tds', 'red_zone_targets', 'end_zone_targets', 'third_down_targets',
                    'fourth_down_targets', 'high_leverage_targets', 'explosive_plays',
                    'first_downs', 'yards_Q1', 'yards_Q2', 'yards_Q3', 'yards_Q4']
        
        mean_cols = ['catch_rate', 'yards_per_target', 'adot', 'yac_per_reception',
                     'success_rate', 'epa', 'wpa', 'target_share', 'air_yard_share',
                     'avg_depth', 'qb_comp_pct', 'qb_cpoe']
        
        agg_dict = {}
        for col in sum_cols:
            if col in df.columns:
                agg_dict[col] = 'sum'
        for col in mean_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'
        
        # Count games
        agg_dict['game_id'] = 'count'
        
        season_df = df.groupby(['receiver_player_name', 'season']).agg(agg_dict).reset_index()
        season_df.rename(columns={'game_id': 'games'}, inplace=True)
        
        # Calculate per-game stats
        if 'receiving_yards' in season_df.columns:
            season_df['yards_per_game'] = season_df['receiving_yards'] / season_df['games']
        if 'receptions' in season_df.columns:
            season_df['receptions_per_game'] = season_df['receptions'] / season_df['games']
        if 'targets' in season_df.columns:
            season_df['targets_per_game'] = season_df['targets'] / season_df['games']
        
        return season_df


class WRClusterAnalysis:
    """
    Cluster WR players by playing style using unsupervised learning.
    
    Identifies player archetypes like:
    - Deep threats
    - Possession receivers
    - YAC monsters
    - Red zone specialists
    - Slot receivers
    """
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.cluster_labels = None
        self.feature_cols = None
        
    def prepare_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for clustering analysis."""
        # Key features that define WR playing style
        style_features = [
            'adot',                  # Average depth of target (deep vs short)
            'yac_per_reception',     # Yards after catch ability
            'catch_rate',            # Hands/reliability
            'target_share',          # Volume/usage
            'air_yard_share',        # Deep target share
            'td_rate',               # Scoring ability
            'success_rate',          # Efficiency
            'yards_per_target',      # Overall efficiency
            'explosive_plays',       # Big play ability
            'first_downs'            # Chain mover
        ]
        
        # Filter to available features
        self.feature_cols = [c for c in style_features if c in df.columns]
        
        # Need sufficient games for meaningful stats
        df_filtered = df[df['games'] >= 8].copy() if 'games' in df.columns else df.copy()
        
        # Drop rows with missing values in clustering features
        df_filtered = df_filtered.dropna(subset=self.feature_cols)
        
        return df_filtered
    
    def fit_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit K-Means clustering on WR data."""
        df_prep = self.prepare_clustering_features(df)
        
        if len(df_prep) < self.n_clusters:
            raise ValueError(f"Not enough samples ({len(df_prep)}) for {self.n_clusters} clusters")
        
        X = df_prep[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        df_prep['cluster'] = self.cluster_labels
        
        # PCA for visualization
        X_pca = self.pca.fit_transform(X_scaled)
        df_prep['pca_1'] = X_pca[:, 0]
        df_prep['pca_2'] = X_pca[:, 1]
        
        # Calculate silhouette score
        sil_score = silhouette_score(X_scaled, self.cluster_labels)
        print(f"Silhouette Score: {sil_score:.3f}")
        
        return df_prep
    
    def get_cluster_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get average profile for each cluster."""
        profiles = df.groupby('cluster')[self.feature_cols].mean()
        profiles['count'] = df.groupby('cluster').size()
        return profiles
    
    def label_clusters(self, profiles: pd.DataFrame) -> Dict[int, str]:
        """Automatically label clusters based on their characteristics."""
        labels = {}
        
        for cluster in profiles.index:
            profile = profiles.loc[cluster]
            
            # Determine cluster type based on dominant characteristics
            if profile.get('adot', 0) > profiles['adot'].mean() * 1.2:
                if profile.get('td_rate', 0) > profiles['td_rate'].mean():
                    labels[cluster] = "Deep Threat / Big Play"
                else:
                    labels[cluster] = "Deep Ball Specialist"
            elif profile.get('yac_per_reception', 0) > profiles['yac_per_reception'].mean() * 1.2:
                labels[cluster] = "YAC Monster"
            elif profile.get('catch_rate', 0) > profiles['catch_rate'].mean() * 1.1:
                labels[cluster] = "Possession Receiver"
            elif profile.get('target_share', 0) > profiles['target_share'].mean() * 1.3:
                labels[cluster] = "Alpha / WR1"
            else:
                labels[cluster] = "Role Player"
        
        return labels


class WRBreakoutPredictor:
    """
    Predict which WRs are likely to have breakout seasons.
    
    Breakout defined as: 50%+ increase in receiving yards from previous season
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_cols = None
        
    def create_breakout_dataset(self, season_df: pd.DataFrame) -> pd.DataFrame:
        """Create dataset with previous season stats to predict next season."""
        # Sort by player and season
        df = season_df.sort_values(['receiver_player_name', 'season']).copy()
        
        # Create lag features (previous season stats)
        lag_cols = ['receiving_yards', 'targets', 'receptions', 'tds', 'catch_rate',
                    'yards_per_target', 'target_share', 'games']
        
        for col in lag_cols:
            if col in df.columns:
                df[f'{col}_prev'] = df.groupby('receiver_player_name')[col].shift(1)
        
        # Create target: next season yards
        df['yards_next'] = df.groupby('receiver_player_name')['receiving_yards'].shift(-1)
        
        # Calculate improvement
        df['yards_improvement'] = (df['yards_next'] - df['receiving_yards']) / (df['receiving_yards'] + 1)
        df['is_breakout'] = (df['yards_improvement'] > 0.5).astype(int)
        
        # Drop rows without previous/next season data
        df = df.dropna(subset=['receiving_yards_prev', 'yards_next'])
        
        return df
    
    def train_model(self, df: pd.DataFrame) -> Dict:
        """Train breakout prediction model."""
        # Features from current and previous season
        feature_cols = [c for c in df.columns if c.endswith('_prev') or 
                       c in ['receiving_yards', 'targets', 'catch_rate', 'yards_per_target', 
                            'target_share', 'games', 'tds', 'adot', 'yac_per_reception']]
        
        self.feature_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[self.feature_cols].fillna(0)
        y = df['yards_next']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        print(f"Breakout Prediction Model Performance:")
        print(f"  MAE:  {metrics['mae']:.1f} yards")
        print(f"  RMSE: {metrics['rmse']:.1f} yards")
        print(f"  R²:   {metrics['r2']:.4f}")
        
        return metrics
    
    def get_breakout_candidates(self, df: pd.DataFrame, current_season: int) -> pd.DataFrame:
        """Get players most likely to break out next season."""
        current_data = df[df['season'] == current_season].copy()
        
        if len(current_data) == 0:
            return pd.DataFrame()
        
        X = current_data[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        current_data['predicted_yards'] = self.model.predict(X_scaled)
        current_data['predicted_improvement'] = (
            (current_data['predicted_yards'] - current_data['receiving_yards']) / 
            (current_data['receiving_yards'] + 1)
        )
        
        # Top breakout candidates
        breakout = current_data.nlargest(20, 'predicted_improvement')
        
        return breakout[['receiver_player_name', 'season', 'receiving_yards', 
                        'predicted_yards', 'predicted_improvement']]


class WRMatchupAnalyzer:
    """Analyze WR performance against different defenses."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def get_defense_rankings(self) -> pd.DataFrame:
        """Rank defenses by how well they defend WRs."""
        defense_stats = self.df.groupby('defteam').agg({
            'receiving_yards': 'mean',
            'tds': 'mean',
            'catch_rate': 'mean',
            'epa': 'mean',
            'targets': 'count'
        }).reset_index()
        
        defense_stats.columns = ['defense', 'yards_allowed_per_game', 'tds_allowed',
                                 'catch_rate_allowed', 'epa_allowed', 'sample_size']
        
        # Rank (lower is better defense)
        defense_stats['rank'] = defense_stats['yards_allowed_per_game'].rank()
        
        return defense_stats.sort_values('yards_allowed_per_game')
    
    def get_wr_vs_defense(self, player_name: str) -> pd.DataFrame:
        """Get a WR's performance against each defense."""
        player_df = self.df[self.df['receiver_player_name'] == player_name]
        
        vs_defense = player_df.groupby('defteam').agg({
            'receiving_yards': ['sum', 'mean', 'count'],
            'tds': 'sum',
            'targets': 'sum',
            'receptions': 'sum'
        }).reset_index()
        
        vs_defense.columns = ['defense', 'total_yards', 'avg_yards', 'games',
                             'total_tds', 'total_targets', 'total_receptions']
        
        return vs_defense.sort_values('avg_yards', ascending=False)
    
    def predict_matchup(self, player_name: str, defense: str) -> Dict:
        """Predict WR performance in a specific matchup."""
        # Player's average stats
        player_stats = self.df[self.df['receiver_player_name'] == player_name].mean()
        
        # Defense average allowed
        defense_stats = self.df[self.df['defteam'] == defense].mean()
        
        # Simple weighted prediction
        predicted_yards = (player_stats['receiving_yards'] * 0.6 + 
                          defense_stats['receiving_yards'] * 0.4)
        
        return {
            'player': player_name,
            'defense': defense,
            'predicted_yards': predicted_yards,
            'player_avg': player_stats['receiving_yards'],
            'defense_avg_allowed': defense_stats['receiving_yards']
        }


class WRConsistencyAnalyzer:
    """Analyze WR consistency and volatility."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def calculate_consistency_scores(self) -> pd.DataFrame:
        """Calculate consistency score for each WR."""
        consistency = self.df.groupby('receiver_player_name').agg({
            'receiving_yards': ['mean', 'std', 'count', 'min', 'max'],
            'targets': ['mean', 'std'],
            'receptions': ['mean', 'std']
        }).reset_index()
        
        consistency.columns = ['player', 'yards_mean', 'yards_std', 'games',
                              'yards_min', 'yards_max', 'targets_mean', 'targets_std',
                              'receptions_mean', 'receptions_std']
        
        # Filter to players with enough games
        consistency = consistency[consistency['games'] >= 8]
        
        # Coefficient of variation (lower = more consistent)
        consistency['yards_cv'] = consistency['yards_std'] / (consistency['yards_mean'] + 1)
        
        # Consistency score (0-100, higher = more consistent)
        max_cv = consistency['yards_cv'].max()
        consistency['consistency_score'] = 100 * (1 - consistency['yards_cv'] / max_cv)
        
        # Boom/bust ratio
        consistency['boom_bust_range'] = consistency['yards_max'] - consistency['yards_min']
        
        return consistency.sort_values('consistency_score', ascending=False)
    
    def get_floor_ceiling(self) -> pd.DataFrame:
        """Calculate floor and ceiling for each WR."""
        floor_ceiling = self.df.groupby('receiver_player_name').agg({
            'receiving_yards': ['quantile', 'quantile', 'mean']
        }).reset_index()
        
        # Calculate 10th and 90th percentile
        player_stats = []
        for player in self.df['receiver_player_name'].unique():
            player_data = self.df[self.df['receiver_player_name'] == player]['receiving_yards']
            if len(player_data) >= 5:
                player_stats.append({
                    'player': player,
                    'floor': player_data.quantile(0.1),
                    'ceiling': player_data.quantile(0.9),
                    'median': player_data.median(),
                    'mean': player_data.mean(),
                    'games': len(player_data)
                })
        
        return pd.DataFrame(player_stats).sort_values('ceiling', ascending=False)


class WRWeatherAnalyzer:
    """Analyze how weather affects WR performance."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def analyze_by_weather(self) -> pd.DataFrame:
        """Analyze WR stats by weather conditions."""
        weather_cols = ['is_dome', 'is_rain', 'is_snow', 'is_clear']
        
        results = []
        for weather in weather_cols:
            if weather in self.df.columns:
                weather_df = self.df[self.df[weather] == 1]
                if len(weather_df) > 0:
                    results.append({
                        'condition': weather.replace('is_', ''),
                        'avg_yards': weather_df['receiving_yards'].mean(),
                        'avg_targets': weather_df['targets'].mean(),
                        'catch_rate': weather_df['catch_rate'].mean(),
                        'sample_size': len(weather_df)
                    })
        
        return pd.DataFrame(results)
    
    def analyze_by_temperature(self) -> pd.DataFrame:
        """Analyze WR stats by temperature ranges."""
        if 'temp_f' not in self.df.columns:
            return pd.DataFrame()
        
        # Create temperature bins
        bins = [0, 32, 50, 70, 100]
        labels = ['Cold (<32°F)', 'Cool (32-50°F)', 'Mild (50-70°F)', 'Warm (>70°F)']
        
        df = self.df.copy()
        df['temp_range'] = pd.cut(df['temp_f'], bins=bins, labels=labels)
        
        temp_analysis = df.groupby('temp_range').agg({
            'receiving_yards': 'mean',
            'catch_rate': 'mean',
            'targets': 'mean',
            'game_id': 'count'
        }).reset_index()
        
        temp_analysis.columns = ['temp_range', 'avg_yards', 'catch_rate', 
                                 'avg_targets', 'sample_size']
        
        return temp_analysis
    
    def analyze_by_wind(self) -> pd.DataFrame:
        """Analyze WR stats by wind conditions."""
        if 'wind_mph' not in self.df.columns:
            return pd.DataFrame()
        
        bins = [0, 5, 10, 15, 50]
        labels = ['Calm (0-5)', 'Light (5-10)', 'Moderate (10-15)', 'Windy (15+)']
        
        df = self.df.copy()
        df['wind_range'] = pd.cut(df['wind_mph'], bins=bins, labels=labels)
        
        wind_analysis = df.groupby('wind_range').agg({
            'receiving_yards': 'mean',
            'adot': 'mean',  # Deep balls affected by wind
            'catch_rate': 'mean',
            'game_id': 'count'
        }).reset_index()
        
        wind_analysis.columns = ['wind_range', 'avg_yards', 'avg_depth', 
                                'catch_rate', 'sample_size']
        
        return wind_analysis


class WRSituationalAnalyzer:
    """Analyze WR performance in different game situations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def analyze_by_quarter(self) -> pd.DataFrame:
        """Analyze performance by quarter."""
        quarter_cols = ['yards_Q1', 'yards_Q2', 'yards_Q3', 'yards_Q4']
        available = [c for c in quarter_cols if c in self.df.columns]
        
        if not available:
            return pd.DataFrame()
        
        quarter_stats = self.df[available].sum()
        total_yards = quarter_stats.sum()
        
        quarter_df = pd.DataFrame({
            'quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
            'total_yards': [quarter_stats.get(c, 0) for c in quarter_cols],
            'pct_of_total': [quarter_stats.get(c, 0) / total_yards * 100 for c in quarter_cols]
        })
        
        return quarter_df
    
    def analyze_by_game_script(self) -> pd.DataFrame:
        """Analyze performance when trailing vs leading."""
        if 'trailing_pct' not in self.df.columns:
            return pd.DataFrame()
        
        df = self.df.copy()
        
        # Classify games by game script
        df['game_script'] = 'Neutral'
        df.loc[df['trailing_pct'] > 0.6, 'game_script'] = 'Trailing'
        df.loc[df['leading_pct'] > 0.6, 'game_script'] = 'Leading'
        
        script_analysis = df.groupby('game_script').agg({
            'receiving_yards': 'mean',
            'targets': 'mean',
            'catch_rate': 'mean',
            'adot': 'mean',
            'game_id': 'count'
        }).reset_index()
        
        script_analysis.columns = ['game_script', 'avg_yards', 'avg_targets',
                                   'catch_rate', 'avg_depth', 'sample_size']
        
        return script_analysis
    
    def analyze_red_zone(self) -> pd.DataFrame:
        """Analyze red zone efficiency."""
        rz_cols = ['red_zone_targets', 'end_zone_targets', 'tds']
        available = [c for c in rz_cols if c in self.df.columns]
        
        if len(available) < 2:
            return pd.DataFrame()
        
        rz_stats = self.df.groupby('receiver_player_name')[available].sum().reset_index()
        
        if 'red_zone_targets' in rz_stats.columns and 'tds' in rz_stats.columns:
            rz_stats['rz_td_rate'] = rz_stats['tds'] / (rz_stats['red_zone_targets'] + 1)
        
        return rz_stats.sort_values('tds', ascending=False).head(50)


class WRFantasyProjector:
    """Project fantasy football points for WRs."""
    
    # Standard PPR scoring
    SCORING = {
        'receiving_yards': 0.1,
        'receptions': 1.0,
        'tds': 6.0
    }
    
    def __init__(self, df: pd.DataFrame, scoring: Dict = None):
        self.df = df
        if scoring:
            self.SCORING = scoring
        
    def calculate_fantasy_points(self) -> pd.DataFrame:
        """Calculate fantasy points for each game."""
        df = self.df.copy()
        
        df['fantasy_points'] = 0
        
        if 'receiving_yards' in df.columns:
            df['fantasy_points'] += df['receiving_yards'] * self.SCORING['receiving_yards']
        if 'receptions' in df.columns:
            df['fantasy_points'] += df['receptions'] * self.SCORING['receptions']
        if 'tds' in df.columns:
            df['fantasy_points'] += df['tds'] * self.SCORING['tds']
        
        return df
    
    def get_season_projections(self) -> pd.DataFrame:
        """Get season total fantasy point projections."""
        df = self.calculate_fantasy_points()
        
        season_totals = df.groupby(['receiver_player_name', 'season']).agg({
            'fantasy_points': 'sum',
            'game_id': 'count',
            'receiving_yards': 'sum',
            'receptions': 'sum',
            'tds': 'sum',
            'targets': 'sum'
        }).reset_index()
        
        season_totals.columns = ['player', 'season', 'fantasy_points', 'games',
                                 'total_yards', 'total_receptions', 'total_tds', 'total_targets']
        
        season_totals['ppg'] = season_totals['fantasy_points'] / season_totals['games']
        
        return season_totals.sort_values('fantasy_points', ascending=False)
    
    def project_next_season(self, model_type: str = 'rf') -> pd.DataFrame:
        """Use ML to project next season fantasy points."""
        df = self.calculate_fantasy_points()
        
        # Aggregate to season
        season_df = df.groupby(['receiver_player_name', 'season']).agg({
            'fantasy_points': 'sum',
            'receiving_yards': 'sum',
            'receptions': 'sum',
            'tds': 'sum',
            'targets': 'sum',
            'catch_rate': 'mean',
            'yards_per_target': 'mean',
            'target_share': 'mean',
            'game_id': 'count'
        }).reset_index()
        
        season_df.columns = ['player', 'season', 'fantasy_points', 'yards', 
                            'receptions', 'tds', 'targets', 'catch_rate',
                            'yards_per_target', 'target_share', 'games']
        
        # Create lag features
        season_df = season_df.sort_values(['player', 'season'])
        for col in ['fantasy_points', 'yards', 'receptions', 'tds', 'targets', 'games']:
            season_df[f'{col}_prev'] = season_df.groupby('player')[col].shift(1)
        
        season_df['fp_next'] = season_df.groupby('player')['fantasy_points'].shift(-1)
        
        # Prepare training data
        feature_cols = [c for c in season_df.columns if c.endswith('_prev')]
        df_train = season_df.dropna(subset=feature_cols + ['fp_next'])
        
        if len(df_train) < 50:
            return pd.DataFrame()
        
        X = df_train[feature_cols]
        y = df_train['fp_next']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print(f"Fantasy Projection Model (R²): {r2_score(y_test, y_pred):.4f}")
        
        # Project for current players
        latest_season = season_df['season'].max()
        current = season_df[season_df['season'] == latest_season].copy()
        
        # These become the "prev" values for next season prediction
        for col in ['fantasy_points', 'yards', 'receptions', 'tds', 'targets', 'games']:
            current[f'{col}_prev'] = current[col]
        
        X_current = current[[c for c in feature_cols if c in current.columns]].fillna(0)
        current['projected_fp'] = model.predict(X_current)
        
        return current[['player', 'fantasy_points', 'projected_fp']].sort_values(
            'projected_fp', ascending=False
        ).head(50)


def run_full_wr_analysis():
    """Run complete WR analytics suite."""
    print("=" * 70)
    print("NFL WIDE RECEIVER ADVANCED ANALYTICS SUITE")
    print("=" * 70)
    
    # Load data
    loader = WRDataLoader()
    try:
        df = loader.load_all_seasons()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Season aggregation
    print("\n" + "=" * 50)
    print("AGGREGATING TO SEASON TOTALS")
    print("=" * 50)
    season_df = loader.aggregate_to_season()
    print(f"Season-level records: {len(season_df)}")
    
    # 1. Clustering Analysis
    print("\n" + "=" * 50)
    print("PLAYER STYLE CLUSTERING")
    print("=" * 50)
    try:
        cluster_analyzer = WRClusterAnalysis(n_clusters=5)
        clustered_df = cluster_analyzer.fit_clusters(season_df)
        profiles = cluster_analyzer.get_cluster_profiles(clustered_df)
        labels = cluster_analyzer.label_clusters(profiles)
        print("\nCluster Profiles:")
        for cluster, label in labels.items():
            count = profiles.loc[cluster, 'count']
            print(f"  Cluster {cluster}: {label} ({count:.0f} player-seasons)")
    except Exception as e:
        print(f"Clustering error: {e}")
    
    # 2. Consistency Analysis
    print("\n" + "=" * 50)
    print("CONSISTENCY ANALYSIS")
    print("=" * 50)
    consistency_analyzer = WRConsistencyAnalyzer(df)
    consistency = consistency_analyzer.calculate_consistency_scores()
    print("\nTop 10 Most Consistent WRs:")
    print(consistency[['player', 'yards_mean', 'consistency_score', 'games']].head(10).to_string(index=False))
    
    # 3. Weather Analysis
    print("\n" + "=" * 50)
    print("WEATHER IMPACT ANALYSIS")
    print("=" * 50)
    weather_analyzer = WRWeatherAnalyzer(df)
    weather_results = weather_analyzer.analyze_by_weather()
    if len(weather_results) > 0:
        print("\nPerformance by Weather Condition:")
        print(weather_results.to_string(index=False))
    
    temp_results = weather_analyzer.analyze_by_temperature()
    if len(temp_results) > 0:
        print("\nPerformance by Temperature:")
        print(temp_results.to_string(index=False))
    
    # 4. Situational Analysis
    print("\n" + "=" * 50)
    print("SITUATIONAL ANALYSIS")
    print("=" * 50)
    situational = WRSituationalAnalyzer(df)
    
    quarter_stats = situational.analyze_by_quarter()
    if len(quarter_stats) > 0:
        print("\nYards by Quarter:")
        print(quarter_stats.to_string(index=False))
    
    script_stats = situational.analyze_by_game_script()
    if len(script_stats) > 0:
        print("\nPerformance by Game Script:")
        print(script_stats.to_string(index=False))
    
    # 5. Fantasy Projections
    print("\n" + "=" * 50)
    print("FANTASY FOOTBALL PROJECTIONS (PPR)")
    print("=" * 50)
    fantasy = WRFantasyProjector(df)
    season_fp = fantasy.get_season_projections()
    latest_season = season_fp['season'].max()
    print(f"\nTop Fantasy WRs ({latest_season}):")
    print(season_fp[season_fp['season'] == latest_season].head(20).to_string(index=False))
    
    # 6. Breakout Prediction
    print("\n" + "=" * 50)
    print("BREAKOUT PLAYER PREDICTION")
    print("=" * 50)
    breakout = WRBreakoutPredictor()
    breakout_df = breakout.create_breakout_dataset(season_df)
    if len(breakout_df) > 50:
        breakout.train_model(breakout_df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_full_wr_analysis()
