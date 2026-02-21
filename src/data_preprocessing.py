"""
Data Preprocessing Module for NFL Player Performance Prediction

This module handles:
- Loading and merging datasets
- Outlier detection and treatment (IQR method)
- Missing value handling
- Feature normalization and scaling
- Categorical variable encoding
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class NFLDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for NFL player statistics.
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns: List[str] = []
        self.target_column: str = ''
        
    def load_qb_data(self, filepath: str) -> pd.DataFrame:
        """Load quarterback statistics from CSV file."""
        df = pd.read_csv(filepath)
        print(f"Loaded QB data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def load_rb_data(self, filepath: str) -> pd.DataFrame:
        """Load running back statistics from CSV file."""
        df = pd.read_csv(filepath)
        print(f"Loaded RB data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def load_wr_data(self, filepath: str) -> pd.DataFrame:
        """Load wide receiver statistics from CSV file."""
        df = pd.read_csv(filepath)
        print(f"Loaded WR data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def load_te_data(self, filepath: str) -> pd.DataFrame:
        """Load tight end statistics from CSV file."""
        df = pd.read_csv(filepath)
        print(f"Loaded TE data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def detect_outliers_iqr(self, df: pd.DataFrame, columns: List[str], 
                            factor: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using the IQR (Interquartile Range) method.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            factor: IQR multiplier (default 1.5 for mild outliers, 3.0 for extreme)
            
        Returns:
            DataFrame with boolean mask for outliers
        """
        outlier_mask = pd.DataFrame(index=df.index)
        
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        return outlier_mask
    
    def treat_outliers(self, df: pd.DataFrame, columns: List[str], 
                       method: str = 'clip', factor: float = 1.5) -> pd.DataFrame:
        """
        Treat outliers using specified method.
        
        Args:
            df: Input DataFrame
            columns: Columns to treat
            method: 'clip' to cap values, 'remove' to drop rows, 'median' to replace
            factor: IQR multiplier
            
        Returns:
            DataFrame with treated outliers
        """
        df_treated = df.copy()
        
        for col in columns:
            if col in df_treated.columns and df_treated[col].dtype in ['int64', 'float64']:
                Q1 = df_treated[col].quantile(0.25)
                Q3 = df_treated[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                if method == 'clip':
                    df_treated[col] = df_treated[col].clip(lower=lower_bound, upper=upper_bound)
                elif method == 'median':
                    median_val = df_treated[col].median()
                    mask = (df_treated[col] < lower_bound) | (df_treated[col] > upper_bound)
                    df_treated.loc[mask, col] = median_val
                elif method == 'remove':
                    mask = (df_treated[col] >= lower_bound) & (df_treated[col] <= upper_bound)
                    df_treated = df_treated[mask]
        
        return df_treated
    
    def handle_missing_values(self, df: pd.DataFrame, 
                              numeric_strategy: str = 'median',
                              categorical_strategy: str = 'most_frequent') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            numeric_strategy: Strategy for numeric columns ('mean', 'median', 'most_frequent')
            categorical_strategy: Strategy for categorical columns
            
        Returns:
            DataFrame with imputed values
        """
        df_imputed = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df_imputed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df_imputed.select_dtypes(include=['object']).columns.tolist()
        
        # Impute numeric columns one by one to avoid column mismatch issues
        if numeric_cols:
            for col in numeric_cols:
                if df_imputed[col].isna().any():
                    if numeric_strategy == 'median':
                        fill_val = df_imputed[col].median()
                    elif numeric_strategy == 'mean':
                        fill_val = df_imputed[col].mean()
                    else:
                        fill_val = df_imputed[col].mode().iloc[0] if not df_imputed[col].mode().empty else 0
                    
                    # If all values are NaN, fill with 0
                    if pd.isna(fill_val):
                        fill_val = 0
                    df_imputed[col] = df_imputed[col].fillna(fill_val)
        
        # Impute categorical columns
        if categorical_cols:
            for col in categorical_cols:
                if df_imputed[col].isna().any():
                    mode_val = df_imputed[col].mode()
                    fill_val = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
                    df_imputed[col] = df_imputed[col].fillna(fill_val)
        
        return df_imputed
    
    def encode_categorical(self, df: pd.DataFrame, 
                           columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical variables using Label Encoding.
        
        Args:
            df: Input DataFrame
            columns: Specific columns to encode (if None, encode all object columns)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                # Handle NaN values before encoding
                df_encoded[col] = df_encoded[col].fillna('Unknown')
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, columns: List[str], 
                       fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features using the configured scaler.
        
        Args:
            df: Input DataFrame
            columns: Columns to scale
            fit: Whether to fit the scaler (True for training, False for test)
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        valid_cols = [col for col in columns if col in df_scaled.columns 
                      and df_scaled[col].dtype in ['int64', 'float64']]
        
        if valid_cols:
            if fit:
                df_scaled[valid_cols] = self.scaler.fit_transform(df_scaled[valid_cols])
            else:
                df_scaled[valid_cols] = self.scaler.transform(df_scaled[valid_cols])
        
        return df_scaled
    
    def get_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get comprehensive statistics for all features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with statistics for each feature
        """
        stats = pd.DataFrame()
        
        for col in df.columns:
            col_stats = {
                'column': col,
                'dtype': str(df[col].dtype),
                'missing_count': df[col].isna().sum(),
                'missing_pct': (df[col].isna().sum() / len(df)) * 100,
                'unique_values': df[col].nunique()
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_stats.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                })
            
            stats = pd.concat([stats, pd.DataFrame([col_stats])], ignore_index=True)
        
        return stats
    
    def prepare_qb_features(self, df: pd.DataFrame, 
                            target: str = 'Yds') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare quarterback features for modeling.
        
        Args:
            df: Raw QB DataFrame
            target: Target column name
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Define relevant QB features for passing yards prediction
        feature_cols = [
            'Age', 'G', 'GS', 'Cmp', 'Att', 'Cmp%', 'TD', 'TD%', 'Int', 'Int%',
            '1D', 'Succ%', 'Lng', 'Y/A', 'AY/A', 'Y/C', 'Y/G', 'Rate', 'Sk',
            'Yds_Lost', 'Sk%', 'NY/A', 'ANY/A', 
            'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rush_1D', 'Rush_Y/A', 'Rush_Y/G'
        ]
        
        # Advanced features if available
        advanced_cols = [
            'pass_air_yds', 'pass_yac', 'pass_drops', 'pass_drop_pct',
            'pass_poor_throws', 'pass_poor_throw_pct', 'pass_on_target', 
            'pass_on_target_pct', 'pocket_time', 'pass_blitzed', 'pass_hurried',
            'pass_hits', 'pass_pressured', 'pass_pressured_pct', 'rush_scrambles'
        ]
        
        # Use available columns
        available_features = [col for col in feature_cols + advanced_cols 
                             if col in df.columns]
        
        # Filter to rows with target
        df_filtered = df[df[target].notna()].copy()
        
        # Get features and target
        X = df_filtered[available_features].copy()
        y = df_filtered[target].copy()
        
        return X, y
    
    def prepare_rb_features(self, df: pd.DataFrame, 
                            target: str = 'Yds') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare running back features for modeling.
        
        Args:
            df: Raw RB DataFrame
            target: Target column name (rushing yards)
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Define relevant RB features for rushing yards prediction
        feature_cols = [
            'Age', 'G', 'GS', 'Att', 'TD', '1D', 'Succ%', 'Lng', 'Y/A', 'Y/G', 'A/G',
            'Tgt', 'Rec', 'Yds.1', 'Y/R', 'TD.1', '1D.1', 'rec_success', 'rec_long',
            'R/G', 'Y/G.1', 'catch_pct', 'Y/Tgt', 'Touch', 'yds_per_touch'
        ]
        
        # Advanced features if available
        advanced_cols = [
            'YBC', 'YBC/Att', 'YAC', 'YAC/Att', 'BrkTkl', 'Att/Br',
            'YBC.1', 'YBC/R', 'YAC.1', 'YAC/R', 'ADOT', 'Drop', 'Drop%'
        ]
        
        # Use available columns
        available_features = [col for col in feature_cols + advanced_cols 
                             if col in df.columns]
        
        # Filter to rows with target
        df_filtered = df[df[target].notna()].copy()
        
        # Get features and target
        X = df_filtered[available_features].copy()
        y = df_filtered[target].copy()
        
        return X, y
    
    def prepare_wr_features(self, df: pd.DataFrame, 
                            target: str = 'receiving_yards') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare wide receiver features for modeling.
        
        Args:
            df: Raw WR DataFrame
            target: Target column name (receiving yards)
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Features from WR dataset
        feature_cols = [
            'targets', 'receptions', 'air_yards', 'yac', 'tds', 'epa', 'wpa',
            'avg_depth', 'catch_rate', 'team_pass_attempts', 'team_air_yards',
            'air_yard_share', 'target_share', 'yards_per_target', 
            'red_zone_targets', 'end_zone_targets', 'third_down_targets',
            'fourth_down_targets', 'high_leverage_targets', 'qb_completions',
            'qb_attempts', 'qb_cpoe', 'qb_comp_pct', 'adot', 'yac_per_reception',
            'td_rate', 'explosive_plays', 'first_downs', 'success_rate',
            'big_play_rate'
        ]
        
        # Use available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Filter to rows with target
        if target in df.columns:
            df_filtered = df[df[target].notna()].copy()
        else:
            df_filtered = df.copy()
        
        # Get features and target
        X = df_filtered[available_features].copy()
        y = df_filtered[target].copy() if target in df.columns else pd.Series()
        
        return X, y
    
    def prepare_te_features(self, df: pd.DataFrame, 
                            target: str = 'Yds') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare tight end features for modeling.
        
        Args:
            df: Raw TE DataFrame
            target: Target column name (receiving yards)
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Define relevant TE features for receiving yards prediction
        feature_cols = [
            'Age', 'G', 'GS', 'Tgt', 'Rec', 'Y/R', 'TD', '1D', 'rec_success',
            'rec_long', 'R/G', 'Y/G', 'catch_pct', 'Y/Tgt', 'Touch', 'yds_per_touch'
        ]
        
        # Rushing features if available
        rushing_cols = ['Att', 'Yds.1', 'TD.1', '1D.1', 'Succ%', 'Lng', 'Y/A']
        
        # Use available columns
        available_features = [col for col in feature_cols + rushing_cols 
                             if col in df.columns]
        
        # Filter to rows with target
        df_filtered = df[df[target].notna()].copy()
        
        # Get features and target
        X = df_filtered[available_features].copy()
        y = df_filtered[target].copy()
        
        return X, y
    
    def full_preprocessing_pipeline(self, df: pd.DataFrame, 
                                    feature_cols: List[str],
                                    target_col: str,
                                    treat_outliers: bool = True,
                                    scale: bool = True,
                                    fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Run full preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            target_col: Target column name
            treat_outliers: Whether to treat outliers
            scale: Whether to scale features
            fit: Whether to fit transformers (True for train, False for test)
            
        Returns:
            Tuple of (preprocessed features, target)
        """
        df_processed = df.copy()
        
        # 1. Handle missing values
        if fit:
            print("Handling missing values...")
        df_processed = self.handle_missing_values(df_processed)
        
        # 2. Encode categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols 
                           if col not in [target_col, 'Player', 'PlayerID', 'Awards', 'awards']]
        if fit:
            print("Encoding categorical variables...")
            df_processed = self.encode_categorical(df_processed, categorical_cols)
        else:
            # Apply existing encoders for test data
            for col in categorical_cols:
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].fillna('Unknown')
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        known_labels = set(le.classes_)
                        # Map unseen labels to the most common training label
                        most_common = le.classes_[0]
                        df_processed[col] = df_processed[col].astype(str).apply(
                            lambda x, kl=known_labels, mc=most_common: x if x in kl else mc
                        )
                        df_processed[col] = le.transform(df_processed[col])
                    else:
                        # No encoder fitted for this column - drop it
                        df_processed = df_processed.drop(columns=[col])
        
        # 3. Treat outliers
        if treat_outliers:
            if fit:
                print("Treating outliers...")
            numeric_features = [col for col in feature_cols 
                               if col in df_processed.columns 
                               and df_processed[col].dtype in ['int64', 'float64']]
            df_processed = self.treat_outliers(df_processed, numeric_features, method='clip')
        
        # 4. Extract features and target
        available_features = [col for col in feature_cols if col in df_processed.columns]
        self.feature_columns = available_features
        X = df_processed[available_features].copy()
        y = df_processed[target_col].copy() if target_col in df_processed.columns else pd.Series()
        
        # 5. Scale features
        if scale:
            if fit:
                print("Scaling features...")
            X = self.scale_features(X, available_features, fit=fit)
        
        if fit:
            print(f"Preprocessing complete. Features shape: {X.shape}")
        
        return X, y


def create_time_based_split(df: pd.DataFrame, season_col: str = 'Season',
                            test_seasons: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create chronological train-test split based on seasons.
    
    Args:
        df: Input DataFrame
        season_col: Column containing season information
        test_seasons: Number of most recent seasons for testing
        
    Returns:
        Tuple of (train DataFrame, test DataFrame)
    """
    unique_seasons = sorted(df[season_col].unique())
    
    if len(unique_seasons) <= test_seasons:
        raise ValueError("Not enough seasons for the specified test split")
    
    test_season_cutoff = unique_seasons[-test_seasons]
    
    train_df = df[df[season_col] < test_season_cutoff].copy()
    test_df = df[df[season_col] >= test_season_cutoff].copy()
    
    print(f"Training seasons: {unique_seasons[:-test_seasons]}")
    print(f"Testing seasons: {unique_seasons[-test_seasons:]}")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    return train_df, test_df


if __name__ == "__main__":
    # Example usage
    preprocessor = NFLDataPreprocessor(scaler_type='standard')
    
    # Test with QB data
    try:
        qb_df = preprocessor.load_qb_data('data/processed/all_qb_stats_full.csv')
        stats = preprocessor.get_feature_statistics(qb_df)
        print("\nQB Data Statistics:")
        print(stats.head(10))
    except FileNotFoundError:
        print("QB data file not found. Please ensure the data files are in the correct location.")
