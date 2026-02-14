"""
NFL Player Performance Models

This module contains regression models for predicting NFL player performance:
- Linear Regression (baseline)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- ElasticNet (L1 + L2 regularization)
- K-Nearest Neighbors Regressor
- Random Forest Regressor (main model)
- Multi-Output Regression for simultaneous predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


class NFLPerformanceModel:
    """
    Comprehensive model class for predicting NFL player performance.
    Supports multiple regression algorithms and hyperparameter tuning.
    """
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize the performance model.
        
        Args:
            model_type: Type of model to use
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected model type."""
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(random_state=self.random_state),
            'lasso': Lasso(random_state=self.random_state, max_iter=10000),
            'elasticnet': ElasticNet(random_state=self.random_state, max_iter=10000),
            'knn': KNeighborsRegressor(),
            'random_forest': RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(random_state=self.random_state)
        }
        
        if self.model_type not in models:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           f"Available: {list(models.keys())}")
        
        self.model = models[self.model_type]
    
    def get_hyperparameter_grid(self) -> Dict:
        """Get hyperparameter grid for the current model type."""
        grids = {
            'linear': {},
            'ridge': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            'elasticnet': {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        return grids.get(self.model_type, {})
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              tune_hyperparameters: bool = False,
              cv_folds: int = 5) -> Dict[str, float]:
        """
        Train the model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target vector
            tune_hyperparameters: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training metrics
        """
        # Convert to numpy arrays if needed
        X_train = X.values if isinstance(X, pd.DataFrame) else X
        y_train = y.values if isinstance(y, pd.Series) else y
        
        if tune_hyperparameters and self.get_hyperparameter_grid():
            print(f"Tuning hyperparameters for {self.model_type}...")
            grid_search = GridSearchCV(
                self.model,
                self.get_hyperparameter_grid(),
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"Best parameters: {self.best_params}")
        else:
            self.model.fit(X_train, y_train)
        
        # Calculate cross-validation scores
        cv_scores = cross_val_score(self.model, X_train, y_train, 
                                    cv=cv_folds, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else list(range(X.shape[1]))
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Training predictions
        y_pred = self.model.predict(X_train)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred)),
            'train_r2': r2_score(y_train, y_pred),
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std()
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        X_pred = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_pred)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_true = y.values if isinstance(y, pd.Series) else y
        y_pred = self.predict(X)
        
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        return self.feature_importance
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.model_type = data['model_type']
        self.best_params = data['best_params']
        self.feature_importance = data['feature_importance']
        print(f"Model loaded from {filepath}")


class MultiOutputNFLModel:
    """
    Multi-output regression model for predicting multiple performance metrics simultaneously.
    This follows the methodology from Elimam et al. (2025) for multi-output regression.
    """
    
    def __init__(self, base_model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize multi-output model.
        
        Args:
            base_model_type: Type of base estimator
            random_state: Random seed
        """
        self.base_model_type = base_model_type
        self.random_state = random_state
        self.model = None
        self.target_columns = []
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the base model for multi-output regression."""
        base_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(random_state=self.random_state),
            'knn': KNeighborsRegressor(n_neighbors=5, weights='distance'),
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        base_estimator = base_models.get(self.base_model_type, 
                                          base_models['random_forest'])
        self.model = MultiOutputRegressor(base_estimator, n_jobs=-1)
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """
        Train multi-output model.
        
        Args:
            X: Feature matrix
            y: Multi-target DataFrame
            
        Returns:
            Training metrics for each target
        """
        self.target_columns = y.columns.tolist()
        
        X_train = X.values if isinstance(X, pd.DataFrame) else X
        y_train = y.values if isinstance(y, pd.DataFrame) else y
        
        self.model.fit(X_train, y_train)
        
        # Calculate metrics for each target
        y_pred = self.model.predict(X_train)
        
        metrics = {}
        for i, target in enumerate(self.target_columns):
            metrics[target] = {
                'train_mae': mean_absolute_error(y_train[:, i], y_pred[:, i]),
                'train_rmse': np.sqrt(mean_squared_error(y_train[:, i], y_pred[:, i])),
                'train_r2': r2_score(y_train[:, i], y_pred[:, i])
            }
        
        # Average metrics
        metrics['average'] = {
            'train_mae': np.mean([m['train_mae'] for m in metrics.values() if isinstance(m, dict)]),
            'train_rmse': np.mean([m['train_rmse'] for m in metrics.values() if isinstance(m, dict)]),
            'train_r2': np.mean([m['train_r2'] for m in metrics.values() if isinstance(m, dict)])
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make multi-output predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with predictions for each target
        """
        X_pred = X.values if isinstance(X, pd.DataFrame) else X
        predictions = self.model.predict(X_pred)
        
        return pd.DataFrame(predictions, columns=self.target_columns)
    
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate multi-output model.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Evaluation metrics for each target
        """
        y_true = y.values if isinstance(y, pd.DataFrame) else y
        y_pred = self.model.predict(X.values if isinstance(X, pd.DataFrame) else X)
        
        metrics = {}
        for i, target in enumerate(self.target_columns):
            metrics[target] = {
                'mae': mean_absolute_error(y_true[:, i], y_pred[:, i]),
                'rmse': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
                'r2': r2_score(y_true[:, i], y_pred[:, i])
            }
        
        # Average RMSE (aRMSE) as used in Elimam et al.
        metrics['average'] = {
            'mae': np.mean([m['mae'] for m in metrics.values() if isinstance(m, dict)]),
            'rmse': np.mean([m['rmse'] for m in metrics.values() if isinstance(m, dict)]),
            'r2': np.mean([m['r2'] for m in metrics.values() if isinstance(m, dict)])
        }
        
        return metrics


def compare_models(X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   tune_hyperparameters: bool = False) -> pd.DataFrame:
    """
    Compare all available model types on the given data.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        tune_hyperparameters: Whether to tune hyperparameters
        
    Returns:
        DataFrame with comparison results
    """
    model_types = ['linear', 'ridge', 'lasso', 'elasticnet', 'knn', 
                   'random_forest', 'gradient_boosting']
    
    results = []
    
    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        model = NFLPerformanceModel(model_type=model_type)
        
        train_metrics = model.train(X_train, y_train, 
                                    tune_hyperparameters=tune_hyperparameters)
        test_metrics = model.evaluate(X_test, y_test)
        
        results.append({
            'model': model_type,
            'train_mae': train_metrics['train_mae'],
            'train_rmse': train_metrics['train_rmse'],
            'train_r2': train_metrics['train_r2'],
            'test_mae': test_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'test_r2': test_metrics['r2'],
            'cv_rmse': train_metrics['cv_rmse_mean']
        })
    
    return pd.DataFrame(results).sort_values('test_rmse')


def train_position_specific_model(position: str, X_train: pd.DataFrame, 
                                   y_train: pd.Series, X_test: pd.DataFrame,
                                   y_test: pd.Series) -> Tuple[NFLPerformanceModel, Dict]:
    """
    Train a position-specific model with the best algorithm (Random Forest).
    
    Args:
        position: Player position (QB, RB, WR, TE)
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        Trained model and evaluation metrics
    """
    print(f"\n{'='*50}")
    print(f"Training {position} Performance Model")
    print(f"{'='*50}")
    
    # Use Random Forest as the primary model (as per literature review)
    model = NFLPerformanceModel(model_type='random_forest')
    
    # Train with hyperparameter tuning
    train_metrics = model.train(X_train, y_train, tune_hyperparameters=True)
    
    # Evaluate on test set
    test_metrics = model.evaluate(X_test, y_test)
    
    # Print results
    print(f"\n{position} Model Results:")
    print(f"  Training MAE:  {train_metrics['train_mae']:.2f}")
    print(f"  Training RMSE: {train_metrics['train_rmse']:.2f}")
    print(f"  Training R²:   {train_metrics['train_r2']:.4f}")
    print(f"  Test MAE:      {test_metrics['mae']:.2f}")
    print(f"  Test RMSE:     {test_metrics['rmse']:.2f}")
    print(f"  Test R²:       {test_metrics['r2']:.4f}")
    
    # Print feature importance
    if model.feature_importance is not None:
        print(f"\nTop 10 Most Important Features for {position}:")
        print(model.feature_importance.head(10).to_string(index=False))
    
    return model, {**train_metrics, **test_metrics}


if __name__ == "__main__":
    # Example usage
    print("NFL Performance Models Module")
    print("Available model types:")
    print("  - linear: Linear Regression")
    print("  - ridge: Ridge Regression (L2)")
    print("  - lasso: Lasso Regression (L1)")
    print("  - elasticnet: ElasticNet (L1+L2)")
    print("  - knn: K-Nearest Neighbors")
    print("  - random_forest: Random Forest (recommended)")
    print("  - gradient_boosting: Gradient Boosting")
