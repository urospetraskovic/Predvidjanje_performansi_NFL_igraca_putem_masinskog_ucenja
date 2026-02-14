"""
Evaluation and Visualization Module for NFL Player Performance Prediction

This module provides:
- Comprehensive model evaluation metrics (MAE, RMSE, R²)
- SHAP-based feature importance analysis
- Visualization utilities for results
- Cross-validation utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, learning_curve
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Comprehensive evaluation class for NFL performance prediction models.
    """
    
    def __init__(self, model, model_name: str = "Model"):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model with predict method
            model_name: Name for display purposes
        """
        self.model = model
        self.model_name = model_name
        self.predictions = None
        self.actual = None
        self.metrics = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate standard regression metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with MAE, RMSE, R²
        """
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        }
        
        # Additional metrics
        metrics['Max_Error'] = np.max(np.abs(y_true - y_pred))
        metrics['Median_AE'] = np.median(np.abs(y_true - y_pred))
        
        return metrics
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Evaluation metrics
        """
        self.actual = y_test.values if isinstance(y_test, pd.Series) else y_test
        
        if hasattr(self.model, 'predict'):
            self.predictions = self.model.predict(X_test)
        else:
            self.predictions = self.model.model.predict(
                X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            )
        
        self.metrics = self.calculate_metrics(self.actual, self.predictions)
        
        return self.metrics
    
    def print_evaluation_report(self):
        """Print formatted evaluation report."""
        print(f"\n{'='*50}")
        print(f"Evaluation Report: {self.model_name}")
        print(f"{'='*50}")
        print(f"  Mean Absolute Error (MAE):     {self.metrics['MAE']:.2f}")
        print(f"  Root Mean Squared Error (RMSE): {self.metrics['RMSE']:.2f}")
        print(f"  R-Squared (R²):                 {self.metrics['R2']:.4f}")
        print(f"  Mean Absolute % Error (MAPE):   {self.metrics['MAPE']:.2f}%")
        print(f"  Maximum Error:                  {self.metrics['Max_Error']:.2f}")
        print(f"  Median Absolute Error:          {self.metrics['Median_AE']:.2f}")
        print(f"{'='*50}")
    
    def plot_predictions_vs_actual(self, save_path: Optional[str] = None):
        """
        Create scatter plot of predictions vs actual values.
        
        Args:
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(self.actual, self.predictions, alpha=0.5, edgecolors='none', s=50)
        
        # Perfect prediction line
        min_val = min(self.actual.min(), self.predictions.min())
        max_val = max(self.actual.max(), self.predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Annotations
        ax.set_xlabel('Actual Yards', fontsize=12)
        ax.set_ylabel('Predicted Yards', fontsize=12)
        ax.set_title(f'{self.model_name}: Predictions vs Actual\n'
                    f'R² = {self.metrics["R2"]:.4f}, RMSE = {self.metrics["RMSE"]:.2f}', 
                    fontsize=14)
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_residuals(self, save_path: Optional[str] = None):
        """
        Create residual analysis plots.
        
        Args:
            save_path: Path to save the figure
        """
        residuals = self.actual - self.predictions
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(self.predictions, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        
        # Q-Q plot (simplified)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot')
        
        plt.suptitle(f'{self.model_name}: Residual Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        plt.close()
    
    def plot_error_distribution(self, save_path: Optional[str] = None):
        """
        Plot error distribution analysis.
        
        Args:
            save_path: Path to save the figure
        """
        errors = np.abs(self.actual - self.predictions)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box plot of errors
        axes[0].boxplot(errors)
        axes[0].set_ylabel('Absolute Error')
        axes[0].set_title('Error Distribution (Box Plot)')
        
        # CDF of errors
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1].plot(sorted_errors, cdf)
        axes[1].set_xlabel('Absolute Error')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title('Cumulative Distribution of Errors')
        axes[1].axhline(y=0.9, color='r', linestyle='--', label='90th percentile')
        axes[1].legend()
        
        plt.suptitle(f'{self.model_name}: Error Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        plt.close()


def plot_feature_importance(feature_importance: pd.DataFrame, 
                           top_n: int = 15,
                           title: str = "Feature Importance",
                           save_path: Optional[str] = None):
    """
    Plot feature importance from trained model.
    
    Args:
        feature_importance: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save figure
    """
    top_features = feature_importance.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(range(len(top_features)), top_features['importance'].values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title(title)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    plt.close()


def plot_model_comparison(comparison_results: pd.DataFrame,
                         metric: str = 'test_rmse',
                         save_path: Optional[str] = None):
    """
    Plot comparison of different models.
    
    Args:
        comparison_results: DataFrame with model comparison results
        metric: Metric to compare
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE comparison
    sorted_by_rmse = comparison_results.sort_values('test_rmse')
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_by_rmse)))
    axes[0].barh(sorted_by_rmse['model'], sorted_by_rmse['test_rmse'], color=colors)
    axes[0].set_xlabel('Test RMSE')
    axes[0].set_title('Model Comparison: RMSE (lower is better)')
    axes[0].invert_yaxis()
    
    # MAE comparison
    sorted_by_mae = comparison_results.sort_values('test_mae')
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_by_mae)))
    axes[1].barh(sorted_by_mae['model'], sorted_by_mae['test_mae'], color=colors)
    axes[1].set_xlabel('Test MAE')
    axes[1].set_title('Model Comparison: MAE (lower is better)')
    axes[1].invert_yaxis()
    
    # R² comparison
    sorted_by_r2 = comparison_results.sort_values('test_r2', ascending=False)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_by_r2)))
    axes[2].barh(sorted_by_r2['model'], sorted_by_r2['test_r2'], color=colors)
    axes[2].set_xlabel('Test R²')
    axes[2].set_title('Model Comparison: R² (higher is better)')
    axes[2].invert_yaxis()
    
    plt.suptitle('Algorithm Comparison for NFL Performance Prediction', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    plt.close()


def plot_learning_curve(model, X: pd.DataFrame, y: pd.Series,
                       title: str = "Learning Curve",
                       save_path: Optional[str] = None):
    """
    Plot learning curve to assess overfitting.
    
    Args:
        model: Model to evaluate
        X: Feature matrix
        y: Target vector
        title: Plot title
        save_path: Path to save figure
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model.model if hasattr(model, 'model') else model,
        X, y,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    train_rmse = np.sqrt(-train_scores)
    test_rmse = np.sqrt(-test_scores)
    
    train_mean = train_rmse.mean(axis=1)
    train_std = train_rmse.std(axis=1)
    test_mean = test_rmse.mean(axis=1)
    test_std = test_rmse.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    ax.plot(train_sizes, train_mean, 'o-', label='Training RMSE')
    ax.plot(train_sizes, test_mean, 'o-', label='Cross-Validation RMSE')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('RMSE')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    plt.close()


def create_position_comparison_dashboard(results: Dict[str, Dict],
                                         save_path: Optional[str] = None):
    """
    Create a dashboard comparing results across positions.
    
    Args:
        results: Dictionary with position names as keys and metrics as values
        save_path: Path to save figure
    """
    positions = list(results.keys())
    metrics = ['MAE', 'RMSE', 'R2']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Bar chart for MAE
    mae_values = [results[pos].get('mae', results[pos].get('MAE', 0)) for pos in positions]
    axes[0, 0].bar(positions, mae_values, color='steelblue')
    axes[0, 0].set_ylabel('MAE (Yards)')
    axes[0, 0].set_title('Mean Absolute Error by Position')
    for i, v in enumerate(mae_values):
        axes[0, 0].text(i, v + 1, f'{v:.1f}', ha='center')
    
    # Bar chart for RMSE
    rmse_values = [results[pos].get('rmse', results[pos].get('RMSE', 0)) for pos in positions]
    axes[0, 1].bar(positions, rmse_values, color='coral')
    axes[0, 1].set_ylabel('RMSE (Yards)')
    axes[0, 1].set_title('Root Mean Squared Error by Position')
    for i, v in enumerate(rmse_values):
        axes[0, 1].text(i, v + 1, f'{v:.1f}', ha='center')
    
    # Bar chart for R²
    r2_values = [results[pos].get('r2', results[pos].get('R2', 0)) for pos in positions]
    axes[1, 0].bar(positions, r2_values, color='forestgreen')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('R-Squared by Position')
    axes[1, 0].set_ylim(0, 1)
    for i, v in enumerate(r2_values):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Summary table
    axes[1, 1].axis('off')
    table_data = []
    for pos in positions:
        table_data.append([
            pos,
            f"{results[pos].get('mae', results[pos].get('MAE', 0)):.1f}",
            f"{results[pos].get('rmse', results[pos].get('RMSE', 0)):.1f}",
            f"{results[pos].get('r2', results[pos].get('R2', 0)):.4f}"
        ])
    
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['Position', 'MAE', 'RMSE', 'R²'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Summary Statistics', pad=20)
    
    plt.suptitle('NFL Performance Prediction: Position Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    plt.close()


def generate_full_report(model, X_test, y_test, model_name: str,
                        feature_importance: Optional[pd.DataFrame] = None,
                        output_dir: str = 'results'):
    """
    Generate a comprehensive evaluation report with all visualizations.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name for the model
        feature_importance: Feature importance DataFrame
        output_dir: Directory for saving plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = ModelEvaluator(model, model_name)
    metrics = evaluator.evaluate(X_test, y_test)
    
    # Print report
    evaluator.print_evaluation_report()
    
    # Generate plots
    evaluator.plot_predictions_vs_actual(
        save_path=f"{output_dir}/{model_name}_predictions.png"
    )
    evaluator.plot_residuals(
        save_path=f"{output_dir}/{model_name}_residuals.png"
    )
    evaluator.plot_error_distribution(
        save_path=f"{output_dir}/{model_name}_errors.png"
    )
    
    if feature_importance is not None:
        plot_feature_importance(
            feature_importance,
            title=f"{model_name} - Feature Importance",
            save_path=f"{output_dir}/{model_name}_feature_importance.png"
        )
    
    return metrics


if __name__ == "__main__":
    print("NFL Evaluation Module")
    print("Provides comprehensive evaluation and visualization tools for model analysis.")
