"""Vizuelizacija rezultata pretrage hiperparametara."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

POSITIONS = ['QB', 'RB', 'TE', 'WR']


def plot_error_metrics(grid_results, random_results, save_path='../results/error_metrics_comparison.png'):
    """
    4-panel grafik: MAE, RMSE, R² i poboljsanje Grid vs Random Search.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Error Metrics: CV vs Test po modelu i poziciji',
                 fontsize=16, fontweight='bold')

    for ax_idx, metric in enumerate(['MAE', 'RMSE']):
        ax = axes[0, ax_idx]
        cv_vals, test_vals, labels = [], [], []

        for pos in POSITIONS:
            if pos not in grid_results:
                continue
            for model_name in sorted(grid_results[pos]):
                if metric == 'MAE':
                    cv_vals.append(grid_results[pos][model_name].get('cv_mae', 0))
                    test_vals.append(grid_results[pos][model_name].get('test_mae', 0))
                else:
                    cv_vals.append(np.sqrt(grid_results[pos][model_name].get('cv_mae', 0) ** 2))
                    test_vals.append(grid_results[pos][model_name].get('test_rmse', 0))
                labels.append(f"{pos}-{model_name}")

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, cv_vals,   w, label='CV (Validation)', alpha=0.8, color='skyblue')
        ax.bar(x + w / 2, test_vals, w, label='Test',             alpha=0.8, color='orange')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    # R²
    ax = axes[1, 0]
    r2_vals, labels = [], []
    for pos in POSITIONS:
        if pos not in grid_results:
            continue
        for model_name in sorted(grid_results[pos]):
            r2_vals.append(grid_results[pos][model_name].get('test_r2', 0))
            labels.append(f"{pos}-{model_name}")
    x = np.arange(len(r2_vals))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(r2_vals)))
    ax.bar(x, r2_vals, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('R² Score Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-0.3, 1.0)

    # Poboljsanje Grid vs Random
    ax = axes[1, 1]
    improvements, labels = [], []
    for pos in POSITIONS:
        if pos not in grid_results or pos not in random_results:
            continue
        for model_name in sorted(grid_results[pos]):
            if model_name not in random_results.get(pos, {}):
                continue
            r_rmse = random_results[pos][model_name].get('test_rmse', 0)
            g_rmse = grid_results[pos][model_name].get('test_rmse', 0)
            if r_rmse > 0:
                improvements.append((r_rmse - g_rmse) / r_rmse * 100)
                labels.append(f"{pos}-{model_name}")
    x = np.arange(len(improvements))
    colors = ['green' if v > 0 else 'red' for v in improvements]
    ax.bar(x, improvements, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Grid Search Improvement vs Random Search', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Grafika sacuvana: {save_path}")


def plot_errors_by_position(grid_results, save_path='../results/error_by_position.png'):
    """
    2x2 grafici (jedan po poziciji): CV MAE / Test RMSE / Test R² za svaki model.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, pos in enumerate(POSITIONS):
        ax = axes[idx]
        if pos not in grid_results:
            continue
        models   = sorted(grid_results[pos])
        x        = np.arange(len(models))
        width    = 0.25
        cv_maes    = [grid_results[pos][m].get('cv_mae', 0) for m in models]
        test_rmses = [grid_results[pos][m].get('test_rmse', 0) for m in models]
        test_r2s   = [grid_results[pos][m].get('test_r2', 0) for m in models]

        ax.bar(x - width, cv_maes,    width, label='CV MAE',          alpha=0.8, color='#FF6B6B')
        ax.bar(x,         test_rmses, width, label='Test RMSE',        alpha=0.8, color='#4ECDC4')
        ax.bar(x + width, test_r2s,   width, label='Test R² (scaled)', alpha=0.8, color='#45B7D1')

        for i, (cv, rmse, r2) in enumerate(zip(cv_maes, test_rmses, test_r2s)):
            ax.text(i - width, cv   + 0.5, f'{cv:.2f}',   ha='center', fontsize=8)
            ax.text(i,         rmse + 0.5, f'{rmse:.2f}', ha='center', fontsize=8)
            ax.text(i + width, r2   + 0.05, f'{r2:.3f}',  ha='center', fontsize=8)

        ax.set_ylabel('Error / Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{pos} — Model Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Grafika sacuvana: {save_path}")


def plot_strategy_comparison(comparison_df):
    """
    Poredjenje sve tri strategije: RMSE i R² pivot grafici.
    """
    if comparison_df.empty:
        print("Nema podataka za prikaz.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    rmse_pivot = comparison_df.pivot_table(index='pos_model', columns='strategy', values='test_rmse')
    rmse_pivot.plot(kind='bar', ax=axes[0], color=['#3498db', '#2ecc71', '#e74c3c'], width=0.8)
    axes[0].set_title('Test RMSE — Poredjenje sve tri strategije', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('RMSE (yds/g)')
    axes[0].legend(title='Strategija')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

    r2_pivot = comparison_df.pivot_table(index='pos_model', columns='strategy', values='test_r2')
    r2_pivot.plot(kind='bar', ax=axes[1], color=['#3498db', '#2ecc71', '#e74c3c'], width=0.8)
    axes[1].set_title('Test R² — Poredjenje sve tri strategije', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('R²')
    axes[1].legend(title='Strategija')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    plt.suptitle('Tri optimizacione strategije — sve pozicije',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
