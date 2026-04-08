"""Finalna komparativna tabela svih modela i strategija."""

import numpy as np
import pandas as pd
from IPython.display import HTML, display


def show_scrollable_df(df, height=420):
    """Prikazi DataFrame u skrolabilnom HTML kontejneru."""
    display(HTML(
        f"<div style='max-height:{height}px;overflow:auto;border:1px solid #cfcfcf;"
        f"border-radius:6px;padding:4px;'>"
        + df.to_html(index=True, border=0)
        + "</div>"
    ))


def build_final_comparison_table(test_results=None,
                                  random_results=None,
                                  grid_results=None,
                                  bayesian_results=None):
    """
    Spaja rezultate iz svih izvora u jednu DataFrame.
    Svaki argument je opcionalan (None = preskoci).
    """
    rows = []

    if test_results and isinstance(test_results, dict):
        for pos, df_pos in test_results.items():
            for model_name, row in df_pos.iterrows():
                rows.append({
                    'Position': pos,
                    'Model':    model_name,
                    'Strategy': 'Final Test (GridSearchCV)',
                    'MAE':  float(row['MAE']),
                    'RMSE': float(row['RMSE']),
                    'R2':   float(row['R2']),
                })

    for src, strategy_name in [
        (random_results,   'Random Search'),
        (grid_results,     'Grid Search'),
        (bayesian_results, 'Bayesian Optimization'),
    ]:
        if not src or not isinstance(src, dict):
            continue
        for pos, pos_dict in src.items():
            for model_name, vals in pos_dict.items():
                rows.append({
                    'Position': pos,
                    'Model':    model_name,
                    'Strategy': strategy_name,
                    'MAE':  float(vals.get('test_mae', np.nan)),
                    'RMSE': float(vals.get('test_rmse', np.nan)),
                    'R2':   float(vals.get('test_r2', np.nan)),
                })

    if not rows:
        print("Nema dostupnih rezultata.")
        return pd.DataFrame()

    df = (pd.DataFrame(rows)
          .dropna(subset=['RMSE'])
          .sort_values(['Position', 'Model', 'Strategy', 'RMSE'])
          .reset_index(drop=True))
    return df


def show_final_comparison(test_results=None, random_results=None,
                           grid_results=None, bayesian_results=None,
                           height=520):
    """Gradi i prikazuje finalnu komparativnu tabelu."""
    df = build_final_comparison_table(
        test_results, random_results, grid_results, bayesian_results
    )
    if df.empty:
        return df

    show_scrollable_df(df, height=height)
    print(f"\nUkupno redova: {len(df)}")

    best = df.loc[df['RMSE'].idxmin()]
    print(f"\nGlobalno najbolji (najmanji RMSE):")
    print(f"  {best['Position']} | {best['Model']} | {best['Strategy']} "
          f"| MAE={best['MAE']:.3f}  RMSE={best['RMSE']:.3f}  R2={best['R2']:.3f}")
    return df
