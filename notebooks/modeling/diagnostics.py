"""
Dijagnostika overfitting/underfitting: Train vs Validation vs Test RMSE kroz vreme.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from IPython.display import HTML, display

POSITIONS = ['QB', 'RB', 'TE', 'WR']


def rmse_r2_on_scale(y_true, y_pred, is_wr):
    """RMSE i R² u originalnim jedinicama (WR: inverse log1p)."""
    if is_wr:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return rmse, r2


def temporal_validation_metrics(estimator, X, y, is_wr, n_splits=3):
    """TimeSeriesSplit CV — vraca prosecne RMSE i R² na validacionim foldovima."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_vals, r2_vals = [], []
    for tr_idx, va_idx in tscv.split(X):
        m = clone(estimator)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = m.predict(X.iloc[va_idx])
        rmse_va, r2_va = rmse_r2_on_scale(y.iloc[va_idx], pred, is_wr)
        rmse_vals.append(rmse_va)
        r2_vals.append(r2_va)
    return float(np.mean(rmse_vals)), float(np.mean(r2_vals))


def diagnose_fold(train_rmse, val_rmse, test_rmse, train_r2, val_r2, test_r2):
    """Klasifikuje fold kao Underfit / Overfit / Balanced."""
    gap_val  = (val_rmse  - train_rmse) / max(train_rmse, 1e-9)
    gap_test = (test_rmse - train_rmse) / max(train_rmse, 1e-9)
    if train_r2 < 0.15 and val_r2 < 0.15 and test_r2 < 0.15:
        return 'Underfit'
    if gap_val > 0.30 and gap_test > 0.30:
        return 'Overfit'
    return 'Balanced'


def run_diagnostics(lagged, processed, model_configs, folds,
                    cal_best_per_pos, pos_scol, pos_id_cols,
                    prepare_fold_data, impute_lag2_and_median,
                    scale_fold, fit_fold_model, binary_cols,
                    cal_model_configs):
    """
    Pokrece dijagnostiku za sve pozicije i foldove.
    Vraca diag_df (DataFrame) i prikazuje HTML tabelu + grafike.
    """
    records = []

    for pos in POSITIONS:
        if pos not in lagged:
            continue
        best_model_name = cal_best_per_pos.get(pos)
        if best_model_name is None or best_model_name not in cal_model_configs:
            continue

        model_cfg = cal_model_configs[best_model_name]
        scol      = pos_scol[pos]
        id_cols   = pos_id_cols[pos]
        df_full   = lagged[pos]
        is_wr     = (pos == 'WR')
        feat_cols = [c for c in df_full.columns if c not in id_cols + ['target']]

        print(f'\n{pos} — dijagnosticki model: {best_model_name}')

        for fold in folds:
            tr_df, te_df, X_tr, X_te, y_tr, y_te = prepare_fold_data(
                df_full, scol, feat_cols, fold['train_end'], fold['test_year']
            )
            if tr_df is None:
                continue

            X_tr_i, X_te_i = impute_lag2_and_median(X_tr, X_te)
            X_tr_s, X_te_s = scale_fold(X_tr_i, X_te_i, binary_cols)

            fitted, best_params = fit_fold_model(model_cfg, X_tr_s, y_tr, len(X_tr_s))

            train_rmse, train_r2 = rmse_r2_on_scale(y_tr, fitted.predict(X_tr_s), is_wr)
            test_rmse,  test_r2  = rmse_r2_on_scale(y_te, fitted.predict(X_te_s), is_wr)
            val_rmse,   val_r2   = temporal_validation_metrics(fitted, X_tr_s, y_tr, is_wr)

            diagnosis = diagnose_fold(train_rmse, val_rmse, test_rmse,
                                      train_r2, val_r2, test_r2)
            records.append({
                'position':   pos,
                'model':      best_model_name,
                'fold':       fold['name'],
                'test_year':  fold['test_year'],
                'train_rmse': train_rmse,
                'val_rmse':   val_rmse,
                'test_rmse':  test_rmse,
                'train_r2':   train_r2,
                'val_r2':     val_r2,
                'test_r2':    test_r2,
                'diagnosis':  diagnosis,
                'best_params': str(best_params),
            })

    diag_df = pd.DataFrame(records)

    if diag_df.empty:
        print('\nNema podataka za dijagnostiku.')
        return diag_df

    diag_sorted  = diag_df.sort_values(['position', 'test_year']).reset_index(drop=True)
    display(HTML(
        "<div style='max-height:420px;overflow:auto;border:1px solid #ccc;"
        "border-radius:6px;padding:4px;'>"
        + diag_sorted.to_html(index=False, border=0)
        + "</div>"
    ))

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    axes = axes.flatten()
    for i, pos in enumerate(POSITIONS):
        ax = axes[i]
        d  = diag_df[diag_df['position'] == pos].sort_values('test_year')
        if d.empty:
            ax.set_title(f'{pos} (nema podataka)')
            continue
        ax.plot(d['test_year'], d['train_rmse'], marker='o', lw=2, label='Train RMSE')
        ax.plot(d['test_year'], d['val_rmse'],   marker='o', lw=2, label='Validation RMSE')
        ax.plot(d['test_year'], d['test_rmse'],  marker='o', lw=2, label='Test RMSE')
        ax.set_title(f'{pos} — RMSE kroz vreme', fontsize=12, fontweight='bold')
        ax.set_xlabel('Test godina')
        ax.set_ylabel('RMSE')
        ax.grid(alpha=0.3)
        ax.legend()

    plt.suptitle('Dijagnostika: Train vs Validation vs Test RMSE',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print('\nSazetak dijagnoze po poziciji:')
    print(diag_df.groupby(['position', 'diagnosis']).size().unstack(fill_value=0))

    return diag_df
