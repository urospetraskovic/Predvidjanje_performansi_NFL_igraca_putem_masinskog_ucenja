import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

_FOLDS_DEFAULT = [
    {'name': 'Fold 1', 'train_end': 2020, 'test_year': 2021},
    {'name': 'Fold 2', 'train_end': 2021, 'test_year': 2022},
    {'name': 'Fold 3', 'train_end': 2022, 'test_year': 2023},
    {'name': 'Fold 4', 'train_end': 2023, 'test_year': 2024},
]

_YEAR_COLORS  = {2021: '#e67e22', 2022: '#3498db', 2023: '#9b59b6', 2024: '#2ecc71'}
_YEAR_MARKERS = {2021: 'o', 2022: 's', 2023: '^', 2024: 'D'}

_PLAYER_COL    = {'QB': 'Player', 'RB': 'Player', 'TE': 'Player', 'WR': 'receiver_player_name'}
_SEASON_COL    = {'QB': 'Season', 'RB': 'Season', 'TE': 'Season', 'WR': 'season'}


def collect_walkforward_predictions(
    best_models,
    lagged,
    pos_id_cols,
    binary_cols,
    cal_model_configs,
    cal_best_params,
    positions=('QB', 'RB', 'TE', 'WR'),
    folds=None,
):
    """
    Za svaki fold i poziciju trenira model s cal_best_params (svježi imputer+scaler)
    i prikuplja out-of-sample predikcije.
    WR: y_actual i y_pred se konvertuju expm1() → yds/g skala.

    Vraća all_preds: {pos: [{'year', 'player', 'actual', 'pred'}, ...]}
    """
    if folds is None:
        folds = _FOLDS_DEFAULT

    all_preds = {pos: [] for pos in positions}

    for pos in positions:
        best_name, _ = best_models[pos]
        scol         = _SEASON_COL[pos]
        id_cols      = pos_id_cols[pos]
        pcol         = _PLAYER_COL[pos]
        df_full      = lagged[pos]
        feat_cols    = [c for c in df_full.columns if c not in id_cols + ['target']]
        is_wr        = (pos == 'WR')

        print(f'{pos} — treniranje po foldu ({best_name}) ...')

        for fold in folds:
            tr_df = df_full[df_full[scol] <= fold['train_end']]
            te_df = df_full[df_full[scol] == fold['test_year']]
            if len(te_df) == 0:
                continue

            X_tr = tr_df[feat_cols].copy()
            y_tr = tr_df['target'].copy()
            X_te = te_df[feat_cols].copy()
            y_te = te_df['target'].copy()

            lag2_c = [c for c in X_tr.columns if c.endswith('_lag2')]
            X_tr[lag2_c] = X_tr[lag2_c].fillna(0)
            X_te[lag2_c] = X_te[lag2_c].fillna(0)

            imp    = SimpleImputer(strategy='median')
            X_tr_i = pd.DataFrame(imp.fit_transform(X_tr), columns=X_tr.columns)
            X_te_i = pd.DataFrame(imp.transform(X_te),     columns=X_te.columns)

            bin_c = [c for c in binary_cols if c in X_tr.columns]
            num_c = [c for c in X_tr.columns if c not in bin_c]
            sc    = StandardScaler()
            X_tr_i[num_c] = sc.fit_transform(X_tr_i[num_c])
            X_te_i[num_c] = sc.transform(X_te_i[num_c])

            best_p     = cal_best_params.get((pos, best_name, fold['name']), {})
            base_model = copy.deepcopy(cal_model_configs[best_name]['model'])
            if best_p:
                base_model.set_params(**best_p)
            base_model.fit(X_tr_i, y_tr)
            preds = base_model.predict(X_te_i)

            if is_wr:
                y_actual = np.expm1(y_te.values)
                y_pred   = np.expm1(preds)
            else:
                y_actual = y_te.values
                y_pred   = preds

            for name, ya, yp in zip(te_df[pcol].values, y_actual, y_pred):
                all_preds[pos].append({'year': fold['test_year'], 'player': name,
                                       'actual': ya, 'pred': yp})

        print(f'  → {len(all_preds[pos])} predikcija prikupljeno ({len(folds)} folda)')

    return all_preds


def plot_walkforward_scatter(
    all_preds,
    best_models,
    positions=('QB', 'RB', 'TE', 'WR'),
    annotate_positions=('QB', 'RB', 'TE'),
):
    """
    Crta 2×2 scatter Predicted vs Actual za sve pozicije i godine.
    Anotacije igrača se prikazuju samo za pozicije u `annotate_positions`.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    for ax, pos in zip(axes.flat, positions):
        df_p         = pd.DataFrame(all_preds[pos])
        best_name, _ = best_models[pos]
        is_wr        = (pos == 'WR')
        annotate     = pos in annotate_positions

        all_actual = df_p['actual'].values
        all_pred   = df_p['pred'].values
        lo = min(all_actual.min(), all_pred.min()) * 0.92
        hi = max(all_actual.max(), all_pred.max()) * 1.05

        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.35, linewidth=1.2, label='Idealno')

        for yr, grp in df_p.groupby('year'):
            ax.scatter(
                grp['actual'], grp['pred'],
                color=_YEAR_COLORS[yr], marker=_YEAR_MARKERS[yr],
                s=40, alpha=0.65, edgecolor='white', linewidth=0.4,
                label=str(yr), zorder=3,
            )
            if annotate:
                for _, row in grp.iterrows():
                    parts = str(row['player']).split()
                    short = f'{parts[0][0]}. {" ".join(parts[1:])}' if len(parts) > 1 else row['player']
                    ax.annotate(short, xy=(row['actual'], row['pred']),
                                xytext=(3, 3), textcoords='offset points',
                                fontsize=5.8, alpha=0.70, color='#2c2c2c')

        overall_r2  = r2_score(all_actual, all_pred)
        overall_mae = mean_absolute_error(all_actual, all_pred)

        yr_stats = df_p.groupby('year').apply(
            lambda g: pd.Series({
                'r2':  r2_score(g['actual'], g['pred']),
                'mae': mean_absolute_error(g['actual'], g['pred']),
            })
        )

        handles, labels = ax.get_legend_handles_labels()
        new_labels = []
        for lbl in labels:
            if lbl.isdigit() and int(lbl) in yr_stats.index:
                r = yr_stats.loc[int(lbl), 'r2']
                m = yr_stats.loc[int(lbl), 'mae']
                new_labels.append(f'{lbl}  R²={r:.3f}  MAE={m:.1f}')
            else:
                new_labels.append(lbl)

        ax.legend(handles, new_labels, fontsize=8, loc='upper left')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('Stvarne vrijednosti (yds/g)', fontsize=10)
        ax.set_ylabel('Predviđene vrijednosti (yds/g)', fontsize=10)
        ax.set_title(
            f'{pos} — {best_name}\n'
            f'{"WR: yds/g skala (expm1)  " if is_wr else ""}'
            f'Ukupno R²={overall_r2:.3f}  MAE={overall_mae:.2f}\n'
            f'N={len(df_p)} predikcija  |  test godine 2021–2024 (walk-forward)',
            fontsize=10, fontweight='bold'
        )
        ax.grid(True, alpha=0.18)

    fig.suptitle(
        'Predicted vs Actual — SVE GODINE (2021–2024)  |  Walk-Forward, 0 data leakage\n'
        'Svaka tačka = 100% out-of-sample predikcija  |  Model treniran SAMO na podacima prije test godine',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.show()


def print_walkforward_metrics(all_preds, positions=('QB', 'RB', 'TE', 'WR')):
    """Ispisuje tabelarni pregled MAE, RMSE i R² po poziciji i test godini."""
    print('\n' + '═' * 68)
    print('  Pregled metrika po poziciji i test godini (walk-forward)')
    print('═' * 68)
    print(f'  {"Pos":<5} {"Godina":>7} {"N":>5} {"MAE":>9} {"RMSE":>9} {"R²":>8}')
    print(f'  {"─" * 58}')
    for pos in positions:
        df_p = pd.DataFrame(all_preds[pos])
        for yr, grp in df_p.groupby('year'):
            mae  = mean_absolute_error(grp['actual'], grp['pred'])
            rmse = np.sqrt(np.mean((grp['actual'] - grp['pred']) ** 2))
            r2   = r2_score(grp['actual'], grp['pred'])
            print(f'  {pos:<5} {yr:>7} {len(grp):>5} {mae:>9.3f} {rmse:>9.3f} {r2:>8.3f}')
        all_a   = df_p['actual'].values
        all_p_v = df_p['pred'].values
        print(f'  {pos:<5} {"AVG":>7} {len(df_p):>5} '
              f'{mean_absolute_error(all_a, all_p_v):>9.3f} '
              f'{np.sqrt(np.mean((all_a - all_p_v) ** 2)):>9.3f} '
              f'{r2_score(all_a, all_p_v):>8.3f}')
        print(f'  {"─" * 58}')


def plot_walkforward_predicted_vs_actual(
    best_models,
    lagged,
    pos_id_cols,
    binary_cols,
    cal_model_configs,
    cal_best_params,
    positions=('QB', 'RB', 'TE', 'WR'),
    folds=None,
    annotate_positions=('QB', 'RB', 'TE'),
):
    all_preds = collect_walkforward_predictions(
        best_models, lagged, pos_id_cols, binary_cols,
        cal_model_configs, cal_best_params, positions, folds,
    )
    plot_walkforward_scatter(all_preds, best_models, positions, annotate_positions)
    print_walkforward_metrics(all_preds, positions)
    return all_preds
