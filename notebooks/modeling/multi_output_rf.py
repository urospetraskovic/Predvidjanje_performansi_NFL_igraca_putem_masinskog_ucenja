"""
Multi-output Random Forest regresija — prošireni targeti po poziciji.

Outputi po poziciji:
  QB  -> yards_per_game, attempts_per_game, completions_per_game, first_downs_per_game
  RB  -> yards_per_game, carries_per_game, yards_per_attempt, ybc_per_game
  TE  -> yards_per_game, receptions_per_game, targets_per_game, first_downs_per_game
  WR  -> yards_per_game, receptions_per_game, targets_per_game, air_yards_per_game, yac_per_game

Kolone iz raw/master podataka:
  QB: Cmp (completions), 1D (first_downs), Att (attempts), G (games)
  RB: Rush_Y/A (yards_per_attempt), adv_Rush_YBC (ybc total), Rush_Att, G
  TE: Tgt (targets), Rec_1D (first_downs), Rec, G
  WR: targets, air_yards, yac, receptions, games_played
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

YARDS_WEIGHT = 0.7
OTHER_WEIGHT = 0.3   # ravnomerno rasporedjeno na preostale targete
STAT2_WEIGHT = OTHER_WEIGHT  # backward-compat alias

# Konfiguracija targeta po poziciji.
# Svaki target je recnik:
#   'col'       — kolona u raw_df koja se koristi (None = direktno per-game ako je vec normalizovano)
#   'label'     — ime izlaza
#   'is_total'  — True: vrednost je sezonski zbir → deliti s games_col
#                 False: vrednost je vec per-game (ili ratio)
#   'is_yards'  — True: ovo je glavni yards target (prima WR log1p tretman)
MULTI_TARGET_CONFIG = {
    'QB': {
        'player_col': 'Player',
        'season_col': 'Season',
        'games_col':  'G',
        'targets': [
            {'col': None,   'label': 'yards_per_game',        'is_total': False, 'is_yards': True},
            {'col': '1D',   'label': 'first_downs_per_game',  'is_total': True,  'is_yards': False},
        ],
    },
    'RB': {
        'player_col': 'Player',
        'season_col': 'Season',
        'games_col':  'G',
        'targets': [
            {'col': None,            'label': 'yards_per_game',     'is_total': False, 'is_yards': True},
            {'col': 'Rush_Att',      'label': 'carries_per_game',   'is_total': True,  'is_yards': False},
            {'col': 'Rush_Y/A',      'label': 'yards_per_attempt',  'is_total': False, 'is_yards': False},
        ],
    },
    'TE': {
        'player_col': 'Player',
        'season_col': 'Season',
        'games_col':  'G',
        'targets': [
            {'col': None,      'label': 'yards_per_game',        'is_total': False, 'is_yards': True},
            {'col': 'Rec',     'label': 'receptions_per_game',   'is_total': True,  'is_yards': False},
            {'col': 'Tgt',     'label': 'targets_per_game',      'is_total': True,  'is_yards': False},
            {'col': 'Rec_1D',  'label': 'first_downs_per_game',  'is_total': True,  'is_yards': False},
        ],
    },
    'WR': {
        'player_col': 'receiver_player_name',
        'season_col': 'season',
        'games_col':  'games_played',
        'targets': [
            {'col': None,         'label': 'yards_per_game',        'is_total': False, 'is_yards': True},
            {'col': 'receptions', 'label': 'receptions_per_game',   'is_total': True,  'is_yards': False},
            {'col': 'targets',    'label': 'targets_per_game',      'is_total': True,  'is_yards': False},
            {'col': 'air_yards',  'label': 'air_yards_per_game',    'is_total': True,  'is_yards': False},
            {'col': 'yac',        'label': 'yac_per_game',          'is_total': True,  'is_yards': False},
        ],
    },
}

# Backward-compat alias (stari kod koji importuje MULTI_STAT2_CONFIG)
MULTI_STAT2_CONFIG = {
    pos: {
        'stat2_label': cfg['targets'][1]['label'],
        'player_col':  cfg['player_col'],
        'season_col':  cfg['season_col'],
        'games_col':   cfg['games_col'],
        'stat2_col':   cfg['targets'][1]['col'],
    }
    for pos, cfg in MULTI_TARGET_CONFIG.items()
}


# ── Priprema podataka ─────────────────────────────────────────────────────────

def _extract_one_target(raw_df, col, is_total, games_col, player_col, season_col):
    """
    Izvlaci jedan target iz raw_df.
    Ako is_total=True deli ukupan zbir s games_col da dobije per-game vrednost.
    Vraca DataFrame [player_col, season_col, 'value'].
    """
    needed = [player_col, season_col, games_col] + ([col] if col else [])
    df = raw_df[[c for c in needed if c in raw_df.columns]].copy()

    if col and col not in df.columns:
        return None

    df = df.dropna(subset=[games_col] + ([col] if col else []))
    df = df.sort_values(games_col, ascending=False)
    df = df.drop_duplicates(subset=[player_col, season_col], keep='first')

    if col is None:
        # yards_per_game — dolazi iz y_yards vec izracunatog u processed
        return None
    elif is_total:
        value = (df[col] / df[games_col].replace(0, np.nan)).clip(lower=0)
    else:
        value = df[col].clip(lower=0)

    result = df[[player_col, season_col]].copy()
    result['value'] = value.values
    return result


def build_multi_output_splits(lagged, processed, raw_dfs,
                               multi_target_config=None, test_season=2024):
    """
    Gradi multi-output Y matrice za sve pozicije.
    Y sadrzi yards_per_game + sve dodatne targete definisane u multi_target_config.
    X matrice ostaju nepromenjene iz `processed`.
    """
    if multi_target_config is None:
        multi_target_config = MULTI_TARGET_CONFIG

    splits = {}
    for pos, (X_tr, X_te, y_yards_tr, y_yards_te) in processed.items():
        cfg        = multi_target_config[pos]
        player_col = cfg['player_col']
        season_col = cfg['season_col']
        games_col  = cfg['games_col']
        raw_df     = raw_dfs[pos]

        enriched = lagged[pos].copy()
        train_mask = enriched[season_col] < test_season
        test_mask  = enriched[season_col] == test_season

        y_cols_train = {'yards_per_game': y_yards_tr.values}
        y_cols_test  = {'yards_per_game': y_yards_te.values}

        for tgt in cfg['targets']:
            if tgt['is_yards']:
                continue  # yards je vec u y_yards

            col = tgt['col']
            label = tgt['label']

            tgt_df = _extract_one_target(raw_df, col, tgt['is_total'],
                                          games_col, player_col, season_col)
            if tgt_df is None:
                # fallback: all zeros
                y_cols_train[label] = np.zeros(len(y_yards_tr))
                y_cols_test[label]  = np.zeros(len(y_yards_te))
                print(f'  [{pos}] Upozorenje: kolona "{col}" nije nadjena — koristi 0')
                continue

            merged = enriched.merge(tgt_df.rename(columns={'value': label}),
                                    on=[player_col, season_col], how='left')

            train_vals = merged.loc[train_mask, label].values
            test_vals  = merged.loc[test_mask,  label].values

            if len(train_vals) != len(y_yards_tr):
                print(f'  [{pos}/{label}] Duzina train ({len(train_vals)}) != yards ({len(y_yards_tr)}) — skip')
                y_cols_train[label] = np.zeros(len(y_yards_tr))
                y_cols_test[label]  = np.zeros(len(y_yards_te))
                continue

            median = np.nanmedian(train_vals)
            y_cols_train[label] = np.where(np.isnan(train_vals), median, train_vals)
            y_cols_test[label]  = np.where(np.isnan(test_vals),  median, test_vals)

        Y_train = pd.DataFrame(y_cols_train, index=X_tr.index)
        Y_test  = pd.DataFrame(y_cols_test,  index=X_te.index)
        splits[pos] = (X_tr, X_te, Y_train, Y_test)

        print(f'[{pos}] Y_train shape: {Y_train.shape}  targeti: {list(Y_train.columns)}')

    return splits


# ── Trening ───────────────────────────────────────────────────────────────────

def tune_multi_output_rf(X_train, Y_train, n_splits=3):
    """GridSearchCV za native multi-output RF. Vraca (best_model, best_params, cv_mae)."""
    param_grid = {
        'n_estimators':      [100, 200],
        'max_depth':         [None, 20],
        'min_samples_split': [2, 5],
    }
    rf   = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid = GridSearchCV(
        rf, param_grid,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring='neg_mean_absolute_error',
        n_jobs=-1, refit=True,
    )
    grid.fit(X_train, Y_train)
    return grid.best_estimator_, grid.best_params_, -grid.best_score_


def train_multi_output_models(multi_splits, n_splits=3, multi_target_config=None):
    """Trenira RF za svaku poziciju. Vraca (best_models, best_params)."""
    if multi_target_config is None:
        multi_target_config = MULTI_TARGET_CONFIG

    best_models, best_params = {}, {}
    for pos, (X_tr, X_te, Y_tr, Y_te) in multi_splits.items():
        labels = list(Y_tr.columns)
        print(f'\n[{pos}] GridSearchCV — {labels}')
        print(f'  X_train: {X_tr.shape}  Y_train: {Y_tr.shape}')
        model, params, cv_mae = tune_multi_output_rf(X_tr, Y_tr, n_splits)
        best_models[pos] = model
        best_params[pos] = params
        print(f'  Parametri: {params}  |  CV MAE: {cv_mae:.4f}')
    return best_models, best_params


# ── Evaluacija ────────────────────────────────────────────────────────────────

def compute_target_metrics(y_true, y_pred):
    """MAE, RMSE, R2 za jedan target."""
    return {
        'MAE':  mean_absolute_error(y_true, y_pred),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2':   r2_score(y_true, y_pred),
    }


def evaluate_multi_output(best_models, multi_splits,
                           yards_weight=YARDS_WEIGHT, other_weight=OTHER_WEIGHT,
                           multi_target_config=None):
    """
    Evaluira modele po svakom targetu i agregirano (weighted score).
    yards dobija yards_weight; svi ostali targeti dele other_weight ravnomerno.
    Vraca (metrics_df, eval_detail).
    """
    if multi_target_config is None:
        multi_target_config = MULTI_TARGET_CONFIG

    rows, detail = [], {}
    for pos, model in best_models.items():
        _, X_te, _, Y_te = multi_splits[pos]
        is_wr = (pos == 'WR')
        cfg   = multi_target_config[pos]

        preds  = model.predict(X_te)   # shape: (n_samples, n_targets)
        labels = list(Y_te.columns)
        n_others = len(labels) - 1
        per_other_w = other_weight / max(n_others, 1)

        detail[pos] = {}
        ws = 0.0
        pos_rows = []

        for i, label in enumerate(labels):
            y_true = Y_te[label].values
            y_pred = preds[:, i]

            # WR yards — inverse log1p
            if is_wr and label == 'yards_per_game':
                y_true = np.expm1(y_true)
                y_pred = np.expm1(y_pred)

            m = compute_target_metrics(y_true, y_pred)
            w = yards_weight if label == 'yards_per_game' else per_other_w
            ws += w * m['RMSE']

            detail[pos][label] = {**m, 'true': y_true, 'pred': y_pred, 'label': label, 'weight': w}
            pos_rows.append({'label': label, **m})

        detail[pos]['weighted_score'] = ws
        detail[pos]['labels'] = labels

        for r in pos_rows:
            rows.append({
                'position':       pos,
                'model':          'RandomForest',
                'target':         r['label'],
                'MAE':            round(r['MAE'],  4),
                'RMSE':           round(r['RMSE'], 4),
                'R2':             round(r['R2'],   4),
                'weighted_score': round(ws,        4),
            })

    return pd.DataFrame(rows), detail


# ── Cuvanje ───────────────────────────────────────────────────────────────────

def save_multi_output_results(metrics_df, eval_detail, best_mo_models, best_mo_params,
                               results_dir='../results', models_dir='../models',
                               multi_target_config=None):
    """Cuva CSV rezultate i joblib modele."""
    if multi_target_config is None:
        multi_target_config = MULTI_TARGET_CONFIG

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'multi_output_results.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f'Rezultati sacuvani: {csv_path}')

    os.makedirs(models_dir, exist_ok=True)
    for pos, model in best_mo_models.items():
        path = os.path.join(models_dir, f'{pos}_multi_output_rf.joblib')
        joblib.dump(model, path)
        ws     = eval_detail[pos]['weighted_score']
        labels = eval_detail[pos]['labels']
        print(f'  {pos}: {path}  |  targeti={labels}  |  ws={ws:.4f}  |  params={best_mo_params[pos]}')


# ── Vizuelizacija ─────────────────────────────────────────────────────────────

def plot_pred_vs_actual_multi(eval_detail, positions=None,
                               save_path='../results/multi_output_pred_vs_actual.png'):
    """
    n_targets x n_pos matrica grafika: Predicted vs Actual za svaki target.
    """
    if positions is None:
        positions = list(eval_detail.keys())

    n_pos     = len(positions)
    n_targets = max(len(eval_detail[p]['labels']) for p in positions)

    fig, axes = plt.subplots(n_targets, n_pos, figsize=(5 * n_pos, 4 * n_targets))
    if n_pos == 1 and n_targets == 1:
        axes = np.array([[axes]])
    elif n_pos == 1:
        axes = axes.reshape(-1, 1)
    elif n_targets == 1:
        axes = axes.reshape(1, -1)

    for col, pos in enumerate(positions):
        labels = eval_detail[pos]['labels']
        for row, label in enumerate(labels):
            ax = axes[row][col]
            d  = eval_detail[pos][label]
            lo = min(float(d['true'].min()), float(d['pred'].min())) * 0.95
            hi = max(float(d['true'].max()), float(d['pred'].max())) * 1.05
            ax.scatter(d['true'], d['pred'], alpha=0.45, s=20,
                       color='steelblue', edgecolors='none')
            ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5)
            ax.set_title(
                f'{pos} — {label}\nMAE={d["MAE"]:.3f}  RMSE={d["RMSE"]:.3f}  R²={d["R2"]:.3f}',
                fontsize=8,
            )
            ax.set_xlabel('Stvarne', fontsize=8)
            ax.set_ylabel('Predvidjene', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Sakri prazne subplotove ako pozicija ima manje targeta
        for row in range(len(labels), n_targets):
            axes[row][col].set_visible(False)

    plt.suptitle('Multi-output RF: Predicted vs Actual (test 2024)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f'Grafik sacuvan: {save_path}')
    plt.show()


def print_multi_output_summary(eval_detail, multi_target_config=None):
    """Stampa zaključak multi-output evaluacije po svim targetima."""
    if multi_target_config is None:
        multi_target_config = MULTI_TARGET_CONFIG

    n_others = sum(1 for pos in eval_detail for lbl in eval_detail[pos]['labels'] if lbl != 'yards_per_game')
    print('=' * 75)
    print('ZAKLJUCAK: Random Forest multi-output regresija')

    for pos, d in eval_detail.items():
        labels = d['labels']
        print(f'\n{pos}:')
        for label in labels:
            m = d[label]

            print(f'  {label:<25} MAE={m["MAE"]:.3f}  RMSE={m["RMSE"]:.3f}  R2={m["R2"]:.3f} ')
        
