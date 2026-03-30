"""
Tri nivoa optimizacije hiperparametara:
  NIVO 1 — Random Search
  NIVO 2 — Grid Search (oko Random best)
  NIVO 3 — Bayesian Optimization (Optuna)

Podrzava i originalne 3 modela (XGBoost, LightGBM, RandomForest) i
dopunske 5 modela (LinearRegression, Ridge, Lasso, ElasticNet, KNN).
"""

import copy
import numpy as np
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

POSITIONS = ['QB', 'RB', 'TE', 'WR']

RANDOM_SEARCH_GRIDS = {
    'XGBoost': {
        'n_estimators':    randint(50, 400),
        'max_depth':       randint(2, 12),
        'learning_rate':   uniform(0.001, 0.2),
        'subsample':       uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
    },
    'LightGBM': {
        'n_estimators':  randint(50, 400),
        'max_depth':     randint(2, 12),
        'learning_rate': uniform(0.001, 0.2),
        'num_leaves':    randint(10, 100),
        'subsample':     uniform(0.5, 0.5),
    },
    'RandomForest': {
        'n_estimators':    randint(50, 300),
        'max_depth':       randint(5, 30),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf':  randint(1, 5),
    },
}

EXTRA_RANDOM_DISTS = {
    'Ridge': {'alpha': loguniform(1e-3, 1e3)},
    'Lasso': {'alpha': loguniform(1e-4, 1e1)},
    'ElasticNet': {
        'alpha':    loguniform(1e-4, 1e1),
        'l1_ratio': uniform(0.05, 0.9),
    },
    'KNN': {
        'n_neighbors': randint(3, 51),
        'weights':     ['uniform', 'distance'],
        'p':           [1, 2],
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _eval_on_scale(y_true, y_pred, is_wr):
    if is_wr:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2


def _create_refined_grid(best_params, model_name):
    """Kreiraj Grid oko najboljeg Random Search regiona."""
    bp = best_params
    if model_name == 'XGBoost':
        return {
            'n_estimators': range(max(50, int(bp['n_estimators']) - 50),
                                  int(bp['n_estimators']) + 100, 25),
            'max_depth':    range(max(2, int(bp['max_depth']) - 2),
                                  int(bp['max_depth']) + 3),
            'learning_rate': np.arange(max(0.001, bp['learning_rate'] - 0.03),
                                        bp['learning_rate'] + 0.04, 0.01),
            'subsample': [0.7, 0.8, 0.9, 1.0],
        }
    if model_name == 'LightGBM':
        return {
            'n_estimators': range(max(50, int(bp['n_estimators']) - 50),
                                  int(bp['n_estimators']) + 100, 25),
            'max_depth':    range(max(2, int(bp['max_depth']) - 2),
                                  int(bp['max_depth']) + 3),
            'learning_rate': np.arange(max(0.001, bp['learning_rate'] - 0.03),
                                        bp['learning_rate'] + 0.04, 0.01),
            'num_leaves': range(max(10, int(bp['num_leaves']) - 10),
                                int(bp['num_leaves']) + 20, 5),
        }
    if model_name == 'RandomForest':
        return {
            'n_estimators': range(max(50, int(bp['n_estimators']) - 50),
                                  int(bp['n_estimators']) + 100, 50),
            'max_depth':    range(max(5, int(bp['max_depth']) - 5),
                                  int(bp['max_depth']) + 10, 2),
            'min_samples_split': range(max(2, int(bp['min_samples_split']) - 1),
                                       int(bp['min_samples_split']) + 3),
        }
    return {}


def _extra_refined_grid(model_name, best_params):
    """Grid za dopunske modele (Ridge/Lasso/ElasticNet/KNN)."""
    if model_name == 'Ridge':
        a = float(best_params.get('alpha', 1.0))
        return {'alpha': np.unique(np.clip([a * f for f in (0.25, 0.5, 1, 2, 4)], 1e-4, 1e4)).tolist()}
    if model_name == 'Lasso':
        a = float(best_params.get('alpha', 0.1))
        return {'alpha': np.unique(np.clip([a * f for f in (0.25, 0.5, 1, 2, 4)], 1e-5, 1e2)).tolist()}
    if model_name == 'ElasticNet':
        a = float(best_params.get('alpha', 0.1))
        l = float(best_params.get('l1_ratio', 0.5))
        return {
            'alpha':    np.unique(np.clip([a * f for f in (0.25, 0.5, 1, 2, 4)], 1e-5, 1e2)).tolist(),
            'l1_ratio': np.unique(np.clip([l + d for d in (-0.2, -0.1, 0, 0.1, 0.2)], 0.01, 0.99)).tolist(),
        }
    if model_name == 'KNN':
        k = int(best_params.get('n_neighbors', 15))
        k_vals = np.unique(np.clip([k + d for d in (-6, -3, 0, 3, 6)], 1, 60)).astype(int)
        return {
            'n_neighbors': k_vals.tolist(),
            'weights': list({best_params.get('weights', 'uniform'), 'uniform', 'distance'}),
            'p':       list({best_params.get('p', 2), 1, 2}),
        }
    return {}


# ── Bayesian objectives ───────────────────────────────────────────────────────

def _make_objective(model_name, model_configs):
    """Vraca Optuna objective funkciju za dati model."""

    def objective(trial, X, y):
        if model_name == 'XGBoost':
            params = {
                'n_estimators':    trial.suggest_int('n_estimators', 50, 300),
                'max_depth':       trial.suggest_int('max_depth', 2, 12),
                'learning_rate':   trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample':       trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
        elif model_name == 'LightGBM':
            params = {
                'n_estimators':  trial.suggest_int('n_estimators', 50, 300),
                'max_depth':     trial.suggest_int('max_depth', 2, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'num_leaves':    trial.suggest_int('num_leaves', 10, 100),
                'subsample':     trial.suggest_float('subsample', 0.5, 1.0),
            }
        elif model_name == 'RandomForest':
            params = {
                'n_estimators':    trial.suggest_int('n_estimators', 50, 300),
                'max_depth':       trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 5),
            }
        elif model_name == 'Ridge':
            params = {'alpha': trial.suggest_float('alpha', 1e-4, 1e4, log=True)}
        elif model_name == 'Lasso':
            params = {'alpha': trial.suggest_float('alpha', 1e-5, 1e2, log=True)}
        elif model_name == 'ElasticNet':
            params = {
                'alpha':    trial.suggest_float('alpha', 1e-5, 1e2, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.01, 0.99),
            }
        elif model_name == 'KNN':
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
                'weights':     trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p':           trial.suggest_categorical('p', [1, 2]),
            }
        else:
            return 0.0

        model = copy.deepcopy(model_configs[model_name]['model'])
        model.set_params(**params)
        scores = cross_val_score(
            model, X, y,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
        )
        return -scores.mean()

    return objective


# ── NIVO 1: Random Search ────────────────────────────────────────────────────

def run_random_search(processed, model_configs, exp_logger=None,
                      models=None, n_iter=20):
    """
    Pokrece Random Search za sve pozicije i modele.
    Vraca dict: {pos: {model_name: {best_params, cv_mae, test_mae, test_rmse, test_r2}}}
    """
    if models is None:
        models = ['XGBoost', 'LightGBM', 'RandomForest']

    results = {}
    tscv = TimeSeriesSplit(n_splits=3)

    for pos in POSITIONS:
        if pos not in processed:
            continue
        X_tr, X_te, y_tr, y_te = processed[pos]
        is_wr = (pos == 'WR')
        results[pos] = {}
        print(f"\n{pos} — Random Search")
        print("-" * 70)

        for model_name in models:
            if model_name not in model_configs:
                continue

            base = copy.deepcopy(model_configs[model_name]['model'])

            if model_name == 'LinearRegression':
                base.fit(X_tr, y_tr)
                cv_mae = -cross_val_score(
                    copy.deepcopy(model_configs[model_name]['model']),
                    X_tr, y_tr, cv=tscv,
                    scoring='neg_mean_absolute_error', n_jobs=-1,
                ).mean()
                best_params = {}
                best_model = base
            else:
                dists = RANDOM_SEARCH_GRIDS.get(model_name) or EXTRA_RANDOM_DISTS.get(model_name, {})
                rs = RandomizedSearchCV(
                    base, dists, n_iter=n_iter,
                    cv=tscv, scoring='neg_mean_absolute_error',
                    n_jobs=-1, random_state=42,
                )
                rs.fit(X_tr, y_tr)
                best_model  = rs.best_estimator_
                cv_mae      = -rs.best_score_
                best_params = rs.best_params_

            mae, rmse, r2 = _eval_on_scale(y_te, best_model.predict(X_te), is_wr)
            results[pos][model_name] = {
                'best_params': best_params,
                'cv_mae':      float(cv_mae),
                'test_mae':    float(mae),
                'test_rmse':   float(rmse),
                'test_r2':     float(r2),
            }

            if exp_logger is not None:
                exp_logger.log_experiment(pos, model_name, 'Random Search',
                                          best_params, cv_mae, mae, rmse, r2)

            print(f"  {model_name:<20} CV MAE={cv_mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    if exp_logger is not None:
        exp_logger.save()
    print("\nRandom Search zavrsen.")
    return results


# ── NIVO 2: Grid Search ───────────────────────────────────────────────────────

def run_grid_search(processed, model_configs, random_results, exp_logger=None,
                    models=None):
    """
    Grid Search oko Random Search best parametara.
    Vraca dict: {pos: {model_name: {...}}}
    """
    if models is None:
        models = list(random_results.get(POSITIONS[0], {}).keys())

    results = {}
    tscv = TimeSeriesSplit(n_splits=3)

    for pos in POSITIONS:
        if pos not in processed or pos not in random_results:
            continue
        X_tr, X_te, y_tr, y_te = processed[pos]
        is_wr = (pos == 'WR')
        results[pos] = {}
        print(f"\n{pos} — Grid Search")
        print("-" * 70)

        for model_name in models:
            if model_name not in model_configs:
                continue

            base         = copy.deepcopy(model_configs[model_name]['model'])
            best_random  = random_results.get(pos, {}).get(model_name, {}).get('best_params', {})

            if model_name == 'LinearRegression':
                base.fit(X_tr, y_tr)
                cv_mae = -cross_val_score(
                    copy.deepcopy(model_configs[model_name]['model']),
                    X_tr, y_tr, cv=tscv,
                    scoring='neg_mean_absolute_error', n_jobs=-1,
                ).mean()
                best_params = {}
                best_model  = base
            else:
                grid = (_create_refined_grid(best_random, model_name)
                        or _extra_refined_grid(model_name, best_random))
                gs = GridSearchCV(
                    base, grid, cv=tscv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1, refit=True,
                )
                gs.fit(X_tr, y_tr)
                best_model  = gs.best_estimator_
                cv_mae      = -gs.best_score_
                best_params = gs.best_params_

            mae, rmse, r2 = _eval_on_scale(y_te, best_model.predict(X_te), is_wr)
            prev_rmse = random_results.get(pos, {}).get(model_name, {}).get('test_rmse', rmse)
            improvement = (prev_rmse - rmse) / max(prev_rmse, 1e-9) * 100

            results[pos][model_name] = {
                'best_params': best_params,
                'cv_mae':      float(cv_mae),
                'test_mae':    float(mae),
                'test_rmse':   float(rmse),
                'test_r2':     float(r2),
            }

            if exp_logger is not None:
                exp_logger.log_experiment(pos, model_name, 'Grid Search',
                                          best_params, cv_mae, mae, rmse, r2)

            print(f"  {model_name:<20} CV MAE={cv_mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}  "
                  f"({improvement:+.1f}% vs Random)")

    if exp_logger is not None:
        exp_logger.save()
    print("\nGrid Search zavrsen.")
    return results


# ── NIVO 3: Bayesian Optimization ─────────────────────────────────────────────

def run_bayesian_optimization(processed, model_configs, exp_logger=None,
                              models=None, n_trials=15):
    """
    Bayesian Optimization (Optuna TPE) za sve pozicije i modele.
    Vraca dict: {pos: {model_name: {..., n_trials}}}
    """
    if models is None:
        models = ['XGBoost', 'LightGBM', 'RandomForest']

    results = {}

    for pos in POSITIONS:
        if pos not in processed:
            continue
        X_tr, X_te, y_tr, y_te = processed[pos]
        is_wr = (pos == 'WR')
        results[pos] = {}
        print(f"\n{pos} — Bayesian Optimization")
        print("-" * 70)

        for model_name in models:
            if model_name not in model_configs:
                continue

            if model_name == 'LinearRegression':
                base = copy.deepcopy(model_configs[model_name]['model'])
                base.fit(X_tr, y_tr)
                tscv = TimeSeriesSplit(n_splits=3)
                cv_mae = -cross_val_score(
                    copy.deepcopy(model_configs[model_name]['model']),
                    X_tr, y_tr, cv=tscv,
                    scoring='neg_mean_absolute_error', n_jobs=-1,
                ).mean()
                best_params = {}
                best_model  = base
                n_done      = 0
            else:
                objective_fn = _make_objective(model_name, model_configs)
                study = optuna.create_study(
                    direction='minimize',
                    sampler=optuna.samplers.TPESampler(seed=42),
                )
                study.optimize(lambda t: objective_fn(t, X_tr, y_tr),
                               n_trials=n_trials, show_progress_bar=False)

                best_params = study.best_trial.params
                cv_mae      = float(study.best_trial.value)
                n_done      = len(study.trials)

                best_model = copy.deepcopy(model_configs[model_name]['model'])
                best_model.set_params(**best_params)
                best_model.fit(X_tr, y_tr)

            mae, rmse, r2 = _eval_on_scale(y_te, best_model.predict(X_te), is_wr)
            results[pos][model_name] = {
                'best_params': best_params,
                'cv_mae':      float(cv_mae),
                'test_mae':    float(mae),
                'test_rmse':   float(rmse),
                'test_r2':     float(r2),
                'n_trials':    n_done,
            }

            if exp_logger is not None:
                exp_logger.log_experiment(pos, model_name, 'Bayesian Optimization',
                                          best_params, cv_mae, mae, rmse, r2)

            print(f"  {model_name:<20} CV MAE={cv_mae:.3f}  RMSE={rmse:.3f}  "
                  f"R²={r2:.3f}  (trials={n_done})")

    if exp_logger is not None:
        exp_logger.save()
    print("\nBayesian Optimization zavrsen.")
    return results


# ── Sumarni prikaz poređenja ──────────────────────────────────────────────────

def build_comparison_df(random_results, grid_results, bayesian_results):
    """Gradi DataFrame sa svim strategijama za poredjenje."""
    rows = []
    for pos in POSITIONS:
        for model_name in ['XGBoost', 'LightGBM', 'RandomForest',
                           'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'KNN']:
            for strategy, src in [('Random Search', random_results),
                                   ('Grid Search', grid_results),
                                   ('Bayesian Optimization', bayesian_results)]:
                if pos in src and model_name in src[pos]:
                    v = src[pos][model_name]
                    rows.append({
                        'position':  pos,
                        'model':     model_name,
                        'strategy':  strategy,
                        'test_rmse': v['test_rmse'],
                        'test_r2':   v['test_r2'],
                    })
    import pandas as pd
    df = pd.DataFrame(rows)
    if not df.empty:
        df['pos_model'] = df['position'] + '-' + df['model']
    return df


def print_best_per_position(comparison_df):
    """Stampa i vraca best model po poziciji (najmanji RMSE)."""
    import pandas as pd
    if comparison_df.empty:
        print("Nema podataka.")
        return pd.DataFrame()
    best = (
        comparison_df.loc[comparison_df.groupby('position')['test_rmse'].idxmin(),
                          ['position', 'model', 'strategy', 'test_rmse', 'test_r2']]
        .sort_values('position')
        .reset_index(drop=True)
    )
    print("\nNajbolji model po poziciji:")
    for _, row in best.iterrows():
        print(f"  {row['position']}: {row['model']} ({row['strategy']}) "
              f"| RMSE={row['test_rmse']:.3f}  R²={row['test_r2']:.3f}")
    return best
