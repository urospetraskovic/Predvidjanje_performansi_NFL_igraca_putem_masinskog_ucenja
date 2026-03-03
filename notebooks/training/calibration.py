import copy
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prepare_fold_data(df_full, scol, feat_cols, train_end, test_year):
    """
    Dijeli DataFrame na train i test skup za jedan fold.

    Returns
    -------
    tr_df, te_df : DataFrames (originalni, sa svim kolonama)
    X_tr, X_te   : feature DataFrames
    y_tr, y_te   : target Series
    Vraća None za sve ako test skup je prazan.
    """
    tr_df = df_full[df_full[scol] <= train_end]
    te_df = df_full[df_full[scol] == test_year]

    if len(te_df) == 0:
        return None, None, None, None, None, None

    X_tr = tr_df[feat_cols].copy()
    y_tr = tr_df['target'].copy()
    X_te = te_df[feat_cols].copy()
    y_te = te_df['target'].copy()

    return tr_df, te_df, X_tr, X_te, y_tr, y_te


def impute_lag2_and_median(X_tr, X_te):
    """
    Korak 1+2: lag2 stupci → 0, zatim MedianImputer fitovan SAMO na train.

    Returns
    -------
    X_tr_i, X_te_i : imputirani DataFrames (novi objekti, originali nepromijenjeni)
    """
    X_tr = X_tr.copy()
    X_te = X_te.copy()

    # lag2 NaN → 0
    lag2_c = [c for c in X_tr.columns if c.endswith('_lag2')]
    X_tr[lag2_c] = X_tr[lag2_c].fillna(0)
    X_te[lag2_c] = X_te[lag2_c].fillna(0)

    # Median imputer — fit SAMO na train
    imp = SimpleImputer(strategy='median')
    X_tr_i = pd.DataFrame(
        imp.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index
    )
    X_te_i = pd.DataFrame(
        imp.transform(X_te), columns=X_te.columns, index=X_te.index
    )

    return X_tr_i, X_te_i


def scale_fold(X_tr_i, X_te_i, binary_cols):
    """
    Korak 3: StandardScaler fitovan SAMO na numericke kolone train seta.
    Binarne kolone (0/1) se ne skaliraju.

    Returns
    -------
    X_tr_s, X_te_s : skalirani DataFrames (in-place modifikacija kopija)
    """
    X_tr_s = X_tr_i.copy()
    X_te_s = X_te_i.copy()

    bin_c = [c for c in binary_cols if c in X_tr_s.columns]
    num_c = [c for c in X_tr_s.columns if c not in bin_c]

    sc = StandardScaler()
    X_tr_s[num_c] = sc.fit_transform(X_tr_s[num_c])
    X_te_s[num_c] = sc.transform(X_te_s[num_c])

    return X_tr_s, X_te_s


def fit_fold_model(mcfg, X_tr, y_tr, n_train):
    """
    Korak 4: GridSearchCV ako model ima params, inače direktan fit.
    Broj CV foldova se automatski prilagođava veličini train seta.

    Returns
    -------
    best_model : fitovani estimator
    best_p     : dict s najboljim hiperparametrima (prazan za modele bez params)
    """
    tscv_fold = TimeSeriesSplit(n_splits=min(5, max(2, n_train // 50)))

    if mcfg['params']:
        grid = GridSearchCV(
            copy.deepcopy(mcfg['model']),
            mcfg['params'],
            cv=tscv_fold,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_tr, y_tr)
        return grid.best_estimator_, grid.best_params_
    else:
        best_model = copy.deepcopy(mcfg['model'])
        best_model.fit(X_tr, y_tr)
        return best_model, {}


def evaluate_fold(best_model, X_te, y_te, is_wr):
    """
    Korak 5+6: Predikcija + WR inverz transformacija + računanje metrika.

    Returns
    -------
    mae, rmse, r2 : float metrike na originalnoj skali
    preds         : numpy array predikcija (na originalnoj skali za WR)
    """
    preds = best_model.predict(X_te)

    if is_wr:
        y_ev = np.expm1(y_te.values)
        p_ev = np.expm1(preds)
    else:
        y_ev = y_te.values
        p_ev = preds

    mae  = mean_absolute_error(y_ev, p_ev)
    rmse = np.sqrt(mean_squared_error(y_ev, p_ev))
    r2   = r2_score(y_ev, p_ev)

    return mae, rmse, r2, p_ev


def print_position_results(pos, cal_records, model_configs, folds, model_avg_rmse, cal_best_params):
    """
    Ispisuje tabelarni pregled RMSE/R² po foldu za datu poziciju i vraća ime
    najboljeg modela (najmanji avg RMSE).

    Returns
    -------
    best_model_name : str
    """
    print(f'\n  {"Model":<20} │', end='')
    for f in folds:
        print(f'  {f["name"] + " (" + str(f["test_year"]) + ")":>18}', end='')
    print(f'  │ {"AvgMAE":>8} {"AvgRMSE":>9} {"AvgR²":>7}')
    print(f'  {"─"*130}')

    for model_name in model_configs.keys():
        sub = [r for r in cal_records if r['pos'] == pos and r['model'] == model_name]
        if not sub:
            continue
        row_str = f'  {model_name:<20} │'
        maes, rmses, r2s = [], [], []
        for f in folds:
            match = [r for r in sub if r['test_year'] == f['test_year']]
            if match:
                row_str += f'  {match[0]["rmse"]:>7.2f} (R²={match[0]["r2"]:>6.3f})'
                maes.append(match[0]['mae'])
                rmses.append(match[0]['rmse'])
                r2s.append(match[0]['r2'])
            else:
                row_str += f'  {"N/A":>18}'

        avg_m  = np.mean(maes)  if maes  else np.nan
        avg_rm = np.mean(rmses) if rmses else np.nan
        avg_r  = np.mean(r2s)   if r2s   else np.nan
        row_str += f'  │ {avg_m:>8.3f} {avg_rm:>9.3f} {avg_r:>7.3f}'
        print(row_str)

    best_model_name = min(model_avg_rmse, key=model_avg_rmse.get)
    print(f'\n  ★ Najbolji model ({pos}): {best_model_name}  |  Avg RMSE = {model_avg_rmse[best_model_name]:.3f}')

    print(f'\n  Hiperparametri {best_model_name} po foldu:')
    for f in folds:
        p = cal_best_params.get((pos, best_model_name, f['name']), {})
        print(f'    {f["name"]} (train ≤{f["train_end"]}): {p if p else "N/A (nema hiperparametara)"}')

    return best_model_name


def print_calibration_summary(cal_df, cal_best_per_pos, best_models, test_results):
    """
    Ispisuje sumarni pregled kalibracije vs single-year (2024) test rezultata.
    """
    print(f'\n\n{"═"*105}')
    print(f'  SUMARNI PREGLED — Kalibracija vs Single-Year (2024) Test')
    print(f'{"═"*105}')
    print(f'  {"Pos":<5} {"Kal. pobjednik":<22} {"AvgMAE":>8} {"AvgRMSE":>9} {"AvgR²":>7}'
          f'  │  {"Single-2024 pobjednik":<22} {"RMSE_2024":>10}  │  {"Isti?":>5}')
    print(f'  {"─"*105}')

    for pos in ['QB', 'RB', 'TE', 'WR']:
        sub = cal_df[cal_df['pos'] == pos].groupby('model').agg(
            avg_mae=('mae', 'mean'), avg_rmse=('rmse', 'mean'), avg_r2=('r2', 'mean')
        )
        cal_best = sub['avg_rmse'].idxmin()
        cal_row  = sub.loc[cal_best]
        s24_name = best_models[pos][0]
        s24_rmse = test_results[pos].loc[s24_name, 'RMSE']
        same = '✓ DA' if cal_best == s24_name else '✗ NE'
        print(f'  {pos:<5} {cal_best:<22} {cal_row["avg_mae"]:>8.3f} {cal_row["avg_rmse"]:>9.3f} '
              f'{cal_row["avg_r2"]:>7.3f}  │  {s24_name:<22} {s24_rmse:>10.3f}  │  {same:>5}')

    print(f'\nKalibracija evaluacije završena — 4 potpuno nezavisna treninga.')
