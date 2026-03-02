import os
import joblib


def print_cv_test_summary(cv_results, test_results, best_models, model_configs):
    print('GridSearchCV CV MAE vs Test rezultati\n')
    print(f'{"Pos":<5} {"Model":<20} {"CV_MAE":>8} │ {"Test_MAE":>9} {"Test_RMSE":>10} {"Test_R²":>8}')
    print('─' * 70)

    for pos in ['QB', 'RB', 'TE', 'WR']:
        cv_df   = cv_results[pos]
        test_df = test_results[pos]
        best    = best_models[pos][0]

        for i, model_name in enumerate(model_configs.keys()):
            if model_name not in test_df.index:
                continue
            marker   = ' ★' if model_name == best else ''
            prefix   = pos if i == 0 else ''
            cv_mae   = cv_df.loc[model_name, 'CV_MAE']
            test_row = test_df.loc[model_name]
            print(f'{prefix:<5} {model_name:<20} '
                  f'{cv_mae:>8.3f} │ '
                  f'{test_row["MAE"]:>9.3f} {test_row["RMSE"]:>10.3f} {test_row["R2"]:>8.3f}{marker}')
        print('─' * 70)


def save_best_models(best_models, test_results, cv_results, models_dir=None):
    if models_dir is None:
        models_dir = os.path.join('..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    print('\nSačuvani modeli:')
    for pos, (name, model) in best_models.items():
        path = os.path.join(models_dir, f'{pos}_best_model.joblib')
        joblib.dump(model, path)
        test_rmse   = test_results[pos].loc[name, 'RMSE']
        best_params = cv_results[pos].loc[name, 'Best_params']
        print(f'  {pos}: {name} → RMSE={test_rmse:.3f} | params={best_params}')
