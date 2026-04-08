import os
import json
import pandas as pd
from datetime import datetime


class ExperimentLogger:
    """Sistemski log svih hyperparameter eksperimenata."""

    def __init__(self, log_path='../results/hyperparameter_experiments.csv'):
        self.log_path = log_path
        self.records = []
        if os.path.exists(log_path):
            self.df = pd.read_csv(log_path)
            print(f"Ucitan postojeci log: {len(self.df)} eksperimenata")
        else:
            self.df = pd.DataFrame()

    def log_experiment(self, position, model_name, strategy, hyperparams,
                       cv_mae, test_mae, test_rmse, test_r2,
                       fold_num=None, is_best=False):
        record = {
            'timestamp':      datetime.now().isoformat(),
            'position':       position,
            'model':          model_name,
            'strategy':       strategy,
            'fold_num':       fold_num,
            'hyperparameters': json.dumps(hyperparams),
            'cv_mae':         float(cv_mae),
            'test_mae':       float(test_mae) if test_mae is not None else None,
            'test_rmse':      float(test_rmse) if test_rmse is not None else None,
            'test_r2':        float(test_r2) if test_r2 is not None else None,
            'is_best':        is_best,
        }
        self.records.append(record)

    def save(self):
        if self.records:
            new_df = pd.DataFrame(self.records)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            self.df.to_csv(self.log_path, index=False)
            print(f"Log sacuvan: {len(self.df)} ukupno eksperimenata")
            self.records = []

    def summary_by_position_model(self):
        if self.df.empty:
            print("Nema eksperimenata za prikazivanje.")
            return None
        summary = self.df.groupby(['position', 'model']).agg(
            avg_test_rmse=('test_rmse', 'mean'),
            avg_test_r2=('test_r2', 'mean'),
            n_experiments=('model', 'count'),
        ).reset_index()
        print("\n" + "=" * 80)
        print("SUMARNI PREGLED EKSPERIMENATA")
        print("=" * 80)
        print(summary.to_string(index=False))
        return summary
