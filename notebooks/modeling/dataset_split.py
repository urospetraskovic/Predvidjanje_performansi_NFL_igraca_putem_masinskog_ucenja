import pandas as pd

def build_train_test_splits(
    lagged: dict[str, pd.DataFrame],
    pos_split: dict,
    test_season: int = 2024,
) -> dict:
    """Temporalni train/test split za svaku poziciju.

    train: season < test_season
    test:  season == test_season

    Vraća dict: pos -> (X_train, X_test, y_train, y_test)
    """
    splits = {}
    for pos, cfg in pos_split.items():
        df       = lagged[pos]
        id_cols  = cfg['id_cols']
        scol     = cfg['season_col']

        train_df  = df[df[scol] < test_season].copy()
        test_df   = df[df[scol] == test_season].copy()
        feat_cols = [c for c in df.columns if c not in id_cols + ['target']]

        splits[pos] = (
            train_df[feat_cols],
            test_df[feat_cols],
            train_df['target'],
            test_df['target'],
        )
    return splits


def print_split_summary(splits: dict) -> None:
    """Prikazuje veličine train/test skupova i broj feature-a po poziciji."""
    print(f'{"Pos":<5} {"X_train":>10} {"X_test":>8} {"y_train":>9} {"y_test":>8} {"features":>10}')
    print('-' * 55)
    for pos, (X_tr, X_te, y_tr, y_te) in splits.items():
        print(f'{pos:<5} {X_tr.shape[0]:>10} {X_te.shape[0]:>8} {y_tr.shape[0]:>9} {y_te.shape[0]:>8} {X_tr.shape[1]:>10}')
