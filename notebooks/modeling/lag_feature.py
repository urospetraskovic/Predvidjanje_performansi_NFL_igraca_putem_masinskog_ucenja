import pandas as pd


# ── Lag feature funkcije ──────────────────────────────────────────────────────

def build_lag_features(
    df: pd.DataFrame,
    player_col: str,
    season_col: str,
    id_cols: list[str],
    static_feats: list[str],
) -> pd.DataFrame:
    """Kreira lag-1 i lag-2 feature matricu za jednu poziciju.

    Lag1 NaN-ovi ostaju (imputiraće se medijanom).
    Lag2 NaN-ovi se pune nulom (nema historije 2 sezone unazad).
    Redovi bez targeta (G=0 edge cases) se uklanjaju.
    """
    static = [c for c in static_feats if c in df.columns]
    lag_cols = [c for c in df.columns if c not in id_cols + static + ['target']]

    base = df[id_cols + static + ['target']].copy()
    lag_src = df[[player_col, season_col] + lag_cols].copy()

    # Lag 1 — pomjeramo sezonu za +1 da matchamo sa tekućom sezonom t
    lag1 = lag_src.rename(columns={c: f'{c}_lag1' for c in lag_cols}).copy()
    lag1[season_col] = lag1[season_col] + 1
    base = base.merge(lag1, on=[player_col, season_col], how='left')

    # Lag 2 — pomjeramo sezonu za +2
    lag2 = lag_src.rename(columns={c: f'{c}_lag2' for c in lag_cols}).copy()
    lag2[season_col] = lag2[season_col] + 2
    base = base.merge(lag2, on=[player_col, season_col], how='left')

    lag2_cols = [f'{c}_lag2' for c in lag_cols]
    base[lag2_cols] = base[lag2_cols].fillna(0)

    base = base.dropna(subset=['target']).reset_index(drop=True)
    return base


def build_all_lag_matrices(pos_config: dict) -> dict[str, pd.DataFrame]:
    """Iterira kroz POS_CONFIG i gradi lag matricu za svaku poziciju."""
    lagged = {}
    for pos, cfg in pos_config.items():
        lagged[pos] = build_lag_features(
            df=cfg['df'].copy(),
            player_col=cfg['player_col'],
            season_col=cfg['season_col'],
            id_cols=cfg['id_cols'],
            static_feats=cfg['static_feats'],
        )
    return lagged


def print_lag_summary(lagged: dict) -> None:
    """Prikazuje tabelu s brojem redova, kolona i lag1 NaN redova po poziciji."""
    print(f'{"Pos":<5} {"Redova":>8} {"Kolona":>8} {"lag1 feats":>12} {"lag2 feats":>12} {"lag1 NaN redovi":>16}')
    print('-' * 65)
    for pos, df in lagged.items():
        lag1_cols = [c for c in df.columns if c.endswith('_lag1')]
        lag2_cols = [c for c in df.columns if c.endswith('_lag2')]
        lag1_nan_rows = df[lag1_cols].isnull().any(axis=1).sum()
        print(f'{pos:<5} {df.shape[0]:>8} {df.shape[1]:>8} {len(lag1_cols):>12} {len(lag2_cols):>12} {lag1_nan_rows:>16}')
