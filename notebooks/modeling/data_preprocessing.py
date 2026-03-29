import os
import numpy as np
import pandas as pd


def load_datasets(data_dir: str | None = None) -> dict[str, pd.DataFrame]:
    if data_dir is None:
        data_dir = os.path.join('..', 'data', 'fully combined')

    qb_raw = pd.read_csv(os.path.join(data_dir, 'qb_master.csv'))
    rb_raw = pd.read_csv(os.path.join(data_dir, 'rb_master.csv'))
    te_raw = pd.read_csv(os.path.join(data_dir, 'te_master.csv'))
    wr_raw = pd.read_csv(os.path.join(data_dir, 'wr_all_seasons_without_playoffs.csv'))

    datasets = {
        'qb_raw': qb_raw,
        'rb_raw': rb_raw,
        'te_raw': te_raw,
        'wr_raw': wr_raw,
    }

    for name, df in [('QB', qb_raw), ('RB', rb_raw), ('TE', te_raw), ('WR', wr_raw)]:
        scol = 'Season' if 'Season' in df.columns else 'season'
        print(
            f'{name}: {df.shape[0]} redova x {df.shape[1]} kolona | '
            f'Sezone: {df[scol].min()}-{df[scol].max()}'
        )

    return datasets

# ── Target  ───────────────────────────────────────────────────

def add_targets_qb_rb_te(qb_raw, rb_raw, te_raw):
    qb = qb_raw[qb_raw['GS'] > 0].copy()
    qb['target'] = qb['Yds'] / qb['G'].replace(0, np.nan)

    rb = rb_raw.copy()
    rb['target'] = rb['Rush_Yds'] / rb['G'].replace(0, np.nan)

    te = te_raw[te_raw['G'] > 0].copy()
    te['target'] = te['Rec_Yds'] / te['G']

    return qb, rb, te


def add_target_wr(wr_raw):
    wr = wr_raw.copy()
    wr['rec_yds_per_game'] = wr['receiving_yards'] / wr['games_played'].replace(0, np.nan)
    wr['target'] = np.log1p(wr['rec_yds_per_game'].clip(lower=0))
    return wr


def print_target_summary(qb, rb, te, wr):
    print('Ciljne promenljive:')
    for name, df in [('QB', qb), ('RB', rb), ('TE', te)]:
        t = df['target']
        scol = 'Season' if 'Season' in df.columns else 'season'
        print(f'{name} (Yds/G):       mean={t.mean():.2f}, median={t.median():.2f}, min={t.min():.2f}, max={t.max():.2f} | Sezone: {df[scol].min()}-{df[scol].max()}')

    t_raw = wr['rec_yds_per_game']
    t_log = wr['target']
    print(f'WR (Yds/G raw):   mean={t_raw.mean():.2f}, median={t_raw.median():.2f}, min={t_raw.min():.2f}, max={t_raw.max():.2f}')
    print(f'WR (log1p Yds/G): mean={t_log.mean():.3f}, median={t_log.median():.3f}, min={t_log.min():.3f}, max={t_log.max():.3f}')


# ── Multi-Output Targets ──────────────────────────────────────────────

# Konfiguracija ciljnih varijabli za multi-output regresiju po poziciji
MULTI_TARGET_CONFIG = {
    'QB': {
        'target_cols': ['target_yds_per_game', 'target_td_per_game', 'target_rate'],
        'target_labels': ['Yds/G', 'TD/G', 'Passer Rating'],
    },
    'RB': {
        'target_cols': ['target_yds_per_game', 'target_td_per_game', 'target_ypa'],
        'target_labels': ['Rush Yds/G', 'Rush TD/G', 'Rush Y/A'],
    },
    'TE': {
        'target_cols': ['target_yds_per_game', 'target_td_per_game', 'target_rec_per_game'],
        'target_labels': ['Rec Yds/G', 'Rec TD/G', 'Rec/G'],
    },
    'WR': {
        'target_cols': ['target_yds_per_game', 'target_td_per_game', 'target_rec_per_game'],
        'target_labels': ['log1p(Rec Yds/G)', 'TD/G', 'Rec/G'],
    },
}


def add_multi_targets_qb(qb_raw):
    """Kreira višestruke ciljne varijable za QB poziciju.

    Ciljne varijable:
        target_yds_per_game : Yds / G  (passing yards per game)
        target_td_per_game  : TD / G   (touchdowns per game)
        target_rate         : Rate     (passer rating)
    """
    qb = qb_raw[qb_raw['GS'] > 0].copy()
    qb['target_yds_per_game'] = qb['Yds'] / qb['G'].replace(0, np.nan)
    qb['target_td_per_game']  = qb['TD']  / qb['G'].replace(0, np.nan)
    qb['target_rate']         = qb['Rate']
    # Kompatibilnost — zadržavamo i originalni 'target'
    qb['target'] = qb['target_yds_per_game']
    return qb


def add_multi_targets_rb(rb_raw):
    """Kreira višestruke ciljne varijable za RB poziciju.

    Ciljne varijable:
        target_yds_per_game : Rush_Yds / G  (rushing yards per game)
        target_td_per_game  : Rush_TD / G   (rushing touchdowns per game)
        target_ypa          : Rush_Y/A      (yards per attempt)
    """
    rb = rb_raw.copy()
    rb['target_yds_per_game'] = rb['Rush_Yds'] / rb['G'].replace(0, np.nan)
    rb['target_td_per_game']  = rb['Rush_TD']  / rb['G'].replace(0, np.nan)
    rb['target_ypa']          = rb['Rush_Y/A']
    # Kompatibilnost
    rb['target'] = rb['target_yds_per_game']
    return rb


def add_multi_targets_te(te_raw):
    """Kreira višestruke ciljne varijable za TE poziciju.

    Ciljne varijable:
        target_yds_per_game : Rec_Yds / G  (receiving yards per game)
        target_td_per_game  : Rec_TD / G   (receiving touchdowns per game)
        target_rec_per_game : Rec / G      (receptions per game)
    """
    te = te_raw[te_raw['G'] > 0].copy()
    te['target_yds_per_game'] = te['Rec_Yds'] / te['G']
    te['target_td_per_game']  = te['Rec_TD']  / te['G']
    te['target_rec_per_game'] = te['Rec']     / te['G']
    # Kompatibilnost
    te['target'] = te['target_yds_per_game']
    return te


def add_multi_targets_wr(wr_raw):
    """Kreira višestruke ciljne varijable za WR poziciju.

    Ciljne varijable:
        target_yds_per_game : log1p(receiving_yards / games_played)
        target_td_per_game  : tds / games_played
        target_rec_per_game : receptions / games_played
    """
    wr = wr_raw.copy()
    wr['rec_yds_per_game']    = wr['receiving_yards'] / wr['games_played'].replace(0, np.nan)
    wr['target_yds_per_game'] = np.log1p(wr['rec_yds_per_game'].clip(lower=0))
    wr['target_td_per_game']  = wr['tds']        / wr['games_played'].replace(0, np.nan)
    wr['target_rec_per_game'] = wr['receptions']  / wr['games_played'].replace(0, np.nan)
    # Kompatibilnost
    wr['target'] = wr['target_yds_per_game']
    return wr


def print_multi_target_summary(datasets_by_pos: dict, config=None):
    """Prikazuje summary statistike za sve multi-output ciljne varijable.

    datasets_by_pos: dict {pos: DataFrame}
    config: MULTI_TARGET_CONFIG ili None za default
    """
    if config is None:
        config = MULTI_TARGET_CONFIG

    print('Multi-output ciljne promenljive:')
    for pos, df in datasets_by_pos.items():
        cfg = config[pos]
        print(f'\n  {pos}:')
        for tcol, label in zip(cfg['target_cols'], cfg['target_labels']):
            if tcol in df.columns:
                t = df[tcol].dropna()
                print(f'    {label:20s} ({tcol}): '
                      f'mean={t.mean():.3f}, median={t.median():.3f}, '
                      f'min={t.min():.3f}, max={t.max():.3f}')
            else:
                print(f'    {label:20s} ({tcol}): NOT FOUND')


# ── Awards ───────────────────────────────────────────────────


def _encode_awards(series):
    def has(val, keywords):
        if pd.isna(val) or str(val).strip() == '':
            return 0
        return int(any(kw in str(val) for kw in keywords))

    df_enc = pd.DataFrame(index=series.index)
    df_enc['award_PB']        = series.apply(lambda v: has(v, ['PB']))
    df_enc['award_AP1']       = series.apply(lambda v: has(v, ['AP-1']))
    df_enc['award_AP2']       = series.apply(lambda v: has(v, ['AP-2']))
    df_enc['award_MVP_top5']  = series.apply(lambda v: has(v, ['MVP-1','MVP-2','MVP-3','MVP-4','MVP-5']))
    df_enc['award_OPoY_top5'] = series.apply(lambda v: has(v, ['OPoY-1','OPoY-2','OPoY-3','OPoY-4','OPoY-5']))
    return df_enc


def apply_award_encoding(qb, rb, te):
    qb = pd.concat([qb, _encode_awards(qb['Awards'])], axis=1)
    rb = pd.concat([rb, _encode_awards(rb['awards'])], axis=1)
    te = pd.concat([te, _encode_awards(te['Awards'])], axis=1)
    return qb, rb, te


def print_award_summary(qb, rb, te):
    award_cols = ['award_PB', 'award_AP1', 'award_AP2', 'award_MVP_top5', 'award_OPoY_top5']
    print('Award encoding — broj sezona po kategoriji:')
    print(f'\n{"Nagrada":<20} {"QB":>6} {"RB":>6} {"TE":>6}')
    print('-' * 38)
    for col in award_cols:
        print(f'{col:<20} {qb[col].sum():>6} {rb[col].sum():>6} {te[col].sum():>6}')

# ── Drop columns ───────────────────────────────────────────────────


def drop_metadata_columns_qb_rb_te(qb, rb, te):
    QB_DROP = (
        ['PlayerID', 'Pos', 'QBrec', 'Lng', 'TD%', 'Int%', 'Sk%', 'Y/G']
        + [c for c in qb.columns if c.startswith('def_')]
        + [c for c in qb.columns if c.startswith('elo_')]
        + ['snp_ST%', 'snp_special_teams']
    )

    RB_DROP = (
        ['PlayerID', 'Lg', 'Pos', 'Rush_Y/G', 'Rec_Y/G']
        + [c for c in rb.columns if c.startswith('def_')]
        + ['snp_ST%', 'snp_ST_Snaps', 'snp_special_teams']
        + ['rec_success', 'catch_pct', 'Y/Tgt', 'Touch',
           'yds_per_touch', 'yds_from_scrimmage', 'rush_receive_td', 'Rec_R/G']
    )

    TE_DROP = (
        ['PlayerID', 'Pos']
        + ['snp_ST%', 'snp_ST_Snaps', 'snp_Def%', 'snp_Def_Snaps']
    )

    qb.drop(columns=[c for c in QB_DROP if c in qb.columns], inplace=True)
    rb.drop(columns=[c for c in RB_DROP if c in rb.columns], inplace=True)
    te.drop(columns=[c for c in TE_DROP if c in te.columns], inplace=True)

    return qb, rb, te


def drop_unused_columns_wr(wr):
    WR_DROP = (
        ['receiver_player_id', 'passer_player_id',
         'defteam', 'home_team', 'away_team', 'player_team']
        + [c for c in wr.columns if '_Q1' in c or '_Q2' in c or '_Q3' in c or '_Q4' in c]
        + [c for c in wr.columns if c.startswith('yards_wp_')
           or c.startswith('receptions_wp_') or c.startswith('targets_wp_')]
        + ['temp_f', 'humidity_pct', 'wind_mph',
           'is_rain', 'is_snow', 'is_clear', 'is_dome', 'surface']
        + ['avg_score_diff', 'avg_quarter', 'trailing_pct', 'leading_pct', 'wp_var']
        + ['pregame_spread', 'pregame_total']
    )

    wr.drop(columns=[c for c in WR_DROP if c in wr.columns], inplace=True)

    return wr


def drop_rr_snp_adv_columns(qb, rb, te, wr):
    qb_extra = (
        [c for c in qb.columns if c.startswith('rr_')]
        + ['snp_Def%', 'snp_Off%', 'snp_defense', 'snp_offense']
    )
    qb.drop(columns=[c for c in qb_extra if c in qb.columns], inplace=True)

    for df in [rb, te, wr]:
        drop = [c for c in df.columns if c.startswith('adv_') or c.startswith('snp_')]
        df.drop(columns=drop, inplace=True)

    return qb, rb, te, wr


def drop_awards_raw(qb, rb, te):
    for df, col in [(qb, 'Awards'), (rb, 'awards'), (te, 'Awards')]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return qb, rb, te


def drop_qb_qbr(qb):
    if 'QBR' in qb.columns:
        qb.drop(columns=['QBR'], inplace=True)
    return qb


def drop_rb_receiving_cols(rb):
    RB_DROP2 = ['Tgt', 'Rec', 'Rec_Yds', 'Rec_Y/R', 'Rec_TD', 'Rec_1D',
                'Rec_Succ%', 'Rec_Lng', 'Rec/G', 'Catch%', 'Rec_Y/Tgt', 'rec_long',
                'Y/Touch', 'Touches', 'Scrimmage_Yds', 'Rush_Rec_TD']
    rb.drop(columns=[c for c in RB_DROP2 if c in rb.columns], inplace=True)
    return rb


def drop_te_rushing_cols(te):
    TE_DROP2 = ['Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rush_1D', 'Rush_Y/G', 'Rush_A/G',
                'Touches', 'Y/Touch', 'Scrimmage_Yds', 'Rush_Succ%', 'Rush_Lng', 'Rush_Y/A']
    te.drop(columns=[c for c in TE_DROP2 if c in te.columns], inplace=True)
    return te


def drop_wr_def_dev_cols(wr):
    WR_DROP2 = ['def_targets_dev', 'def_receptions_dev', 'def_yards_dev',
                'def_tds_dev', 'def_epa_dev']
    wr.drop(columns=[c for c in WR_DROP2 if c in wr.columns], inplace=True)
    return wr


def print_shape_and_columns(qb, rb, te, wr):
    print(f'{"Pozicija":<6} {"Kolona":>7} {"Redova":>7}')
    print('-' * 25)
    for pos, df in [('QB', qb), ('RB', rb), ('TE', te), ('WR', wr)]:
        print(f'{pos:<6} {df.shape[1]:>7} {df.shape[0]:>7}')
    for pos, df in [('QB', qb), ('RB', rb), ('TE', te), ('WR', wr)]:
        print(f'\n{"="*60}')
        print(f'  {pos} — {df.shape[1]} kolona x {df.shape[0]} redova')
        print(f'{"="*60}')
        for i, col in enumerate(df.columns, 1):
            print(f'  {i:>3}. {col}')



# ── Imputacija i skaliranje ───────────────────────────────────────────────────

def impute_lag2_zeros(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lag2_cols = [c for c in X_train.columns if c.endswith('_lag2')]
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[lag2_cols] = X_train[lag2_cols].fillna(0)
    X_test[lag2_cols]  = X_test[lag2_cols].fillna(0)
    return X_train, X_test


def impute_median(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns, index=X_test.index,
    )
    return X_train_imp, X_test_imp


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    binary_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.preprocessing import StandardScaler
    binary_present = [c for c in binary_cols if c in X_train.columns]
    scale_cols     = [c for c in X_train.columns if c not in binary_present]
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols]  = scaler.transform(X_test[scale_cols])
    return X_train, X_test


def impute_and_scale(splits: dict, binary_cols: list[str]) -> dict:
    processed = {}
    for pos, (X_train, X_test, y_train, y_test) in splits.items():
        X_train, X_test = impute_lag2_zeros(X_train, X_test)
        X_train, X_test = impute_median(X_train, X_test)
        X_train, X_test = scale_features(X_train, X_test, binary_cols)
        processed[pos] = (X_train, X_test, y_train, y_test)
    return processed


def print_processed_summary(processed: dict, binary_cols: list[str]) -> None:
    """Prikazuje null-ove i broj skaliranih/binary kolona po poziciji."""
    print(f'{"Pos":<5} {"X_train nulls":>14} {"X_test nulls":>13} {"scaled cols":>12} {"binary cols":>12}')
    print('-' * 60)
    for pos, (X_tr, X_te, _, _) in processed.items():
        binary_present = [c for c in binary_cols if c in X_tr.columns]
        scale_cols     = [c for c in X_tr.columns if c not in binary_present]
        print(f'{pos:<5} {X_tr.isnull().sum().sum():>14} {X_te.isnull().sum().sum():>13} '
              f'{len(scale_cols):>12} {len(binary_present):>12}')


def print_processed_columns(processed: dict) -> None:
    """Pregled svih kolona, tipova i null vrijednosti u processed train/test skupovima."""
    for pos, (X_tr, X_te, y_tr, y_te) in processed.items():
        print(f'\n{"="*65}')
        print(f'  {pos} — X_train: {X_tr.shape[0]} redova x {X_tr.shape[1]} kolona')
        print(f'{"="*65}')

        null_counts = X_tr.isnull().sum()
        print(f'\n  {"#":>4}  {"Kolona":<45} {"Tip":>8}  {"Nulls":>6}')
        print(f'  {"-"*70}')
        for i, col in enumerate(X_tr.columns, 1):
            null_n = null_counts[col]
            marker = ' ◄ NULL!' if null_n > 0 else ''
            print(f'  {i:>4}. {col:<45} {str(X_tr[col].dtype):>8}  {null_n:>6}{marker}')

        total_nulls = null_counts.sum()
        print(f'\n  Ukupno null-ova u X_train: {total_nulls}')
        print(f'  Ukupno null-ova u X_test:  {X_te.isnull().sum().sum()}')
        print(f'  y_train nulls: {y_tr.isnull().sum()} | y_test nulls: {y_te.isnull().sum()}')
