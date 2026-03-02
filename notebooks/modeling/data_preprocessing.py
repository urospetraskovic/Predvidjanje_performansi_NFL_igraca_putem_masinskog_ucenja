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


def prepare_qb_rb_te(qb_raw, rb_raw, te_raw):
    qb = qb_raw[qb_raw['GS'] > 0].copy()
    qb['target'] = qb['Yds'] / qb['G'].replace(0, np.nan)

    rb = rb_raw.copy()
    rb['target'] = rb['Rush_Yds'] / rb['G'].replace(0, np.nan)

    te = te_raw[te_raw['G'] > 0].copy()
    te['target'] = te['Rec_Yds'] / te['G']

    return qb, rb, te


def prepare_wr(wr_raw):
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


def drop_extra_columns(qb, rb, te, wr):
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


def analyze_nulls_by_position(qb, rb, te, wr):
    for pos, df in [('QB', qb), ('RB', rb), ('TE', te), ('WR', wr)]:
        null_counts = df.isnull().sum()
        has_nulls = null_counts[null_counts > 0]
        print(f'\n{"="*55}')
        print(f'  {pos} — kolone sa null vrednostima ({len(has_nulls)} od {df.shape[1]})')
        print(f'{"="*55}')
        if len(has_nulls) == 0:
            print('  Nema null vrednosti.')
        else:
            print(f'  {"Kolona":<35} {"Null#":>6}  {"Null%":>6}')
            print(f'  {"-"*52}')
            for col, cnt in has_nulls.sort_values(ascending=False).items():
                pct = cnt / len(df) * 100
                print(f'  {col:<35} {cnt:>6}  {pct:>5.1f}%')
    return None

def analyze_qb_adv_columns(qb):
    adv_cols = [c for c in qb.columns if c.startswith('adv_')]
    print(f'Ukupno adv_ kolona: {len(adv_cols)}\n')

    null_pct = qb[adv_cols].isnull().mean() * 100
    print('Null% po adv_ koloni:')
    print(null_pct.sort_values(ascending=False).to_string())

    basic_stats = ['Yds', 'Cmp', 'Att', 'Cmp%', 'TD', 'Int', 'Rate', 'Sk', 'AV']
    print('\n\nKorelacija adv_ kolona sa osnovnim statistikama (|r| > 0.85 = visoka duplikacija):')
    print(f'{"adv kolona":<45} {"max |r|":>8}  {"najvise korelirana osnovna"}')
    print('-' * 85)
    for col in adv_cols:
        valid = qb[[col] + basic_stats].dropna()
        if len(valid) < 30:
            print(f'{col:<45} {"(premalo podataka)":>8}')
            continue
        corrs = valid[basic_stats].corrwith(valid[col]).abs()
        max_r = corrs.max()
        max_col = corrs.idxmax()
        marker = ' ◄ DUPLIKAT?' if max_r > 0.85 else ''
        print(f'{col:<45} {max_r:>8.3f}  {max_col}{marker}')
