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
