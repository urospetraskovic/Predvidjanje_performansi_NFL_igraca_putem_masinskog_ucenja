import pandas as pd


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
