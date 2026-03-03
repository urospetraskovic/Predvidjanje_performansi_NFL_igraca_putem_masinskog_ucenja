import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib.colors import Normalize


QB_ERA_BINS   = [1984, 1990, 2000, 2010, 2020, 2026]
QB_ERA_LABELS = ['1985–90', '1991–00', '2001–10', '2011–20', '2021–25']

def plot_qb_yards_by_era(qb, era_bins=None, era_labels=None):
    if era_bins is None:
        era_bins = QB_ERA_BINS
    if era_labels is None:
        era_labels = QB_ERA_LABELS

    df = qb[(qb['GS'] > 0) & (qb['Season'] >= era_bins[0] + 1)].copy()
    df['Era'] = pd.cut(df['Season'], bins=era_bins, labels=era_labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Era', y='Yds', ax=ax, palette='Blues')
    ax.set_title('QB — Passing Yards po erama', fontsize=14, fontweight='bold')
    ax.set_ylabel('Passing Yards')
    ax.set_xlabel('Era')
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()


QB_PASSING_METRICS = [
    ('Yds',  'Prosječni Passing Yards',  '#3498db'),
    ('TD',   'Prosječni Passing TDs',    '#e74c3c'),
    ('Rate', 'Prosječni Passer Rating',  '#2ecc71'),
    ('Cmp%', 'Prosječni Completion %',   '#f39c12'),
]

def plot_qb_passing_trends(qb, metrics=None, min_season=1985):
    if metrics is None:
        metrics = QB_PASSING_METRICS

    starters = qb[(qb['GS'] > 0) & (qb['Season'] >= min_season)].copy()
    agg_cols = [m[0] for m in metrics]
    yearly = starters.groupby('Season')[agg_cols].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    for ax, (col, title, color) in zip(axes.flat, metrics):
        y_vals = yearly[col].dropna()
        x_vals = yearly.loc[y_vals.index, 'Season']
        ax.plot(x_vals, y_vals, color=color, lw=2, marker='o', ms=4)
        z = np.polyfit(x_vals, y_vals, 2)
        p = np.poly1d(z)
        ax.plot(x_vals, p(x_vals), '--', color='gray', lw=1.5, alpha=0.7, label='Kvadratni trend')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Sezona')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle('QB — Evolucija passing metrika kroz sezone (samo starteri)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


RB_ERA_BINS   = [1987, 2000, 2010, 2020, 2026]
RB_ERA_LABELS = ['1988–00', '2001–10', '2011–20', '2021–25']

def plot_rb_yards_by_era(rb, era_bins=None, era_labels=None):
    if era_bins is None:
        era_bins = RB_ERA_BINS
    if era_labels is None:
        era_labels = RB_ERA_LABELS

    df = rb.copy()
    df['Era'] = pd.cut(df['Season'], bins=era_bins, labels=era_labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Era', y='Rush_Yds', ax=ax, palette='Reds')
    ax.set_title('RB — Rushing Yards po erama', fontsize=14, fontweight='bold')
    ax.set_ylabel('Rushing Yards')
    ax.set_xlabel('Era')
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()


RB_RUSHING_METRICS = [
    ('Rush_Yds', 'Prosječni Rushing Yards',    '#e74c3c'),
    ('Rush_TD',  'Prosječni Rushing TDs',       '#c0392b'),
    ('Rush_Y/A', 'Prosječni Yards per Attempt', '#e67e22'),
    ('Rush_A/G', 'Prosječni Attempts per Game', '#d35400'),
]

def plot_rb_rushing_trends(rb, metrics=None):
    if metrics is None:
        metrics = RB_RUSHING_METRICS

    agg_cols = [m[0] for m in metrics]
    yearly = rb.groupby('Season')[agg_cols].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    for ax, (col, title, color) in zip(axes.flat, metrics):
        y_vals = yearly[col].dropna()
        x_vals = yearly.loc[y_vals.index, 'Season']
        ax.plot(x_vals, y_vals, color=color, lw=2, marker='o', ms=4)
        z = np.polyfit(x_vals, y_vals, 2)
        p = np.poly1d(z)
        ax.plot(x_vals, p(x_vals), '--', color='gray', lw=1.5, alpha=0.7, label='Kvadratni trend')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Sezona')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle('RB — Evolucija rushing metrika kroz sezone',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_top10_seasons(qb, rb, te, wr_seasons, data_dir=None):
    import os

    # Load WR regular-season data
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fully combined')
    wr_path = os.path.join(data_dir, 'wr_all_seasons_without_playoffs.csv')
    wr = pd.read_csv(wr_path)
    wr['games_played'] = wr['games_played'].fillna(0)
    wr_reg = wr[wr['games_played'] <= 21]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # QB
    ax = axes[0, 0]
    top = qb.nlargest(10, 'Yds')[['Player', 'Season', 'Yds']].reset_index(drop=True)
    top['label'] = top['Player'] + ' (' + top['Season'].astype(str) + ')'
    bars = ax.barh(range(len(top)), top['Yds'], color='#3498db')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['label'], fontsize=9)
    ax.set_title('Top 10 QB sezone - Passing Yards', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    for bar, val in zip(bars, top['Yds']):
        ax.text(val + 20, bar.get_y() + bar.get_height() / 2, f'{val:,.0f}', va='center', fontsize=9)

    # RB
    ax = axes[0, 1]
    top = rb.nlargest(10, 'Rush_Yds')[['Player', 'Season', 'Rush_Yds']].reset_index(drop=True)
    top['label'] = top['Player'] + ' (' + top['Season'].astype(str) + ')'
    bars = ax.barh(range(len(top)), top['Rush_Yds'], color='#e74c3c')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['label'], fontsize=9)
    ax.set_title('Top 10 RB sezone - Rushing Yards', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    for bar, val in zip(bars, top['Rush_Yds']):
        ax.text(val + 10, bar.get_y() + bar.get_height() / 2, f'{val:,.0f}', va='center', fontsize=9)

    # WR
    ax = axes[1, 0]
    top = wr_reg.nlargest(10, 'receiving_yards')[['receiver_player_name', 'season', 'receiving_yards']].reset_index(drop=True)
    top['label'] = top['receiver_player_name'] + ' (' + top['season'].astype(str) + ')'
    bars = ax.barh(range(len(top)), top['receiving_yards'], color='#2ecc71')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['label'], fontsize=9)
    ax.set_title('Top 10 WR sezone - Receiving Yards (Regular Season)', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    for bar, val in zip(bars, top['receiving_yards']):
        ax.text(val + 10, bar.get_y() + bar.get_height() / 2, f'{val:,.0f}', va='center', fontsize=9)

    # TE
    ax = axes[1, 1]
    top = te.nlargest(10, 'Rec_Yds')[['Player', 'Season', 'Rec_Yds']].reset_index(drop=True)
    top['label'] = top['Player'] + ' (' + top['Season'].astype(str) + ')'
    bars = ax.barh(range(len(top)), top['Rec_Yds'], color='#9b59b6')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['label'], fontsize=9)
    ax.set_title('Top 10 TE sezone - Receiving Yards', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    for bar, val in zip(bars, top['Rec_Yds']):
        ax.text(val + 5, bar.get_y() + bar.get_height() / 2, f'{val:,.0f}', va='center', fontsize=9)

    fig.suptitle('Top 10 pojedinačnih sezona', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def _col(df, *opts):
    low = {c.lower(): c for c in df.columns}
    return next((low[o.lower()] for o in opts if o.lower() in low), None)


def plot_qb_career_scatter(qb, xlim=(3.0, 6.6), ylim=(150, 290), vmin=0.25, vmax=0.80):
    from adjustText import adjust_text

    C_NAME = _col(qb, 'Player', 'Name', 'QB')
    C_ATT  = _col(qb, 'Att',  'PA',    'Attempts')
    C_TD   = _col(qb, 'TD',   'TDs',   'PassingTD')
    C_YDS  = _col(qb, 'Yds',  'Yards', 'PassingYds')
    C_G    = _col(qb, 'G',    'Games', 'games')
    C_REC  = _col(qb, 'QBrec','W-L',   'Record')

    def _win_pct(records):
        w = l = 0
        for v in records.astype(str):
            m = re.search(r'(\d+)\s*[-–]\s*(\d+)', v)
            if m:
                w += int(m.group(1)); l += int(m.group(2))
        return w / (w + l) if w + l > 0 else np.nan

    g = qb.groupby(C_NAME)
    career = pd.DataFrame({
        'TD_pct':       100 * g[C_TD].sum() / g[C_ATT].sum(),
        'Yds_per_Game': g[C_YDS].sum() / g[C_G].sum(),
        'Games':        g[C_G].sum(),
        'Win_pct':      g[C_REC].apply(_win_pct),
    }).reset_index().dropna(subset=['TD_pct', 'Yds_per_Game'])

    x = career['TD_pct']
    y = career['Yds_per_Game']
    s = 20 + 380 * (career['Games'] - career['Games'].min()) / (career['Games'].max() - career['Games'].min() + 1e-6)
    c = career['Win_pct'].fillna(0.5).clip(vmin, vmax)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set(xlim=xlim, ylim=ylim,
           xlabel='Career TD%', ylabel='Career Yards per Game',
           title='Career TD% vs Yds/Game   (bubble size = career games  |  color = career Win %)')

    sc = ax.scatter(x, y, s=s, c=c, cmap='RdYlGn', norm=Normalize(vmin, vmax),
                    edgecolor='k', linewidths=0.6, alpha=0.88, zorder=3)

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('Career Win %')
    cb.set_ticks([vmin, 0.4, 0.55, 0.7, vmax])
    cb.set_ticklabels(['25%', '40%', '55%', '70%', '80%'])

    stroke = [pe.withStroke(linewidth=2.5, foreground='white')]
    vis = career[(x.values >= xlim[0]) & (x.values <= xlim[1]) &
                 (y.values >= ylim[0]) & (y.values <= ylim[1])]

    texts = [ax.text(r.TD_pct, r.Yds_per_Game, getattr(r, C_NAME),
                     fontsize=7.5, ha='center', va='center', zorder=5,
                     path_effects=stroke)
             for r in vis.itertuples()]

    adjust_text(texts, ax=ax, expand_points=(1.5, 1.5), expand_text=(1.3, 1.3))
    plt.tight_layout()
    plt.show()