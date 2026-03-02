import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fully combined')

#  QB defaults 
QB_CAREER_PLAYERS = ['Tom Brady', 'Matthew Stafford', 'Kirk Cousins']
QB_CAREER_COLORS  = {'Tom Brady': '#1565C0', 'Matthew Stafford': '#C62828', 'Kirk Cousins': '#2E7D32'}
_QB_CAREER_MARKERS = {'Tom Brady': 'o', 'Matthew Stafford': 's', 'Kirk Cousins': '^'}

# RB defaults 
RB_CAREER_PLAYERS = ['Adrian Peterson', 'Alvin Kamara', 'Saquon Barkley']
RB_CAREER_COLORS  = {'Adrian Peterson': '#6A1B9A', 'Alvin Kamara': '#AD1457', 'Saquon Barkley': '#004C54'}
_RB_CAREER_MARKERS = {'Adrian Peterson': 'D', 'Alvin Kamara': 'v', 'Saquon Barkley': 'P'}


def _career_axes(players, colors, markers, df, y_col, ylabel, fmt):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.patch.set_facecolor('white')
    y_max = df[y_col].max() * 1.28
    for idx, player in enumerate(players):
        ax = axes[idx]
        data = df[df['Player'] == player].sort_values('Season')
        ax.fill_between(data['Season'], data[y_col], alpha=0.15, color=colors[player])
        ax.plot(data['Season'], data[y_col],
                color=colors[player], marker=markers[player],
                linewidth=2.5, markersize=7, markeredgecolor='white', markeredgewidth=1.2,
                zorder=3)
        peak_idx = data[y_col].idxmax()
        peak = data.loc[peak_idx]
        ax.annotate(fmt(peak[y_col], int(peak['Season'])),
                    xy=(peak['Season'], peak[y_col]),
                    xytext=(0, 16), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=colors[player],
                    ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='->', color=colors[player], lw=1.2))
        ax.set_ylim(bottom=0, top=y_max)
        ax.set_title(player, fontsize=14, fontweight='bold', color=colors[player], pad=10)
        ax.set_xlabel('Sezona', fontsize=11)
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        player_seasons = sorted(data['Season'].unique())
        ax.set_xticks(player_seasons[::2])
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.35, color='grey')
        ax.grid(axis='x', visible=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return fig


#  QB functions 

def plot_qb_career_yards(players=None, colors=None, data_dir=None):
    if players is None:
        players = QB_CAREER_PLAYERS
    if colors is None:
        colors = QB_CAREER_COLORS
    if data_dir is None:
        data_dir = _DATA_DIR

    qb_df = pd.read_csv(os.path.join(data_dir, 'qb_master.csv'))
    qb_df = qb_df[qb_df['Player'].isin(players)]
    df = qb_df[['Player', 'Season', 'Yds']].copy()
    df = df.groupby(['Player', 'Season'], as_index=False)['Yds'].sum()
    df = df[df['Yds'] > 0]

    fig = _career_axes(players, colors, _QB_CAREER_MARKERS, df, 'Yds', 'Passing Yards',
                       lambda v, s: f"{int(v):,} yd\n({s})")
    for ax in fig.axes:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    fig.suptitle('Passing Yards kroz sezone – QB usporedba', fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.show()


def plot_qb_career_yards_per_game(players=None, colors=None, data_dir=None):
    if players is None:
        players = QB_CAREER_PLAYERS
    if colors is None:
        colors = QB_CAREER_COLORS
    if data_dir is None:
        data_dir = _DATA_DIR

    qb_df = pd.read_csv(os.path.join(data_dir, 'qb_master.csv'))
    qb_df = qb_df[qb_df['Player'].isin(players)]
    df = qb_df[['Player', 'Season', 'G', 'Yds']].copy()
    df = df.groupby(['Player', 'Season'], as_index=False).agg({'G': 'sum', 'Yds': 'sum'})
    df = df[df['Yds'] > 0]
    df['Y/G'] = df['Yds'] / df['G']

    fig = _career_axes(players, colors, _QB_CAREER_MARKERS, df, 'Y/G', 'Passing Yards per Game',
                       lambda v, s: f"{v:.1f} y/g\n({s})")
    for ax in fig.axes:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
    fig.suptitle('Passing Yards per Game kroz sezone – QB usporedba', fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.show()


#  RB functions 

def plot_rb_career_yards(players=None, colors=None, data_dir=None):
    if players is None:
        players = RB_CAREER_PLAYERS
    if colors is None:
        colors = RB_CAREER_COLORS
    if data_dir is None:
        data_dir = _DATA_DIR

    rb_df = pd.read_csv(os.path.join(data_dir, 'rb_master.csv'))
    rb_df = rb_df[rb_df['Player'].isin(players)]
    df = rb_df[['Player', 'Season', 'Rush_Yds']].copy()
    df = df.groupby(['Player', 'Season'], as_index=False)['Rush_Yds'].sum()
    df = df[df['Rush_Yds'] > 0]

    fig = _career_axes(players, colors, _RB_CAREER_MARKERS, df, 'Rush_Yds', 'Rushing Yards',
                       lambda v, s: f"{int(v):,} yd\n({s})")
    for ax in fig.axes:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    fig.suptitle('Rushing Yards kroz sezone – RB usporedba', fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.show()


def plot_rb_career_yards_per_game(players=None, colors=None, data_dir=None):
    if players is None:
        players = RB_CAREER_PLAYERS
    if colors is None:
        colors = RB_CAREER_COLORS
    if data_dir is None:
        data_dir = _DATA_DIR

    rb_df = pd.read_csv(os.path.join(data_dir, 'rb_master.csv'))
    rb_df = rb_df[rb_df['Player'].isin(players)]
    df = rb_df[['Player', 'Season', 'G', 'Rush_Yds']].copy()
    df = df.groupby(['Player', 'Season'], as_index=False).agg({'G': 'sum', 'Rush_Yds': 'sum'})
    df = df[df['Rush_Yds'] > 0]
    df['Rush_Y/G'] = df['Rush_Yds'] / df['G']

    fig = _career_axes(players, colors, _RB_CAREER_MARKERS, df, 'Rush_Y/G', 'Rushing Yards per Game',
                       lambda v, s: f"{v:.1f} y/g\n({s})")
    for ax in fig.axes:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
    fig.suptitle('Rushing Yards per Game kroz sezone – RB usporedba', fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.show()


#  QB team-change comparison 

QB_BRADY_COLORS = {'NWE': '#002244', 'TAM': '#D50A0A'}

def plot_qb_team_comparison(
    player='Tom Brady',
    team_a='NWE', team_b='TAM',
    season_from=2018, season_to=2021,
    colors=None,
    title='Tom Brady: New England Patriots → Tampa Bay Buccaneers',
    min_games=1,
    data_dir=None,
):
    if colors is None:
        colors = QB_BRADY_COLORS
    if data_dir is None:
        data_dir = _DATA_DIR

    qb_df = pd.read_csv(os.path.join(data_dir, 'qb_master.csv'))
    df = qb_df[qb_df['Player'] == player][['Season', 'Team', 'G', 'Yds', 'Y/G']].copy()
    df = df[df['G'] > min_games]

    part_a = df[(df['Team'] == team_a) & (df['Season'] >= season_from)].sort_values('Season')
    part_b = df[(df['Team'] == team_b) & (df['Season'] <= season_to)].sort_values('Season')
    combined = pd.concat([part_a, part_b])

    divider_x = len(part_a) - 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('white')

    for team, grp in combined.groupby('Team'):
        ax1.bar(grp['Season'].astype(str), grp['Yds'], color=colors[team], edgecolor='white', width=0.6)
        for _, row in grp.iterrows():
            ax1.text(str(int(row['Season'])), row['Yds'] + 60, f"{int(row['Yds']):,}",
                     ha='center', fontsize=9, fontweight='bold')

    ax1.set_title(f'{player} – Passing Yards po timu', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sezona', fontsize=11)
    ax1.set_ylabel('Passing Yards', fontsize=12)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax1.axvline(x=divider_x, color='grey', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for team, grp in combined.groupby('Team'):
        ax2.bar(grp['Season'].astype(str), grp['Y/G'], color=colors[team], edgecolor='white', width=0.6)
        for _, row in grp.iterrows():
            ax2.text(str(int(row['Season'])), row['Y/G'] + 3, f"{row['Y/G']:.1f}",
                     ha='center', fontsize=9, fontweight='bold')

    ax2.set_title(f'{player} – Yards per Game po timu', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sezona', fontsize=11)
    ax2.set_ylabel('Yards / Game', fontsize=12)
    ax2.axvline(x=divider_x, color='grey', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.show()


# RB team-change comparison 

RB_BARKLEY_COLORS = {'NYG': '#0B2265', 'PHI': '#004C54'}

def plot_rb_team_comparison(
    player='Saquon Barkley',
    team_a='NYG', team_b='PHI',
    season_from=2018, season_to=2024,
    colors=None,
    title='Saquon Barkley: New York Giants → Philadelphia Eagles',
    min_games=1,
    data_dir=None,
):
    if colors is None:
        colors = RB_BARKLEY_COLORS
    if data_dir is None:
        data_dir = _DATA_DIR

    rb_df = pd.read_csv(os.path.join(data_dir, 'rb_master.csv'))
    df = rb_df[rb_df['Player'] == player][['Season', 'Team', 'G', 'Rush_Yds', 'Rush_Y/G']].copy()
    df = df[df['G'] > min_games]

    part_a = df[(df['Team'] == team_a) & (df['Season'] >= season_from)].sort_values('Season')
    part_b = df[(df['Team'] == team_b) & (df['Season'] <= season_to)].sort_values('Season')
    combined = pd.concat([part_a, part_b])

    divider_x = len(part_a) - 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('white')

    for team, grp in combined.groupby('Team'):
        ax1.bar(grp['Season'].astype(str), grp['Rush_Yds'], color=colors[team], label=team, edgecolor='white', width=0.6)
        for _, row in grp.iterrows():
            ax1.text(str(int(row['Season'])), row['Rush_Yds'] + 25, f"{int(row['Rush_Yds']):,}",
                     ha='center', fontsize=9, fontweight='bold')

    ax1.set_title(f'{player} – Rushing Yards po timu', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sezona', fontsize=11)
    ax1.set_ylabel('Rushing Yards', fontsize=12)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax1.legend(title='Tim', fontsize=10)
    ax1.axvline(x=divider_x, color='grey', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for team, grp in combined.groupby('Team'):
        ax2.bar(grp['Season'].astype(str), grp['Rush_Y/G'], color=colors[team], label=team, edgecolor='white', width=0.6)
        for _, row in grp.iterrows():
            ax2.text(str(int(row['Season'])), row['Rush_Y/G'] + 1.5, f"{row['Rush_Y/G']:.1f}",
                     ha='center', fontsize=9, fontweight='bold')

    ax2.set_title(f'{player} – Rush Yards per Game po timu', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sezona', fontsize=11)
    ax2.set_ylabel('Rush Yards / Game', fontsize=12)
    ax2.legend(title='Tim', fontsize=10)
    ax2.axvline(x=divider_x, color='grey', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.show()