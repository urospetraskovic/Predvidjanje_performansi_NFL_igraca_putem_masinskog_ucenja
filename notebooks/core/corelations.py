import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_QB_HEATMAP_COLS = [
    "Yds_per_G", "Cmp", "Att", "Cmp%", "TD", "Int",
    "Y/A", "Rate", "Sk", "NY/A", "ANY/A", "G", "GS",
    "Age", "1D", "Lng", "AV",
]


def plot_qb_correlation_heatmap(
    qb: pd.DataFrame,
    cols: list[str] | None = None,
) -> None:
    if "Yds_per_G" not in qb.columns:
        qb["Yds_per_G"] = qb["Yds"] / qb["G"]
    if "TD_per_G" not in qb.columns:
        qb["TD_per_G"] = qb["TD"] / qb["G"]

    if cols is None:
        cols = DEFAULT_QB_HEATMAP_COLS

    cols = [c for c in cols if c in qb.columns]
    corr = qb[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, square=True,
        linewidths=0.5, ax=ax, vmin=-1, vmax=1,
        annot_kws={"size": 8},
    )
    ax.set_title("QB - Korelaciona matrica (Yds per Game)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()


DEFAULT_RB_HEATMAP_COLS = [
    "Rush_Yds_per_G", "Rush_Att", "Rush_TD", "Rush_1D", "Rush_Y/A",
    "Rush_A/G", "G", "GS", "Age", "Rec", "Rec_Yds",
    "Touches", "Scrimmage_Yds", "Fmb",
]


def plot_rb_correlation_heatmap(
    rb: pd.DataFrame,
    cols: list[str] | None = None,
) -> None:
    if "Rush_Yds_per_G" not in rb.columns:
        rb["Rush_Yds_per_G"] = rb["Rush_Yds"] / rb["G"]
    if "Rush_TD_per_G" not in rb.columns:
        rb["Rush_TD_per_G"] = rb["Rush_TD"] / rb["G"]

    if cols is None:
        cols = DEFAULT_RB_HEATMAP_COLS

    cols = [c for c in cols if c in rb.columns]
    corr = rb[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, square=True,
        linewidths=0.5, ax=ax, vmin=-1, vmax=1,
        annot_kws={"size": 9},
    )
    ax.set_title("RB - Korelaciona matrica (Yards per Game)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()


DEFAULT_WR_HEATMAP_COLS = [
    "receiving_yards_per_game", "targets", "receptions", "air_yards", "yac", "tds",
    "catch_rate", "games_played", "first_downs", "epa", "wpa",
    "avg_depth", "target_share", "yards_per_target", "success_rate",
]


def plot_wr_correlation_heatmap(
    wr_seasons: pd.DataFrame,
    cols: list[str] | None = None,
) -> None:
    if "receiving_yards_per_game" not in wr_seasons.columns:
        wr_seasons["receiving_yards_per_game"] = (
            wr_seasons["receiving_yards"] / wr_seasons["games_played"]
        )

    if cols is None:
        cols = DEFAULT_WR_HEATMAP_COLS

    cols = [c for c in cols if c in wr_seasons.columns]
    corr = wr_seasons[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, square=True,
        linewidths=0.5, ax=ax, vmin=-1, vmax=1,
        annot_kws={"size": 8},
    )
    ax.set_title("WR - Korelaciona matrica (Yards per Game)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()


DEFAULT_TE_HEATMAP_COLS = [
    "Rec_Y/G", "Tgt", "Rec", "Rec_TD", "Rec_1D", "Rec_Y/R",
    "Catch%", "G", "GS", "Age", "Touches", "Scrimmage_Yds", "AV",
]


def plot_te_correlation_heatmap(
    te: pd.DataFrame,
    cols: list[str] | None = None,
) -> None:
    if cols is None:
        cols = DEFAULT_TE_HEATMAP_COLS

    cols = [c for c in cols if c in te.columns]
    corr = te[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, square=True,
        linewidths=0.5, ax=ax, vmin=-1, vmax=1,
        annot_kws={"size": 9},
    )
    ax.set_title("TE - Korelaciona matrica (Yards per Game)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()


DEFAULT_QB_PAIR_COLS = ['Yds', 'TD', 'Cmp%', 'Y/A', 'Rate', 'Int']

def plot_qb_pairplot(qb, cols=None):
    if cols is None:
        cols = DEFAULT_QB_PAIR_COLS
    pair_cols = [c for c in cols if c in qb.columns]
    df = qb[pair_cols].dropna()
    g = sns.pairplot(df, diag_kind='kde',
                     plot_kws={'alpha': 0.4, 's': 15, 'color': '#3498db'},
                     diag_kws={'color': '#3498db', 'fill': True, 'alpha': 0.4})
    g.figure.suptitle('QB Pair Plot — Passing Metrike', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


DEFAULT_RB_PAIR_COLS = ['Rush_Yds', 'Rush_TD', 'Rush_Y/A', 'Rush_Att', 'Rec_Yds', 'Scrimmage_Yds']

def plot_rb_pairplot(rb, cols=None):
    if cols is None:
        cols = DEFAULT_RB_PAIR_COLS
    pair_cols = [c for c in cols if c in rb.columns]
    df = rb[pair_cols].dropna()
    g = sns.pairplot(df, diag_kind='kde',
                     plot_kws={'alpha': 0.4, 's': 15, 'color': '#e74c3c'},
                     diag_kws={'color': '#e74c3c', 'fill': True, 'alpha': 0.4})
    g.figure.suptitle('RB Pair Plot — Rushing & Scrimmage Metrike', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


DEFAULT_WR_PAIR_COLS = ['receiving_yards', 'receptions', 'targets', 'tds', 'yac', 'air_yards']

def plot_wr_pairplot(wr_seasons, cols=None):
    if cols is None:
        cols = DEFAULT_WR_PAIR_COLS
    wr = wr_seasons.copy()
    pair_cols = [c for c in cols if c in wr.columns]
    df = wr[pair_cols].dropna()
    g = sns.pairplot(df, diag_kind='kde',
                     plot_kws={'alpha': 0.3, 's': 10, 'color': '#2ecc71'},
                     diag_kws={'color': '#2ecc71', 'fill': True, 'alpha': 0.4})
    g.figure.suptitle('WR Pair Plot — Receiving Metrike', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


DEFAULT_TE_PAIR_COLS = ['Rec_Yds', 'Rec', 'Tgt', 'Rec_TD', 'Rec_Y/R', 'catch_pct']

def plot_te_pairplot(te, cols=None):
    if cols is None:
        cols = DEFAULT_TE_PAIR_COLS
    pair_cols = [c for c in cols if c in te.columns]
    df = te[pair_cols].dropna()
    g = sns.pairplot(df, diag_kind='kde',
                     plot_kws={'alpha': 0.4, 's': 15, 'color': '#9b59b6'},
                     diag_kws={'color': '#9b59b6', 'fill': True, 'alpha': 0.4})
    g.figure.suptitle('TE Pair Plot — Receiving Metrike', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
