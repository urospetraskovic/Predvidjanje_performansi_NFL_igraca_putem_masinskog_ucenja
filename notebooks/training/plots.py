import numpy as np
import matplotlib.pyplot as plt

COLORS = {'QB': '#3498db', 'RB': '#e74c3c', 'TE': '#9b59b6', 'WR': '#2ecc71'}

PLAYER_COL = {
    'QB': 'Player', 'RB': 'Player', 'TE': 'Player', 'WR': 'receiver_player_name'
}
SEASON_COL = {
    'QB': 'Season', 'RB': 'Season', 'TE': 'Season', 'WR': 'season'
}


def get_player_names(lagged, test_year=2024):
    player_names = {}
    for pos in ['QB', 'RB', 'TE', 'WR']:
        pcol = PLAYER_COL[pos]
        scol = SEASON_COL[pos]
        test_df = lagged[pos][lagged[pos][scol] == test_year]
        player_names[pos] = test_df[pcol].values
    return player_names


def plot_predicted_vs_actual(processed, best_models, test_results, lagged,
                              test_year=2024, annotate_positions=('QB', 'RB', 'TE')):

    player_names = get_player_names(lagged, test_year)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for ax, pos in zip(axes.flat, ['QB', 'RB', 'TE', 'WR']):
        _, X_te, _, y_te = processed[pos]
        best_name, best_model = best_models[pos]
        preds = best_model.predict(X_te)
        names = player_names[pos]

        if pos == 'WR':
            y_actual   = np.expm1(y_te.values)
            y_pred     = np.expm1(preds)
            scale_note = ' [expm1]'
        else:
            y_actual   = y_te.values
            y_pred     = preds
            scale_note = ''

        mae  = test_results[pos].loc[best_name, 'MAE']
        rmse = test_results[pos].loc[best_name, 'RMSE']
        r2   = test_results[pos].loc[best_name, 'R2']

        ax.scatter(y_actual, y_pred, color=COLORS[pos], alpha=0.55, s=35,
                   edgecolor='white', linewidth=0.4)

        lo = min(y_actual.min(), y_pred.min()) * 0.95
        hi = max(y_actual.max(), y_pred.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.45, linewidth=1.2, label='Idealno')

        if pos in annotate_positions:
            for x, y, name in zip(y_actual, y_pred, names):
                parts = str(name).split()
                short = f'{parts[0][0]}. {" ".join(parts[1:])}' if len(parts) > 1 else name
                ax.annotate(
                    short,
                    xy=(x, y),
                    xytext=(3, 3),
                    textcoords='offset points',
                    fontsize=6.5,
                    alpha=0.75,
                    color='#222222',
                )

        ax.set_xlabel('Stvarne vrijednosti (yds/g)', fontsize=11)
        ax.set_ylabel('Predviđene vrijednosti (yds/g)', fontsize=11)
        ax.set_title(
            f'{pos} — {best_name}{scale_note}\n'
            f'MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.3f}',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f'Predicted vs Actual — Najbolji model po poziciji (Test sezona {test_year})\n'
        'WR: yds/g skala (expm1 inverz log1p)',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.show()
