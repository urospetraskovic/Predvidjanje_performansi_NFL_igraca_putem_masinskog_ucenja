import numpy as np
import pandas as pd


def plot_rmse_by_fold(ax, cal_df, cal_best_per_pos, positions, colors_pos):
    """Subplot 1: RMSE po foldu — linijski grafikon za najbolji model po poziciji."""
    for pos in positions:
        best_m = cal_best_per_pos[pos]
        sub = cal_df[(cal_df['pos'] == pos) & (cal_df['model'] == best_m)].sort_values('test_year')
        ax.plot(sub['test_year'], sub['rmse'], marker='o', linewidth=2.2, markersize=8,
                label=f'{pos} ({best_m})', color=colors_pos[pos])
    ax.set_xlabel('Test godina', fontsize=11)
    ax.set_ylabel('RMSE (yds/g)', fontsize=11)
    ax.set_title('RMSE po foldu — Najbolji model po poziciji\n(svježi GridSearchCV + trening po foldu)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks([2021, 2022, 2023, 2024])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)


def plot_r2_by_fold(ax, cal_df, cal_best_per_pos, positions, colors_pos):
    """Subplot 2: R² po foldu — linijski grafikon za najbolji model po poziciji."""
    for pos in positions:
        best_m = cal_best_per_pos[pos]
        sub = cal_df[(cal_df['pos'] == pos) & (cal_df['model'] == best_m)].sort_values('test_year')
        ax.plot(sub['test_year'], sub['r2'], marker='o', linewidth=2.2, markersize=8,
                label=f'{pos} ({best_m})', color=colors_pos[pos])
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Test godina', fontsize=11)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('R² po foldu — Najbolji model po poziciji\n(svježi GridSearchCV + trening po foldu)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks([2021, 2022, 2023, 2024])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)


def plot_train_size(ax, cal_df, cal_best_per_pos, positions, colors_pos):
    """Subplot 3: Veličina trening seta po foldu i poziciji (expanding window)."""
    offsets = {'QB': -0.3, 'RB': -0.1, 'TE': 0.1, 'WR': 0.3}
    for pos in positions:
        best_m = cal_best_per_pos[pos]
        sub = cal_df[(cal_df['pos'] == pos) & (cal_df['model'] == best_m)].sort_values('test_year')
        ax.bar([y + offsets[pos] for y in sub['test_year']],
               sub['n_train'], width=0.18, label=f'{pos}', color=colors_pos[pos], alpha=0.85)
    ax.set_xlabel('Test godina', fontsize=11)
    ax.set_ylabel('Broj trening primjera', fontsize=11)
    ax.set_title('Veličina trening seta po foldu i poziciji\n(expanding window)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks([2021, 2022, 2023, 2024])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')


def plot_avg_rmse_bar(ax, cal_df, cal_best_per_pos, positions, model_configs, colors_pos):
    """Subplot 4: Avg RMSE bar chart — svi modeli, grupirano po poziciji. ★ = pobjednik."""
    cal_summary = cal_df.groupby(['pos', 'model']).agg(avg_rmse=('rmse', 'mean')).reset_index()
    model_names = list(model_configs.keys())
    x     = np.arange(len(positions))
    width = 0.09

    for i, m_name in enumerate(model_names):
        vals = []
        for pos in positions:
            row = cal_summary[(cal_summary['pos'] == pos) & (cal_summary['model'] == m_name)]
            vals.append(row['avg_rmse'].values[0] if len(row) else 0)
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=m_name, alpha=0.85)
        for j, pos in enumerate(positions):
            if m_name == cal_best_per_pos[pos]:
                ax.text(x[j] + offset, vals[j] + 0.3, '★', ha='center', fontsize=11, color='gold')

    ax.set_xlabel('Pozicija', fontsize=11)
    ax.set_ylabel('Avg RMSE (4 folda)', fontsize=11)
    ax.set_title('Avg RMSE po modelu i poziciji\n(★ = najbolji, GridSearchCV svježi po foldu)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(positions, fontsize=11)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(True, alpha=0.25, axis='y')


def print_calibration_details(cal_df, cal_best_per_pos, cal_best_params, positions):
    """Ispisuje detaljnu tabelu metrika po poziciji i foldu, sa hiperparametrima."""
    print('\n' + '═' * 95)
    print('DETALJNI REZULTATI PO POZICIJI I FOLDU (sa parametrima)')
    print('═' * 95)

    for pos in positions:
        sub    = cal_df[cal_df['pos'] == pos]
        best_m = cal_best_per_pos[pos]

        rmse_piv = sub.pivot_table(index='model', columns='test_year', values='rmse').round(2)
        r2_piv   = sub.pivot_table(index='model', columns='test_year', values='r2').round(3)
        ntr_piv  = sub.pivot_table(index='model', columns='test_year', values='n_train', aggfunc='first')
        nte_piv  = sub.pivot_table(index='model', columns='test_year', values='n_test',  aggfunc='first')

        rmse_piv.columns = [f'RMSE_{y}' for y in rmse_piv.columns]
        r2_piv.columns   = [f'R²_{y}'   for y in r2_piv.columns]

        avg      = sub.groupby('model').agg(AvgRMSE=('rmse', 'mean'), AvgR2=('r2', 'mean')).round(3)
        combined = pd.concat([rmse_piv, r2_piv, avg], axis=1)

        print(f'\n{pos}  (★ Najbolji avg RMSE: {best_m})')
        print('Train/test veličine: ', end='')
        for yr in [2021, 2022, 2023, 2024]:
            if yr in ntr_piv.columns:
                n  = int(ntr_piv[yr].iloc[0])
                nt = int(nte_piv[yr].iloc[0])
                print(f'  {yr}: train={n}, test={nt}', end='')
        print()
        print(combined.to_string())

        print(f'\n  Parametri {best_m} po foldu:')
        for _, row in sub[sub['model'] == best_m].sort_values('test_year').iterrows():
            print(f'    Train≤{row["train_end"]} → Test {row["test_year"]}: {row["best_params"]}')
