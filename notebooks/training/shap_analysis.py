import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def shorten_feat_name(name):
    return name.replace('_lag1', '¹').replace('_lag2', '²')


def compute_shap_values(best_models, processed, positions):
    shap_data = {}

    for pos in positions:
        best_name, best_model = best_models[pos]
        X_tr, X_te, y_tr, y_te = processed[pos]

        print(f'{pos:<4} ({best_name:<20}) — računanje SHAP ...')

        if isinstance(best_model, (RandomForestRegressor, XGBRegressor, LGBMRegressor)):
            explainer = shap.TreeExplainer(best_model)
            sv        = explainer.shap_values(X_te)
            base      = float(np.atleast_1d(explainer.expected_value)[0])
            exp_type  = 'TreeExplainer'

        elif isinstance(best_model, (LinearRegression, Ridge, Lasso, ElasticNet)):
            explainer = shap.LinearExplainer(best_model, X_tr)
            sv        = explainer.shap_values(X_te)
            base      = float(explainer.expected_value)
            exp_type  = 'LinearExplainer'

        else:
            bg        = shap.sample(X_tr, min(150, len(X_tr)), random_state=42)
            explainer = shap.KernelExplainer(best_model.predict, bg)
            sv        = explainer.shap_values(X_te, nsamples=200)
            base      = float(explainer.expected_value)
            exp_type  = 'KernelExplainer'

        shap_data[pos] = {
            'vals':           sv,
            'X_te':           X_te,
            'base':           base,
            'explainer_type': exp_type,
            'best_name':      best_name,
        }
        print(f'      Explainer: {exp_type} | base value: {base:.4f} | SHAP shape: {sv.shape}')

    print('\nSHAP vrijednosti izračunate za sve pozicije.')
    return shap_data


def plot_shap_beeswarm_and_bar(pos, shap_data, top_n, colors_pos):
    d       = shap_data[pos]
    sv      = d['vals']
    X_te_df = d['X_te']
    bname   = d['best_name']
    etype   = d['explainer_type']
    is_wr   = (pos == 'WR')

    feat_names = [shorten_feat_name(c) for c in X_te_df.columns]
    mean_abs   = np.abs(sv).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[::-1][:top_n]
    top_feats  = [feat_names[i] for i in top_idx]
    sv_top     = sv[:, top_idx]
    X_top      = X_te_df.iloc[:, top_idx].copy()
    X_top.columns = top_feats

    fig, (ax_bee, ax_bar) = plt.subplots(1, 2, figsize=(18, 8))

    # Beeswarm
    plt.sca(ax_bee)
    shap.summary_plot(sv_top, X_top, feature_names=top_feats,
                      plot_size=None, show=False, color_bar=True)
    ax_bee.set_title(
        f'{pos} — SHAP Beeswarm (top {top_n} feature-a)\n'
        f'Model: {bname} | Explainer: {etype}\n'
        f'{"WR target = log1p(yds/g) — SHAP u log prostoru" if is_wr else "Target = yds/g"}',
        fontsize=10, fontweight='bold'
    )
    ax_bee.set_xlabel('SHAP vrijednost (uticaj na predikciju)', fontsize=9)
    ax_bee.tick_params(axis='y', labelsize=8)

    # Bar chart
    vals_sorted  = mean_abs[top_idx][::-1]
    feats_sorted = top_feats[::-1]
    ax_bar.barh(range(top_n), vals_sorted, color=colors_pos[pos],
                alpha=0.85, edgecolor='white', linewidth=0.5)
    ax_bar.set_yticks(range(top_n))
    ax_bar.set_yticklabels(feats_sorted, fontsize=8)
    ax_bar.set_xlabel('Mean |SHAP value|', fontsize=9)
    ax_bar.set_title(
        f'{pos} — Feature Importance (Mean |SHAP|)\n'
        f'Model: {bname} | Test: sezona 2024',
        fontsize=10, fontweight='bold'
    )
    ax_bar.grid(True, alpha=0.22, axis='x')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    for i, fn in enumerate(feats_sorted):
        color = '#2c3e50' if fn.endswith('¹') else ('#7f8c8d' if fn.endswith('²') else '#e67e22')
        ax_bar.get_yticklabels()[i].set_color(color)

    leg = [
        mpatches.Patch(color='#2c3e50', label='lag1 (t-1 sezona)'),
        mpatches.Patch(color='#7f8c8d', label='lag2 (t-2 sezona)'),
        mpatches.Patch(color='#e67e22', label='statički feature (sezona t)'),
    ]
    ax_bar.legend(handles=leg, fontsize=7.5, loc='lower right')

    fig.suptitle(
        f'SHAP Analiza — {pos} (sezona 2024)\n'
        f'¹=lag1 (t-1)  ²=lag2 (t-2)  bez sufiksa=statički (Age, Team_Changed)',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()


def print_shap_top10(pos, shap_data, top_n=10):
    d          = shap_data[pos]
    sv         = d['vals']
    X_te_df    = d['X_te']
    feat_names = [shorten_feat_name(c) for c in X_te_df.columns]
    mean_abs   = np.abs(sv).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[::-1][:top_n]

    print(f'\n{"─"*70}')
    print(f'{pos} — Top {top_n} najvažnijih feature-a (Mean |SHAP|):')
    print(f'{"─"*70}')
    print(f'  {"#":>3}  {"Feature":<45} {"Mean |SHAP|":>12}  {"Avg vrednost u testu":>20}')
    print(f'  {"-"*82}')
    for rank, i in enumerate(top_idx, 1):
        short   = feat_names[i]
        m_shap  = mean_abs[i]
        avg_val = float(X_te_df.iloc[:, i].mean())
        print(f'  {rank:>3}. {short:<45} {m_shap:>12.5f}  {avg_val:>20.4f}')


def run_shap_analysis(best_models, processed,
                      positions=('QB', 'RB', 'TE', 'WR'),
                      colors_pos=None,
                      top_n=15):
    if colors_pos is None:
        colors_pos = {'QB': '#3498db', 'RB': '#e74c3c', 'TE': '#9b59b6', 'WR': '#2ecc71'}

    shap_data = compute_shap_values(best_models, processed, list(positions))

    for pos in positions:
        plot_shap_beeswarm_and_bar(pos, shap_data, top_n, colors_pos)
        print_shap_top10(pos, shap_data, top_n=10)

    plot_shap_comparison(shap_data, list(positions), colors_pos, top_n=10)

    print('\nSHAP analiza završena za sve pozicije.')
    return shap_data
    """import shap
        shap.initjs()

        shap_data = run_shap_analysis(
            best_models     = best_models,
            processed       = processed,
            positions       = ['QB', 'RB', 'TE', 'WR'],
            colors_pos      = {'QB': '#3498db', 'RB': '#e74c3c', 'TE': '#9b59b6', 'WR': '#2ecc71'},
            top_n           = 15,
        )"""


def plot_shap_comparison(shap_data, positions, colors_pos, top_n=10):
    fig, axes = plt.subplots(1, len(positions), figsize=(24, 8))

    for ax, pos in zip(axes, positions):
        d          = shap_data[pos]
        sv         = d['vals']
        X_te_df    = d['X_te']
        bname      = d['best_name']
        feat_names = [shorten_feat_name(c) for c in X_te_df.columns]
        mean_abs   = np.abs(sv).mean(axis=0)
        top_idx    = np.argsort(mean_abs)[::-1][:top_n]
        vals_s     = mean_abs[top_idx][::-1]
        feats_s    = [feat_names[i] for i in top_idx][::-1]

        ax.barh(range(top_n), vals_s, color=colors_pos[pos],
                alpha=0.88, edgecolor='white', linewidth=0.4)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feats_s, fontsize=8)
        ax.set_xlabel('Mean |SHAP|', fontsize=9)
        ax.set_title(f'{pos}\n({bname})', fontsize=11, fontweight='bold',
                     color=colors_pos[pos])
        ax.grid(True, alpha=0.2, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for i, fn in enumerate(feats_s):
            color = '#2c3e50' if fn.endswith('¹') else ('#7f8c8d' if fn.endswith('²') else '#e67e22')
            ax.get_yticklabels()[i].set_color(color)

    fig.suptitle(
        'SHAP Feature Importance — Komparativni prikaz svih pozicija (Top 10)\n'
        'Test sezona 2024 | ¹=lag1 (t-1) | ²=lag2 (t-2) | narančasto=statički',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()
