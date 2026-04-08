import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(r"c:\Users\Win10\Documents\GitHub\Analiza-i-Obrada")
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
FIG_DIR = REPO_ROOT / "Izvestaj" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(NOTEBOOKS_DIR / "core"))
os.chdir(NOTEBOOKS_DIR)

from notebook_setup import setup, load_datasets  # noqa: E402
from missing_data import plot_missing, plot_col_missing_by_year, plot_col_group_missing_by_year  # noqa: E402
from goal_variable import (  # noqa: E402
    plot_target_dist,
    plot_target_kde,
    plot_target_qq,
    plot_pareto_lorenz,
    plot_wr_log_transform,
)
from outliers import (  # noqa: E402
    plot_outlier_boxplots,
    plot_qb_bimodal,
    plot_qb_qqplot,
    plot_wr_log_outliers,
)
from awards import plot_awards_pie  # noqa: E402
from corelations import (  # noqa: E402
    plot_qb_correlation_heatmap,
    plot_rb_correlation_heatmap,
    plot_wr_correlation_heatmap,
    plot_te_correlation_heatmap,
)
from seasonal_trends import (  # noqa: E402
    plot_qb_yards_by_era,
    plot_qb_passing_trends,
    plot_rb_yards_by_era,
    plot_rb_rushing_trends,
    plot_top10_seasons,
    plot_qb_career_scatter,
)
from year_analysis import (  # noqa: E402
    plot_qb_career_yards,
    plot_qb_career_yards_per_game,
    plot_rb_career_yards,
    plot_rb_career_yards_per_game,
    plot_qb_team_comparison,
    plot_rb_team_comparison,
)


def save_new_figures(prefix: str, dpi: int = 200):
    paths = []
    nums = plt.get_fignums()
    for i, n in enumerate(nums, 1):
        fig = plt.figure(n)
        out = FIG_DIR / f"{prefix}_{i}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        paths.append(str(out))
    plt.close("all")
    return paths


def main():
    setup()
    datasets = load_datasets()
    qb = datasets["QB (qb_master)"]
    rb = datasets["RB (rb_master)"]
    te = datasets["TE (te_master)"]
    wr_seasons = datasets["WR (wr_all_seasons)"]

    generated = []

    # Missing data
    for name, df in datasets.items():
        plot_missing(df, name.split("(")[0].strip())
    generated += save_new_figures("missing")

    adv_pass_cols = [c for c in qb.columns if c.startswith("adv_pass")]
    plot_col_missing_by_year(qb, col="QBR", threshold=2006)
    plot_col_group_missing_by_year(qb, cols=adv_pass_cols, threshold=2018, group_label="adv_pass")
    plot_col_group_missing_by_year(qb, cols=adv_pass_cols, threshold=2019, group_label="adv_pass")
    generated += save_new_figures("missing_by_year")

    # Distributions
    plot_data = [
        (qb, "Yds", "QB Passing Yards", "#3498db"),
        (rb, "Rush_Yds", "RB Rushing Yards", "#e74c3c"),
        (wr_seasons, "receiving_yards", "WR Receiving Yards", "#2ecc71"),
        (te, "Rec_Yds", "TE Receiving Yards", "#9b59b6"),
    ]
    plot_target_dist(datasets, plot_data)
    generated += save_new_figures("target_dist")

    plot_target_kde(plot_data)
    generated += save_new_figures("target_kde")

    plot_target_qq(plot_data)
    generated += save_new_figures("target_qq")

    plot_pareto_lorenz(wr_seasons)
    generated += save_new_figures("wr_pareto")

    plot_wr_log_transform(wr_seasons)
    generated += save_new_figures("wr_log_transform")

    # Outliers
    outlier_targets = {
        "QB": (qb, "Yds"),
        "RB": (rb, "Rush_Yds"),
        "TE": (te, "Rec_Yds"),
        "WR": (wr_seasons, "receiving_yards"),
    }
    plot_outlier_boxplots(outlier_targets, z_threshold=3)
    generated += save_new_figures("outliers_box")

    plot_qb_bimodal(qb)
    plot_qb_qqplot(qb)
    generated += save_new_figures("qb_bimodal")

    plot_wr_log_outliers(wr_seasons, z_threshold=3)
    generated += save_new_figures("wr_log_outliers")

    # Awards and correlations
    award_map = {
        "QB": ("Awards", qb),
        "RB": ("awards", rb),
        "TE": ("Awards", te),
    }
    plot_awards_pie(award_map)
    generated += save_new_figures("awards_pie")

    plot_qb_correlation_heatmap(qb)
    generated += save_new_figures("corr_qb")

    plot_rb_correlation_heatmap(rb)
    generated += save_new_figures("corr_rb")

    plot_wr_correlation_heatmap(wr_seasons)
    generated += save_new_figures("corr_wr")

    plot_te_correlation_heatmap(te)
    generated += save_new_figures("corr_te")

    # Trends and career charts
    plot_qb_yards_by_era(qb)
    plot_qb_passing_trends(qb)
    generated += save_new_figures("qb_trends")

    plot_rb_yards_by_era(rb)
    plot_rb_rushing_trends(rb)
    generated += save_new_figures("rb_trends")

    plot_top10_seasons(qb, rb, te, wr_seasons)
    generated += save_new_figures("top10")

    plot_qb_career_scatter(qb)
    generated += save_new_figures("qb_scatter")

    players_qb = ["Tom Brady", "Matthew Stafford", "Kirk Cousins"]
    colors_qb = {"Tom Brady": "#1565C0", "Matthew Stafford": "#C62828", "Kirk Cousins": "#2E7D32"}
    plot_qb_career_yards(players_qb, colors_qb)
    plot_qb_career_yards_per_game(players_qb, colors_qb)
    generated += save_new_figures("qb_career")

    players_rb = ["Adrian Peterson", "Alvin Kamara", "Saquon Barkley"]
    colors_rb = {"Adrian Peterson": "#6A1B9A", "Alvin Kamara": "#AD1457", "Saquon Barkley": "#004C54"}
    plot_rb_career_yards(players_rb, colors_rb)
    plot_rb_career_yards_per_game(players_rb, colors_rb)
    generated += save_new_figures("rb_career")

    plot_qb_team_comparison(
        player="Tom Brady",
        team_a="NWE",
        team_b="TAM",
        season_from=2018,
        season_to=2021,
        colors={"NWE": "#002244", "TAM": "#D50A0A"},
        title="Tom Brady: New England Patriots -> Tampa Bay Buccaneers",
    )
    generated += save_new_figures("qb_team_change")

    plot_rb_team_comparison(
        player="Saquon Barkley",
        team_a="NYG",
        team_b="PHI",
        season_from=2018,
        season_to=2024,
        colors={"NYG": "#0B2265", "PHI": "#004C54"},
        title="Saquon Barkley: New York Giants -> Philadelphia Eagles",
    )
    generated += save_new_figures("rb_team_change")

    print(f"Generated {len(generated)} figures")
    for p in generated:
        print(p)


if __name__ == "__main__":
    main()
