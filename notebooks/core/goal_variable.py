import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, probplot


def plot_target_dist(datasets: dict, plot_data: list):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (df, col, title, color) in zip(axes.flat, plot_data):
        data = df[col].dropna()
        ax.hist(data, bins=40, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=2, label=f"Mean: {data.mean():.0f}")
        ax.axvline(data.median(), color="orange", linestyle="-", linewidth=2, label=f"Median: {data.median():.0f}")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Yards")
        ax.set_ylabel("Frekvencija")
        ax.legend(fontsize=10)
    fig.suptitle("Distribucija ciljnih varijabli - Yards po poziciji", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_target_kde(plot_data: list):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (df, col, title, color) in zip(axes.flat, plot_data):
        data = df[col].dropna()
        kde = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 300)
        ax.plot(x, kde(x), color=color, linewidth=2.5)
        ax.fill_between(x, kde(x), alpha=0.3, color=color)
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=2, label=f"Mean: {data.mean():.0f}")
        ax.axvline(data.median(), color="orange", linestyle="-", linewidth=2, label=f"Median: {data.median():.0f}")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Yards")
        ax.set_ylabel("Gustina")
        ax.legend(fontsize=10)
    fig.suptitle("Distribucija ciljnih varijabli - Gustina (KDE) po poziciji", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_target_qq(plot_data: list):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (df, col, title, color) in zip(axes.flat, plot_data):
        data = df[col].dropna()
        (osm, osr), (slope, intercept, r) = probplot(data, dist="norm")
        slope_f, intercept_f = float(slope), float(intercept)
        ax.scatter(osm, osr, color=color, alpha=0.5, s=15, label="Podaci")
        ax.plot(osm, slope_f * np.array(osm) + intercept_f, color="black", linewidth=1.5, label=f"R\u00b2={float(r)**2:.3f}")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Teorijski kvantili")
        ax.set_ylabel("Uzorak kvantili")
        ax.legend(fontsize=10)
    fig.suptitle("Q-Q Plot ciljnih varijabli - Yards po poziciji", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_pareto_lorenz(df, col: str = "receiving_yards", color: str = "#2ecc71", label: str = "WR Receiving Yards"):
    yds_sorted = df[col].dropna().sort_values(ascending=False).values
    yds_sorted = yds_sorted[yds_sorted > 0]

    total_yds = yds_sorted.sum()
    cumulative_yds = np.cumsum(yds_sorted) / total_yds
    player_pct = np.arange(1, len(yds_sorted) + 1) / len(yds_sorted)

    idx_20 = np.searchsorted(player_pct, 0.20)
    yds_by_top20 = cumulative_yds[idx_20] * 100

    idx_80 = np.searchsorted(cumulative_yds, 0.80)
    pct_for_80 = player_pct[idx_80] * 100

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(player_pct * 100, cumulative_yds * 100, color=color, linewidth=2.5, label=label)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.4, label="Savršena jednakost")

    ax.axvline(20, color="red", linestyle=":", alpha=0.7)
    ax.axhline(yds_by_top20, color="red", linestyle=":", alpha=0.7)
    ax.plot(20, yds_by_top20, "ro", markersize=10)
    ax.annotate(
        f"Top 20% igraca = {yds_by_top20:.1f}% jardi",
        xy=(20, yds_by_top20), xytext=(35, yds_by_top20 - 10),
        fontsize=11, color="red", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    ax.axhline(80, color="blue", linestyle=":", alpha=0.7)
    ax.axvline(pct_for_80, color="blue", linestyle=":", alpha=0.7)
    ax.plot(pct_for_80, 80, "bs", markersize=10)
    ax.annotate(
        f"80% jardi = top {pct_for_80:.1f}% igraca",
        xy=(pct_for_80, 80), xytext=(pct_for_80 + 10, 70),
        fontsize=11, color="blue", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="blue"),
    )

    ax.fill_between(player_pct * 100, cumulative_yds * 100, player_pct * 100, alpha=0.15, color="#e74c3c", label="Gini oblast")

    ax.set_xlabel("% igraca (rangirani od najboljeg)", fontsize=12)
    ax.set_ylabel("% kumulativnih receiving jardi", fontsize=12)
    ax.set_title(f"{label} — Pareto (Lorenz) kriva", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_wr_log_transform(wr_seasons):
    wr_yds = wr_seasons["receiving_yards"].dropna()
    wr_yds_positive = wr_yds[wr_yds > 0]
    wr_yds_log = np.log(wr_yds_positive)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1) Original histogram
    axes[0].hist(wr_yds, bins=40, color="#2ecc71", alpha=0.7, edgecolor="white", linewidth=0.5)
    axes[0].axvline(wr_yds.mean(), color="black", linestyle="--", linewidth=2, label=f"Mean: {wr_yds.mean():.0f}")
    axes[0].axvline(wr_yds.median(), color="orange", linestyle="-", linewidth=2, label=f"Median: {wr_yds.median():.0f}")
    axes[0].set_title("WR Receiving Yards — Original", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Yards")
    axes[0].set_ylabel("Frekvencija")
    axes[0].legend(fontsize=9)

    # 2) Log-transformed histogram
    axes[1].hist(wr_yds_log, bins=40, color="#27ae60", alpha=0.7, edgecolor="white", linewidth=0.5)
    axes[1].axvline(wr_yds_log.mean(), color="black", linestyle="--", linewidth=2, label=f"Mean: {wr_yds_log.mean():.2f}")
    axes[1].axvline(wr_yds_log.median(), color="orange", linestyle="-", linewidth=2, label=f"Median: {wr_yds_log.median():.2f}")
    axes[1].set_title("WR Receiving Yards — Log transformacija", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("log(Yards)")
    axes[1].set_ylabel("Frekvencija")
    axes[1].legend(fontsize=9)

    # 3) Q-Q plot log-transformed
    (osm, osr), (slope, intercept, r) = probplot(wr_yds_log, dist="norm")
    axes[2].scatter(osm, osr, color="#2ecc71", alpha=0.5, s=15, label="Podaci")
    axes[2].plot(osm, slope * np.array(osm) + intercept, color="black", linewidth=1.5, label=f"R\u00b2={r**2:.3f}")
    axes[2].set_title("WR log(Yards) — Q-Q Plot", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Teorijski kvantili")
    axes[2].set_ylabel("Uzorak kvantili")
    axes[2].legend(fontsize=9)

    fig.suptitle(
        f"WR Receiving Yards: efekat log transformacije "
        f"(n={len(wr_yds_positive)}, uklonjeno \u22640: {len(wr_yds) - len(wr_yds_positive)})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.show()

    r2_before = probplot(wr_yds, dist="norm")[1][2] ** 2
    print(f"R\u00b2 pre:   {r2_before:.3f}")
    print(f"R\u00b2 nakon: {r**2:.3f}")
