import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def plot_outlier_boxplots(
    outlier_targets: dict,
    z_threshold: float = 3,
    colors: dict | None = None,
) -> None:
    if colors is None:
        colors = {"QB": "#3498db", "RB": "#e74c3c", "TE": "#9b59b6", "WR": "#2ecc71"}

    fig, axes = plt.subplots(1, len(outlier_targets), figsize=(5 * len(outlier_targets), 6))
    if len(outlier_targets) == 1:
        axes = [axes]

    for ax, (pos, (df, col)) in zip(axes, outlier_targets.items()):
        s = df[col].dropna()

        # IQR granice
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr_val = q3 - q1
        lower, upper = q1 - 1.5 * iqr_val, q3 + 1.5 * iqr_val

        # Z-score granice
        z_low = s.mean() - z_threshold * s.std()
        z_high = s.mean() + z_threshold * s.std()

        ax.boxplot(s, vert=True, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=colors.get(pos, "#95a5a6"), alpha=0.4),
                   medianprops=dict(color="black", linewidth=2),
                   flierprops=dict(marker="o", markersize=4, alpha=0.5))

        # Z-score linije
        ax.axhline(z_low,  color="red",  linestyle="--", alpha=0.7, label=f"Z low: {z_low:.0f}")
        ax.axhline(z_high, color="red",  linestyle="--", alpha=0.7, label=f"Z high: {z_high:.0f}")

        # IQR linije
        ax.axhline(lower, color="blue", linestyle=":", alpha=0.7, label=f"IQR low: {lower:.0f}")
        ax.axhline(upper, color="blue", linestyle=":", alpha=0.7, label=f"IQR high: {upper:.0f}")

        ax.set_title(f"{pos} — {col}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Yards")
        ax.set_xticklabels([pos])
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Boxplot sa Z-score (crveno) i IQR (plavo) granicama — sirove sezonske yarde",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.show()


def plot_qb_bimodal(qb: pd.DataFrame) -> None:
    qb_gs_pos = qb[qb["GS"] > 0]["Yds"].dropna()

    qb_starter = qb[qb["GS"] > 0].copy()
    qb_starter["Yds_per_G"] = qb_starter["Yds"] / qb_starter["G"]
    ypg = qb_starter["Yds_per_G"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1) Ukupne yarde — GS > 0
    axes[0].hist(qb_gs_pos, bins=30, color="#3498db", alpha=0.7, edgecolor="white")
    axes[0].axvline(qb_gs_pos.mean(),   color="black",  linestyle="--", linewidth=2,
                    label=f"Mean: {qb_gs_pos.mean():.0f}")
    axes[0].axvline(qb_gs_pos.median(), color="orange", linewidth=2,
                    label=f"Median: {qb_gs_pos.median():.0f}")
    axes[0].set_title(f"QB Yds — Samo starteri (GS > 0)\nn={len(qb_gs_pos)}",
                      fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Ukupne sezonske Yds")
    axes[0].set_ylabel("Frekvencija")
    axes[0].legend(fontsize=9)

    # 2) Yds per game — GS > 0
    axes[1].hist(ypg, bins=30, color="#2980b9", alpha=0.7, edgecolor="white")
    axes[1].axvline(ypg.mean(),   color="black",  linestyle="--", linewidth=2,
                    label=f"Mean: {ypg.mean():.0f}")
    axes[1].axvline(ypg.median(), color="orange", linewidth=2,
                    label=f"Median: {ypg.median():.0f}")
    axes[1].set_title(f"QB Yds/G — Starteri, per game\nn={len(ypg)}",
                      fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Yds po utakmici")
    axes[1].set_ylabel("Frekvencija")
    axes[1].legend(fontsize=9)

    fig.suptitle("QB: Uklanjanje GS=0 i prelazak na per-game metriku",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_qb_qqplot(qb: pd.DataFrame) -> None:
    qb_starter = qb[qb["GS"] > 0].copy()
    qb_starter["Yds_per_G"] = qb_starter["Yds"] / qb_starter["G"]
    ypg = qb_starter["Yds_per_G"].dropna()

    fig, ax = plt.subplots(figsize=(8, 6))
    (osm_p, osr_p), (slope, intercept, r_pg) = stats.probplot(ypg, dist="norm")
    ax.scatter(osm_p, osr_p, color="#2980b9", alpha=0.5, s=15)
    ax.plot(osm_p, slope * np.array(osm_p) + intercept, color="black",
            linewidth=1.5, label=f"R²={r_pg**2:.3f}")
    ax.set_title("QB Yds/G (GS>0) — Q-Q Plot", fontweight="bold", fontsize=13)
    ax.set_xlabel("Teorijski kvantili")
    ax.set_ylabel("Uzorak kvantili")
    ax.legend()
    plt.tight_layout()
    plt.show()
    print(f"R² (Yds/G, GS>0): {r_pg**2:.3f}")


def outlier_summary(outlier_targets: dict, z_threshold: float = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    z_rows = []
    iqr_rows = []

    for pos, (df, col) in outlier_targets.items():
        s = df[col].dropna()

        # Z-score
        z = (s - s.mean()) / s.std()
        n_z = (z.abs() > z_threshold).sum()
        z_rows.append({
            "Pozicija": pos,
            "Kolona": col,
            "N": len(s),
            "Outliera (|z|>3)": n_z,
            "% outliera": round(n_z / len(s) * 100, 2),
        })

        # IQR
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr_val = q3 - q1
        lower, upper = q1 - 1.5 * iqr_val, q3 + 1.5 * iqr_val
        n_iqr = ((s < lower) | (s > upper)).sum()
        iqr_rows.append({
            "Pozicija": pos,
            "Kolona": col,
            "N": len(s),
            "Q1": round(q1, 1),
            "Q3": round(q3, 1),
            "IQR": round(iqr_val, 1),
            "Donja granica": round(lower, 1),
            "Gornja granica": round(upper, 1),
            "Outliera": n_iqr,
            "% outliera": round(n_iqr / len(s) * 100, 2),
        })

    return pd.DataFrame(z_rows), pd.DataFrame(iqr_rows)


def plot_wr_log_outliers(
    wr_seasons: pd.DataFrame,
    z_threshold: float = 3,
    yds_col: str = "receiving_yards",
) -> None:
    s_raw = wr_seasons[yds_col].dropna()
    s_raw = s_raw[s_raw > 0]
    s_log = np.log(s_raw)

    # IQR & Z granice — raw
    q1_r, q3_r = s_raw.quantile(0.25), s_raw.quantile(0.75)
    iqr_r = q3_r - q1_r
    lo_r, hi_r = q1_r - 1.5 * iqr_r, q3_r + 1.5 * iqr_r
    z_raw = (s_raw - s_raw.mean()) / s_raw.std()
    is_z_raw   = z_raw.abs() > z_threshold
    is_iqr_raw = (s_raw < lo_r) | (s_raw > hi_r)
    z_hi_r = s_raw.mean() + z_threshold * s_raw.std()

    # IQR & Z granice — log
    q1_l, q3_l = s_log.quantile(0.25), s_log.quantile(0.75)
    iqr_l = q3_l - q1_l
    lo_l, hi_l = q1_l - 1.5 * iqr_l, q3_l + 1.5 * iqr_l
    z_log = (s_log - s_log.mean()) / s_log.std()
    is_z_log   = z_log.abs() > z_threshold
    is_iqr_log = (s_log < lo_l) | (s_log > hi_l)
    z_hi_l = s_log.mean() + z_threshold * s_log.std()
    z_lo_l = s_log.mean() - z_threshold * s_log.std()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ─── Sirove: histogram ───
    axes[0, 0].hist(s_raw, bins=50, color="#2ecc71", alpha=0.7, edgecolor="white")
    axes[0, 0].axvline(hi_r,   color="blue", linestyle=":", linewidth=2,
                       label=f"IQR gornja: {hi_r:.0f}")
    axes[0, 0].axvline(z_hi_r, color="red",  linestyle="--", linewidth=2,
                       label=f"Z gornja: {z_hi_r:.0f}")
    axes[0, 0].set_title(
        f"WR Sirove Yds — Histogram\n(IQR outliera: {is_iqr_raw.sum()}, Z outliera: {is_z_raw.sum()})",
        fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Receiving Yards")
    axes[0, 0].legend(fontsize=9)

    # ─── Sirove: boxplot ───
    axes[1, 0].boxplot(s_raw, vert=True, widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor="#2ecc71", alpha=0.4),
                       medianprops=dict(color="black", linewidth=2),
                       flierprops=dict(marker="o", markersize=4, alpha=0.5))
    axes[1, 0].set_title("WR Sirove Yds — Boxplot", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Receiving Yards")
    axes[1, 0].set_xticklabels(["WR"])

    # ─── Log: histogram ───
    axes[0, 1].hist(s_log, bins=50, color="#27ae60", alpha=0.7, edgecolor="white")
    axes[0, 1].axvline(hi_l,   color="blue", linestyle=":", linewidth=2,
                       label=f"IQR gornja: {hi_l:.1f}")
    axes[0, 1].axvline(lo_l,   color="blue", linestyle=":", linewidth=2,
                       label=f"IQR donja: {lo_l:.1f}")
    axes[0, 1].axvline(z_hi_l, color="red",  linestyle="--", linewidth=2,
                       label=f"Z gornja: {z_hi_l:.1f}")
    axes[0, 1].axvline(z_lo_l, color="red",  linestyle="--", linewidth=2,
                       label=f"Z donja: {z_lo_l:.1f}")
    axes[0, 1].set_title(
        f"WR log(Yds) — Histogram\n(IQR outliera: {is_iqr_log.sum()}, Z outliera: {is_z_log.sum()})",
        fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("log(Receiving Yards)")
    axes[0, 1].legend(fontsize=8)

    # ─── Log: boxplot ───
    axes[1, 1].boxplot(s_log, vert=True, widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor="#27ae60", alpha=0.4),
                       medianprops=dict(color="black", linewidth=2),
                       flierprops=dict(marker="o", markersize=4, alpha=0.5))
    axes[1, 1].axhline(lo_l, color="blue", linestyle=":", alpha=0.7,
                       label=f"IQR donja: {lo_l:.1f}")
    axes[1, 1].axhline(hi_l, color="blue", linestyle=":", alpha=0.7,
                       label=f"IQR gornja: {hi_l:.1f}")
    axes[1, 1].set_title("WR log(Yds) — Boxplot", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("log(Receiving Yards)")
    axes[1, 1].set_xticklabels(["WR (log)"])
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("WR: Outlier detekcija — sirove yarde vs log(yarde)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def wr_log_outlier_table(
    wr_seasons: pd.DataFrame,
    z_threshold: float = 3,
    yds_col: str = "receiving_yards",
) -> pd.DataFrame:
    s_raw = wr_seasons[yds_col].dropna()
    s_raw = s_raw[s_raw > 0]
    s_log = np.log(s_raw)

    def _counts(s: pd.Series) -> dict:
        z = (s - s.mean()) / s.std()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_z   = (z.abs() > z_threshold).sum()
        n_iqr = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        return {"N": len(s), "Z outliera": n_z, "IQR outliera": n_iqr,
                "Z %": round(n_z / len(s) * 100, 2),
                "IQR %": round(n_iqr / len(s) * 100, 2)}

    rows = [{"Skala": "Sirove Yds", **_counts(s_raw)},
            {"Skala": "log(Yds)",   **_counts(s_log)}]
    return pd.DataFrame(rows)
