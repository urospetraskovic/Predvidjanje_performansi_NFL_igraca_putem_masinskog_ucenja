import pandas as pd
import matplotlib.pyplot as plt


def null_summary(datasets: dict) -> pd.DataFrame:
    rows = []
    for name, df in datasets.items():
        total_cells = df.shape[0] * df.shape[1]
        total_nulls = df.isnull().sum().sum()
        cols_with_nulls = (df.isnull().sum() > 0).sum()
        rows.append({
            "Skup": name.split("(")[0].strip(),
            "Ukupno ćelija": total_cells,
            "Null ćelija": total_nulls,
            "Null %": round(total_nulls / total_cells * 100, 2),
            "Kolona s nullovima": cols_with_nulls,
            "Ukupno kolona": df.shape[1],
        })
    return pd.DataFrame(rows)


def plot_missing(df, name, top_n=20):
    null_pct = df.isnull().mean() * 100
    null_pct = null_pct[null_pct > 0].sort_values(ascending=False).head(top_n)
    if null_pct.empty:
        print(f"{name}: nema nedostajućih vrednosti.")
        return
    fig, ax = plt.subplots(figsize=(12, max(3, len(null_pct) * 0.3)))
    colors = ["#e74c3c" if v >= 50 else "#f39c12" if v >= 10 else "#2ecc71" for v in null_pct.values]
    ax.barh(null_pct.index[::-1], null_pct.values[::-1], color=colors[::-1])
    ax.axvline(50, color="red", linestyle="--", alpha=0.4, label="50%")
    ax.set_xlabel("% nedostajućih vrednosti")
    ax.set_title(f"{name} — top {len(null_pct)} kolona s nullovima", fontweight="bold")
    ax.legend()
    for i, v in enumerate(null_pct.values[::-1]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_col_missing_by_year(df, col, year_col="Season", threshold=2006):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    splits = [(f"Pre {threshold}", df[year_col] < threshold), (f"Nakon {threshold}", df[year_col] >= threshold)]
    for ax, (label, mask) in zip(axes, splits):
        subset = df[mask]
        missing = subset[col].isnull().sum()
        present = subset[col].notnull().sum()
        ax.bar(["Prisutne", "Nedostajuće"], [present, missing], color=["#2ecc71", "#e74c3c"], edgecolor="black")
        ax.set_title(f"{col} — {label}\n(ukupno redova: {len(subset)})", fontweight="bold")
        ax.set_ylabel("Broj redova")
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(int(bar.get_height())), ha="center", va="bottom", fontsize=11)
        ax.set_ylim(0, len(subset) * 1.15)
    fig.suptitle(f"Nedostajuće vrednosti kolone {col} (prag: {threshold})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_col_group_missing_by_year(df, cols, year_col="Season", threshold=2006, group_label="adv_pass"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    splits = [(f"Pre {threshold}", df[year_col] < threshold), (f"Nakon {threshold}", df[year_col] >= threshold)]
    for ax, (label, mask) in zip(axes, splits):
        subset = df[mask]
        total_cells = len(subset) * len(cols)
        total_missing = subset[cols].isnull().sum().sum()
        total_present = total_cells - total_missing
        ax.bar(["Prisutne", "Nedostajuće"], [total_present, total_missing], color=["#2ecc71", "#e74c3c"], edgecolor="black")
        ax.set_title(f"{group_label} kolone — {label}\n(redova: {len(subset)}, kolona: {len(cols)})", fontweight="bold")
        ax.set_ylabel("Broj ćelija")
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total_cells * 0.01,
                    str(int(bar.get_height())), ha="center", va="bottom", fontsize=11)
        ax.set_ylim(0, total_cells * 1.15)
    fig.suptitle(f"Ukupno nedostajuće vrednosti — {group_label} kolone (prag: {threshold})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()