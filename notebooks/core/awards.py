from collections import Counter

import pandas as pd


def awards_summary(award_map: dict) -> None:
    all_awards_total: Counter = Counter()
    pos_awards: dict[str, Counter] = {}

    for pos_name, (col_name, df) in award_map.items():
        counter: Counter = Counter()
        for val in df[col_name].dropna():
            if pd.isna(val) or str(val).strip() == "":
                continue
            for award in [p.strip() for p in str(val).split(",") if p.strip()]:
                counter[award] += 1
                all_awards_total[award] += 1
        pos_awards[pos_name] = counter

    positions = list(award_map.keys())
    col_w = max(len(p) for p in positions)

    header_parts = f"{'Nagrada':<25}" + "".join(f" {p:>{max(col_w, 6)}}" for p in positions) + f" {'UKUPNO':>8}"
    sep = "-" * len(header_parts)

    print("=" * len(header_parts))
    print(f"PREGLED SVIH VRSTA NAGRADA U DATASETU ({' + '.join(positions)})")
    print("=" * len(header_parts))
    print(f"\n{header_parts}")
    print(sep)

    for award in sorted(all_awards_total, key=all_awards_total.__getitem__, reverse=True):
        row = f"{award:<25}"
        for pos in positions:
            cnt = pos_awards[pos].get(award, 0)
            row += f" {cnt:>{max(col_w, 6)}}"
        row += f" {all_awards_total[award]:>8}"
        print(row)

    print(sep)
    totals = f"{'UKUPNO NAGRADA':<25}" + "".join(
        f" {sum(pos_awards[p].values()):>{max(col_w, 6)}}" for p in positions
    ) + f" {sum(all_awards_total.values()):>8}"
    print(totals)
    print(f"\nBroj razlicitih vrsta nagrada: {len(all_awards_total)}")


# Defaultne kategorije nagrada
DEFAULT_AWARD_CATEGORIES = {
    "Pro Bowl (PB)":    ["PB"],
    "All-Pro 1st Team": ["AP-1"],
    "All-Pro 2nd Team": ["AP-2"],
    "TOP 5 MVP":        ["MVP-1", "MVP-2", "MVP-3", "MVP-4", "MVP-5"],
    "TOP 5 OPoY":       ["OPoY-1", "OPoY-2", "OPoY-3", "OPoY-4", "OPoY-5"],
}


def awards_category_summary(
    award_map: dict,
    award_categories: dict | None = None,
) -> None:
    if award_categories is None:
        award_categories = DEFAULT_AWARD_CATEGORIES

    def _has_any(val: object, keywords: list[str]) -> bool:
        if pd.isna(val) or str(val).strip() == "":
            return False
        s = str(val)
        return any(kw in s for kw in keywords)

    positions = list(award_map.keys())
    rows = []
    for cat_name, keywords in award_categories.items():
        row: dict = {"Nagrada": cat_name}
        for pos_name, (col_name, df) in award_map.items():
            count = df[col_name].apply(lambda v, kw=keywords: _has_any(v, kw)).sum()
            pct = count / len(df) * 100
            row[f"{pos_name} (broj)"] = int(count)
            row[f"{pos_name} (%)"] = round(pct, 1)
        rows.append(row)

    totals = {pos_name: len(df) for pos_name, (_, df) in award_map.items()}
    print("_" * 80)
    print("BROJ SEZONA SA NAGRADOM PO POZICIJI (od ukupno {})".format(
        ", ".join(f"{p}={totals[p]}" for p in positions)
    ))
    print()

    header = f"{'Kategorija':<22} | " + " | ".join(f"{p:>10}" for p in positions)
    print(header)
    print("-" * len(header))

    for row in rows:
        parts = []
        for p in positions:
            parts.append(f"{row[f'{p} (broj)']:>4} ({row[f'{p} (%)']:>5.1f}%)")
        print(f"{row['Nagrada']:<22} | " + " | ".join(f"{s:>10}" for s in parts))

    print("-" * len(header))


# Defaultne kategorije za pie chart
DEFAULT_PIE_AWARD_CHECKS = {
    "Pro Bowl (PB)":          ["PB"],
    "All-Pro (AP-1 / AP-2)":  ["AP-1", "AP-2"],
    "MVP (1–5)":              ["MVP-1", "MVP-2", "MVP-3", "MVP-4", "MVP-5"],
    "OPoY (1–5)":             ["OPoY-1", "OPoY-2", "OPoY-3", "OPoY-4", "OPoY-5"],
}

DEFAULT_PIE_COLORS = {
    "Pro Bowl (PB)":          ["#3498db", "#bdc3c7"],
    "All-Pro (AP-1 / AP-2)":  ["#e67e22", "#bdc3c7"],
    "MVP (1–5)":              ["#9b59b6", "#bdc3c7"],
    "OPoY (1–5)":             ["#1abc9c", "#bdc3c7"],
}


def plot_awards_pie(
    award_map: dict,
    award_checks: dict | None = None,
    award_colors: dict | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if award_checks is None:
        award_checks = DEFAULT_PIE_AWARD_CHECKS
    if award_colors is None:
        award_colors = DEFAULT_PIE_COLORS

    def _has_any(val: object, keywords: list[str]) -> bool:
        if pd.isna(val) or str(val).strip() == "":
            return False
        s = str(val)
        return any(kw in s for kw in keywords)

    for award_name, keywords in award_checks.items():
        colors = award_colors.get(award_name, ["#3498db", "#bdc3c7"])
        fig, axes = plt.subplots(1, len(award_map), figsize=(16, 4.5))
        if len(award_map) == 1:
            axes = [axes]

        for ax, (pos, (col, df)) in zip(axes, award_map.items()):
            total = len(df)
            n_with = int(df[col].apply(lambda v, kw=keywords: _has_any(v, kw)).sum())
            n_without = total - n_with
            labels = [
                f"Sa {award_name}\n({n_with} od {total})",
                f"Bez\n({n_without} sezona)",
            ]
            wedges, texts, autotexts = ax.pie(
                [n_with, n_without], labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 10},
                wedgeprops={"edgecolor": "white", "linewidth": 2},
            )
            for at in autotexts:
                at.set_fontweight("bold")
                at.set_fontsize(12)
            ax.set_title(
                f"{pos} — {award_name}\n(ukupno {total} sezona)",
                fontsize=13, fontweight="bold", pad=12,
            )

        fig.suptitle(
            f"Procenat sezona sa nagradom: {award_name}",
            fontsize=15, fontweight="bold", y=1.04,
        )
        plt.tight_layout()
        plt.show()
