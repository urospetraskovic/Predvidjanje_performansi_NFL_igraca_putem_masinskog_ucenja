import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os


def setup():
    warnings.filterwarnings("ignore")

    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.width", 120)

    plt.rcParams["figure.figsize"] = (14, 6)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    sns.set_palette("tab10")


def load_datasets(data_dir: str | None = None) -> dict:
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fully combined")

    qb       = pd.read_csv(os.path.join(data_dir, "qb_master.csv"))
    rb       = pd.read_csv(os.path.join(data_dir, "rb_master.csv"))
    te       = pd.read_csv(os.path.join(data_dir, "te_master.csv"))
    wr_seasons = pd.read_csv(os.path.join(data_dir, "wr_all_seasons.csv"))

    datasets = {
        "QB (qb_master)":   qb,
        "RB (rb_master)":   rb,
        "TE (te_master)":   te,
        "WR (wr_all_seasons)": wr_seasons,
    }

    print("Skupovi podataka uspesno ucitani:\n")
    for name, df in datasets.items():
        print(f"  {name:40s}    {df.shape[0]:>6} redova  ×  {df.shape[1]:>3} kolona")

    return datasets

def dtype_summary(datasets: dict) -> pd.DataFrame:
    rows = []
    for name, df in datasets.items():
        for dtype, count in df.dtypes.value_counts().items():
            rows.append({"Pozicija": name.split("(")[0].strip(), "Tip": str(dtype), "Broj kolona": count})

    result = pd.DataFrame(rows).pivot_table(index="Pozicija", columns="Tip", values="Broj kolona", fill_value=0).astype(int)
    result.index.name = None
    result.columns.name = None
    return result
