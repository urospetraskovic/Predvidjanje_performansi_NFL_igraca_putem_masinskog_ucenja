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
