import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from load_data import load_data, get_files_from_dir, root_path
import analyze

df = load_data(get_files_from_dir(root_path))

def clean_dataframe(df):
    for d in df:
        d.set_index('Unnamed: 0', inplace=True)
        d.isna()
        d.drop_duplicates()

clean_dataframe(df)
analyze.barplot_dns_request_by_ip(df)