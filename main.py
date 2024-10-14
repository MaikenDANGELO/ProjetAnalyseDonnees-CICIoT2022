import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

from load_data import load_data, get_files_from_dir, root_path
import analyze
import describe

df = load_data(get_files_from_dir(root_path))
df = pd.concat(df)

def clean_dataframe(df):
    df.set_index('Unnamed: 0', inplace=True)
    df.dropna()
    df.drop_duplicates()

clean_dataframe(df)

describe.df_description(df)

"""describe.shapes(df)
describe.boxplot_msm_size(df)
describe.describeDataProtocols(df)
describe.boxplot_sum_et(df)"""
describe.med_et(df)
describe.var(df)
describe.q1(df)
describe.q3(df)
describe.sum_e(df)
describe.max_e(df)
describe.average(df)
#describe.describeTotalLength(df)
#analyze.barplot_dns_request_by_ip(df)   

#analyze.barplot_dns_request_by_ip(df)
#analyze.correlation_heat_map(df)

