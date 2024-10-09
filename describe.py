import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore


def df_description(df):
        print(df.head(5))
        print(df.tail(5))
        print(df.dtypes)

def shapes(df):
    print(df.shape)



def boxplot_msm_size(df):
    sns.boxplot(data = df, x="total_length")
    plt.savefig("result4.png",dpi=3000)
    plt.show()


def describeDataProtocols(df):
    print(df['most_freq_prot'].describe())
    print(df['most_freq_sport'].describe())


def boxplot_sum_et(df):
    print(df['sum_et'].describe())
    sns.boxplot(data = df, x="sum_et")
    plt.savefig("result5.png")
    plt.show()