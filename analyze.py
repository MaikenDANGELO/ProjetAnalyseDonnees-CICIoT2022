import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore


def df_description(df):
        print(df.head(5))
        print(df.tail(5))

def barplot_dns_request_by_ip(df):
    sns.lineplot(data = df, x="inter_arrival_time", y="average_et", hue="device_feature")
    plt.savefig("result.png")
    plt.show()
