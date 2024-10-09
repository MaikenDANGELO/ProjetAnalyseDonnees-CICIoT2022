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



def boxpolt_dns_request_by_ip(df):
    sns.boxplot(data = df, x="inter_arrival_time", y="average_et", hue="device_feature")
    plt.savefig("result.png")
    plt.show()