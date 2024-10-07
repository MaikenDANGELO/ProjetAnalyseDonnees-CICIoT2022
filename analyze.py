import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore


def df_description(df):
    for d in df:
        print(d.head(5))
        print(d.tail(5))

def barplot_dns_request_by_ip(df):
    for d in df:
        sns.barplot(data = d, x="ip_dst_new", y="DNS_count", hue="device_feature")
    plt.savefig("result.png")
    plt.show()

def kdeplot_sourceport_to_destination_port(df):
    for d in df:
        sns.kdeplot(data=d, x="source_port", y="dest_port", hue="device_name")
    plt.savefig("result2.png")
    plt.show()
