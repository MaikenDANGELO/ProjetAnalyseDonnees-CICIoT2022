import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

def barplot_dns_request_by_ip(df):
    for d in df:
        sns.barplot(data = d, x="ip_dst_new", y="DNS_count", hue="device_name")
    plt.savefig("result.png")
    plt.show()