import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

def correlation_heat_map(df):
    df['device_name'] = df['device_name'].map({'amazonplug': 0.0, 'armcrest': 1.0,'arlobasecam':2.0,'arloqcam':3.0,'atomicoffeemaker':4.0,'boruncam':5.0,'dlinkcam':6.0,'echodot':7.0,'echospot':8.0,'echostudio':9.0,'eufyhomebase':10.0,'globelamp':11.0,'heimvisioncam':12.0,'heimvisionlamp':13.0,'homeeyecam':14.0,'luohecam':15.0,'nestcam':16.0,'nestmini':17.0,'netatmocam':18.0,'philipshue':19.0,'roomba':20.0,'simcam':21.0,'smartboard':22.0,'sonos':23.0,'teckin1':24.0,'teckin2':25.0,'yutron1':26.0,'yutron2':27.0})
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.savefig('result.png')
    plt.show()


def barplot_dns_request_by_ip(df):
    sns.lineplot(data = df, x="inter_arrival_time", y="average_et", hue="device_feature")
    plt.savefig("result.png")
    plt.show()
