import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.ensemble import RandomForestClassifier# type: ignore

def correlation_heat_map(df):
    df['device_name'] = df['device_name'].map({'amazonplug': 0.0, 'armcrest': 1.0,'arlobasecam':2.0,'arloqcam':3.0,'atomicoffeemaker':4.0,'boruncam':5.0,'dlinkcam':6.0,'echodot':7.0,'echospot':8.0,'echostudio':9.0,'eufyhomebase':10.0,'globelamp':11.0,'heimvisioncam':12.0,'heimvisionlamp':13.0,'homeeyecam':14.0,'luohecam':15.0,'nestcam':16.0,'nestmini':17.0,'netatmocam':18.0,'philipshue':19.0,'roomba':20.0,'simcam':21.0,'smartboard':22.0,'sonos':23.0,'teckin1':24.0,'teckin2':25.0,'yutron1':26.0,'yutron2':27.0})
    corr_matrix = df.corr()
    matrix = np.triu(corr_matrix)
    plt.subplots(figsize=(30,25), dpi=55)
    sns.heatmap(corr_matrix, cmap='coolwarm', mask=matrix, annot=True)
    plt.show()

def feature_importance(df):
    d = df.pop("device_name")
    model = RandomForestClassifier()
    model.fit(df,d)
    s = pd.Series(model.feature_importances_, index=df.columns)
    s = s.nlargest(5)
    s.plot(kind='bar')
    plt.show()

def pairplot_feature_importance(df):
    print("beginning pairplot..")
    sns.pairplot(data=df[['most_freq_dport','most_freq_d_ip','most_freq_sport','epoch_timestamp', 'L3_ip_dst_count', 'device_category']], hue="device_category", corner=True)
    plt.suptitle("Pair plot most important features")
    plt.show()

def pairplot_feature_importance_per_device(df):
    for name in df['device_name']:
        d = df[df['device_name'] == name]
        sns.pairplot(d[['most_freq_dport','most_freq_d_ip','most_freq_sport','epoch_timestamp']], hue="device_name")
        plt.title("Pair plot " + name)
        plt.savefig("./pairplots/"+name+".png")
        plt.show()

def kde_plot_mfreqdport(df):
    sns.kdeplot(data=df,x="most_freq_dport",hue="device_name")
    plt.show()

def kde_plot_mfreqsport(df):
    sns.kdeplot(data=df,x="most_freq_sport",hue="device_name")
    plt.show()

def kde_plot_mfreqdip(df):
    sns.kdeplot(data=df,x="most_freq_d_ip",hue="device_name")
    plt.show()

def kde_plot_epochtimestamp(df):
    sns.kdeplot(data=df,x="epoch_timestamp",hue="device_name")
    plt.show()

def line_plot_arlobasecam_dip_ts(df):
    d = df[df["device_name"] == "arlobasestationcam"]
    sns.lineplot(data=d, x="most_freq_dport", y="epoch_timestamp")
    plt.show()