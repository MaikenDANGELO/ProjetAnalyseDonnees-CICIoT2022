import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt
import sklearn.metrics as skm
important_features = ['most_freq_sport','epoch_timestamp','most_freq_d_ip','most_freq_dport','L3_ip_dst_count']    

def plot_mean_acc():
    models_df = pd.read_csv('models_testing.csv')
    models_df['train_acc_mean'] = models_df.groupby('model')['train_acc'].transform('mean')
    models_df['train_acc_mean'].sort_values(ascending=False)
    ax = sns.barplot(data=models_df, x="model",y="train_acc_mean",hue="model", order=models_df.sort_values('train_acc_mean',ascending=False).model)
    for i in range(5):
        ax.bar_label(ax.containers[i],fontsize=10)
    plt.title('Mean accuracy of models')
    plt.show()
# Use sklearn.metrics instead of doing it here !! https://scikit-learn.org/1.5/api/sklearn.metrics.html

def test():
    mean_acc_df = pd.read_csv('models_mean_acc.csv')
    ax = sns.barplot(data=mean_acc_df, x='model', y='train_acc', hue='model', order=mean_acc_df.sort_values('train_acc', ascending=False).model)
    for i in range(5):
        ax.bar_label(ax.containers[i],fontsize=10)
    plt.show()

# Il faudra ajouter les valeurs de test.
def plot_models_features():
    df = pd.read_csv('models_mean_acc.csv')
    fig, axs = plt.subplots(2,2)

    ax1 = sns.barplot(data=df, x='model', y='train_acc', hue='model', order=df.sort_values('train_acc', ascending=False).model, ax=axs[0,0])
    axs[0,0].set_title("Train acc")
    for i in range(5):
        ax1.bar_label(ax1.containers[i],fontsize=10)

    ax2 = sns.barplot(data=df, x='model', y='recall_score', hue='model', order=df.sort_values('recall_score', ascending=False).model, ax=axs[0,1])
    axs[0,1].set_title("Recall score")
    for i in range(5):
        ax2.bar_label(ax2.containers[i],fontsize=10)

    ax3 = sns.barplot(data=df, x='model', y='f1_score', hue='model', order=df.sort_values('f1_score', ascending=False).model, ax=axs[1,0])
    axs[1,0].set_title("F1 Score")
    for i in range(5):
        ax3.bar_label(ax3.containers[i],fontsize=10)

    for ax in axs.flat:
        ax.set(xlabel="important features", ylabel="%")
        ax.set_xticklabels(important_features, fontsize=8)

    plt.title('Features Models comparison')
    plt.show()
    