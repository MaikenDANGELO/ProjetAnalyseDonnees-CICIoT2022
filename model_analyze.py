import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt
import sklearn.metrics as skm

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
    print(len(ax.containers))
    for i in range(5):
        ax.bar_label(ax.containers[i],fontsize=10)
    plt.show()
    