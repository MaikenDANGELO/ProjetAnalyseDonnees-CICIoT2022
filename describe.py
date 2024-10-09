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
    plt.savefig("result4.png")
    plt.show()


def describeTotalLength(df):
    print(df['total_length'].describe())
    print(df['total_length'].value_counts())
    df['total_length'].value_counts()[df['total_length'].value_counts() > 300].plot(kind='bar', ylim=(300, 40000))
    #sns.lineplot(data = df, x="total_length", y=df['total_length'].value_counts())
    plt.xlabel('Total Length')
    plt.ylabel('Frequency')
    plt.show()

"""
min_et(same)
med_et(same)
var(same)
ql(same)
sum_e(same)
max_e(same)
average(same)
"""

def min_et(df):
    print(df['min_et'].describe())
    print(df['min_et'].value_counts())
    df['min_et'].value_counts()[df['min_et'].value_counts() > 300].plot(kind='bar', ylim=(300, 40000))
    plt.xlabel('Min ET')
    plt.ylabel('Frequency')
    plt.title('Min ET')
    plt.show()

def med_et(df):
    print(df['med_et'].describe())
    print(df['med_et'].value_counts())
    df['med_et'].value_counts()[df['med_et'].value_counts() > 300].plot(kind='bar', ylim=(300, 40000))
    plt.xlabel('Med ET')
    plt.ylabel('Frequency')
    plt.title('Med ET')
    plt.show()

def var(df):
    print(df['var'].describe())
    print(df['var'].value_counts())
    df['var'].value_counts()[df['var'].value_counts() > 300].plot(kind='bar', ylim=(300, 40000))
    plt.xlabel('Var')
    plt.ylabel('Frequency')
    plt.title('Var')
    plt.show()

def q1(df):
    print(df['q1'].describe())
    print(df['q1'].value_counts())
    df['q1'].value_counts()[df['q1'].value_counts() > 300].plot(kind='bar', ylim=(300, 40000))
    plt.xlabel('Q1')
    plt.ylabel('Frequency')
    plt.title('Q1')
    plt.show()


def q3(df):
    print(df['q3'].describe())
    print(df['q3'].value_counts())
    df['q3'].value_counts()[df['q3'].value_counts() > 300].plot(kind='bar', ylim=(300, 40000))
    plt.xlabel('Q3')
    plt.ylabel('Frequency')
    plt.title('Q3')
    plt.show()



def sum_e(df):
    print(df['sum_e'].describe())
    print(df['sum_e'].value_counts())
    df['sum_e'].value_counts()[df['sum_e'].value_counts() > 300].plot(kind='bar', ylim=(300, 50000))
    plt.xlabel('Sum E')
    plt.ylabel('Frequency')
    plt.title('Sum E')
    plt.show()

def max_e(df):
    print(df['max_e'].describe())
    print(df['max_e'].value_counts())
    df['max_e'].value_counts()[df['max_e'].value_counts() > 300].plot(kind='bar', ylim=(300, 60000))
    plt.xlabel('Max E')
    plt.ylabel('Frequency')
    plt.title('Max E')
    plt.show()

def average(df):
    print(df['average'].describe())
    print(df['average'].value_counts())
    df['average'].value_counts()[df['average'].value_counts() > 300].plot(kind='bar', ylim=(300, 40000))
    plt.xlabel('Average')
    plt.ylabel('Frequency')
    plt.title('Average')
    plt.show()


def describeDataProtocols(df):
    print(df['most_freq_prot'].describe())
    print(df['most_freq_sport'].describe())


def boxplot_sum_et(df):
    print(df['sum_et'].describe())
    sns.boxplot(data = df, x="sum_et")
    plt.savefig("result5.png")
    plt.show()