import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
import csv

important_features = ['most_freq_sport','epoch_timestamp','most_freq_d_ip','most_freq_dport','L3_ip_dst_count']    # Ordre décroissant d'importance
train_acc = []
test_acc = []
csv_rows = [['train_acc','test_acc','feature','model']]



def save_as_csv():
    with open('models_testing.csv','w+',newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_rows)

def line_plot_acc_model(title, axs, x, y):
    sns.lineplot(data=train_acc, color='red', label="train_acc", ax=axs[x,y])
    sns.lineplot(data=test_acc, color='green', label="test_acc", ax=axs[x,y])
    axs[x,y].set_title(title)
    plt.setp(axs,xticks=[0,1,2,3,4],xticklabels=important_features)
    for ax in axs.flat:
        ax.set(xlabel="important features", ylabel="accuracy %")
        ax.set_xticklabels(important_features,rotation=90)
    plt.ylabel("pourcentage de précision")
    plt.legend()
    for i in range(len(train_acc)):
        csv_row = [train_acc[i],test_acc[i],important_features[i],title]
        csv_rows.append(csv_row)
    train_acc.clear()
    test_acc.clear()


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        accuracy = (accuracy_score(y_train, pred) * 100)
        train_acc.append(accuracy)
        print(f"Accuracy Score: {accuracy:.2f}%")
        print("_______________________________________________")
        #print(f"CLASSIFICATION REPORT:\n{clf_report}")
        #print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")  
        accuracy = (accuracy_score(y_test, pred) * 100)
        test_acc.append(accuracy)   
        print(f"Accuracy Score: {accuracy:.2f}%")
        print("_______________________________________________")
        #print(f"CLASSIFICATION REPORT:\n{clf_report}")
        #print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n\n")



def model_test(f, df, axs,x,y):
    for i in range(len(important_features)):
        print("============="+important_features[i]+"=============")
        features_df = pd.DataFrame()
        for j in range(i+1):
            features_df[important_features[j]] = df[important_features[j]]

        features_df['device_category'] = df['device_category']
        f(features_df, important_features[i])
    line_plot_acc_model(f.__name__, axs,x,y)

def model_testing(df):
    models= [random_forest_classifier,gradient_boosting_classifier,ada_boost_classifier,bagging_classifier,extra_trees_classifier]
    #models = [voting_classifier,stacking_classifier]
    fig, axs = plt.subplots(3,2)
    x = 0
    y = 0
    for model in models:
        print("\n\n=========================="+model.__name__+"==========================")
        model_test(model, df, axs,x,y)
        y+=1
        if y>2:
            y=0
            x+=1
    plt.show()
    #save_as_csv()
    
    #voting_classifier(features_df, feature)
    #line_plot_acc_model("VotingClassifier")
    #stacking_classifier(features_df, feature)
    #line_plot_acc_model("StackingClassifier")

        
    

def split(df, column):
    X = pd.DataFrame(df[column])
    y = df.device_category
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train,X_test,y_train,y_test

def random_forest_classifier(df, col):
    #df['device_name'] = df['device_name'].map({'amazonplug': 0, 'armcrest': 1,'arlobasecam':2,'arloqcam':3,'atomicoffeemaker':4,'boruncam':5,'dlinkcam':6,'echodot':7,'echospot':8,'echostudio':9,'eufyhomebase':10,'globelamp':11,'heimvisioncam':12,'heimvisionlamp':13,'homeeyecam':14,'luohecam':15,'nestcam':16,'nestmini':17,'netatmocam':18,'philipshue':19,'roomba':20,'simcam':21,'smartboard':22,'sonos':23,'teckin1':24,'teckin2':25,'yutron1':26,'yutron2':27})
    df['device_category'] = df['device_category'].map({'home_automation': 0, 'camera': 1, 'audio': 2})

    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    df.dropna(inplace=True)

    lr_clf = RandomForestClassifier(random_state=42)
    X_train, X_test, y_train, y_test = split(df, col)
    lr_clf.fit(X_train, y_train)

    print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)



def gradient_boosting_classifier(df, col):
    df['device_category'] = df['device_category'].map({'home_automation': 0, 'camera': 1, 'audio': 2})

    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    df.dropna(inplace=True)

    lr_clf = GradientBoostingClassifier(random_state=42)
    X_train, X_test, y_train, y_test = split(df, col)
    lr_clf.fit(X_train, y_train)

    print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)


def ada_boost_classifier(df, col):
    df['device_category'] = df['device_category'].map({'home_automation': 0, 'camera': 1, 'audio': 2})

    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    df.dropna(inplace=True)

    lr_clf = AdaBoostClassifier(random_state=42)
    X_train, X_test, y_train, y_test = split(df, col)
    lr_clf.fit(X_train, y_train)

    print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)

def bagging_classifier(df, col):
    df['device_category'] = df['device_category'].map({'home_automation': 0, 'camera': 1, 'audio': 2})

    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    df.dropna(inplace=True)

    lr_clf = BaggingClassifier(random_state=42)
    X_train, X_test, y_train, y_test = split(df, col)
    lr_clf.fit(X_train, y_train)

    print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)

def extra_trees_classifier(df, col):
    df['device_category'] = df['device_category'].map({'home_automation': 0, 'camera': 1, 'audio': 2})

    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    df.dropna(inplace=True)

    lr_clf = ExtraTreesClassifier(random_state=42)
    X_train, X_test, y_train, y_test = split(df, col)
    lr_clf.fit(X_train, y_train)

    print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)

def voting_classifier(df, col):
    df['device_category'] = df['device_category'].map({'home_automation': 0, 'camera': 1, 'audio': 2})

    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    df.dropna(inplace=True)

    clf1 = RandomForestClassifier(random_state=42)
    clf2 = GradientBoostingClassifier(random_state=42)
    clf3 = AdaBoostClassifier(random_state=42)
    clf4 = BaggingClassifier(random_state=42)
    clf5 = ExtraTreesClassifier(random_state=42)

    lr_clf = VotingClassifier(estimators=[('rf', clf1), ('gbc', clf2), ('ada', clf3), ('bag', clf4), ('ext', clf5)], voting='hard')
    X_train, X_test, y_train, y_test = split(df, col)
    lr_clf.fit(X_train, y_train)

    print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)


def stacking_classifier(df, col):
    df['device_category'] = df['device_category'].map({'home_automation': 0, 'camera': 1, 'audio': 2})

    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    df.dropna(inplace=True)

    clf1 = RandomForestClassifier(random_state=42)
    clf2 = GradientBoostingClassifier(random_state=42)
    clf3 = AdaBoostClassifier(random_state=42)
    clf4 = BaggingClassifier(random_state=42)
    clf5 = ExtraTreesClassifier(random_state=42)

    lr_clf = StackingClassifier(estimators=[('rf', clf1), ('gbc', clf2), ('ada', clf3), ('bag', clf4), ('ext', clf5)], final_estimator=RandomForestClassifier(random_state=42))
    X_train, X_test, y_train, y_test = split(df, col)
    lr_clf.fit(X_train, y_train)

    print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)