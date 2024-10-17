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

train_acc = []
test_acc = []

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        accuracy = (accuracy_score(y_train, pred) * 100)
        train_acc.append(accuracy)
        print(f"Accuracy Score: {accuracy:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")  
        accuracy = (accuracy_score(y_test, pred) * 100)
        test_acc.append(accuracy)   
        print(f"Accuracy Score: {accuracy:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n\n")

def model_testing(df):
    important_features = ['most_freq_sport','epoch_timestamp','most_freq_d_ip','most_freq_dport','L3_ip_dst_count']    # Ordre décroissant d'importance

    for feature in important_features:
        print("============="+feature+"=============")
        features_df = pd.DataFrame()
        features_df[feature] = df[feature]
        features_df['device_category'] = df['device_category']
        random_forest_classifier(features_df, feature)
        gradient_boosting_classifier(features_df, feature)
        ada_boost_classifier(features_df, feature)
        bagging_classifier(features_df, feature)
        extra_trees_classifier(features_df, feature)
        voting_classifier(features_df, feature)
        stacking_classifier(features_df, feature)

        
    sns.lineplot(data=train_acc, color='red', label="train_acc")
    sns.lineplot(data=test_acc, color='green', label="test_acc")
    plt.xlabel("important features")
    plt.xticks([0,1,2,3,4],important_features, rotation=90)
    plt.ylabel("pourcentage de précision")
    plt.legend()
    plt.show()

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