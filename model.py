import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

def split(df):
    X = df.drop('epoch_timestamp', axis=1)
    y = df.epoch_timestamp
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train,X_test,y_train,y_test

def linear_regression(df):
    df['device_name'] = df['device_name'].map({'amazonplug': 0.0, 'armcrest': 1.0,'arlobasecam':2.0,'arloqcam':3.0,'atomicoffeemaker':4.0,'boruncam':5.0,'dlinkcam':6.0,'echodot':7.0,'echospot':8.0,'echostudio':9.0,'eufyhomebase':10.0,'globelamp':11.0,'heimvisioncam':12.0,'heimvisionlamp':13.0,'homeeyecam':14.0,'luohecam':15.0,'nestcam':16.0,'nestmini':17.0,'netatmocam':18.0,'philipshue':19.0,'roomba':20.0,'simcam':21.0,'smartboard':22.0,'sonos':23.0,'teckin1':24.0,'teckin2':25.0,'yutron1':26.0,'yutron2':27.0})
    df['device_category'] = df['device_category'].map({'home_automation': 0.0, 'camera': 1.0, 'audio': 2.0})
    df['epoch_timestamp'] = pd.to_numeric(df['epoch_timestamp'], errors='coerce')
    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    df['epoch_timestamp'].dropna(inplace=True)
    lr_clf = RandomForestRegressor(random_state=42)
    X_train, X_test, y_train, y_test = split(df)
    lr_clf.fit(X_train, y_train)

    print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)

    # FAUT RÉGLER TOUT ÇA