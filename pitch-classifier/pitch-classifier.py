# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 20:54:03 2020

@author: Tom
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import precision_score, accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#Data Ingest
raw_pitch_data = pd.read_csv('pitches.csv', nrows = 100000)

#Data Exploration
raw_pitch_data.columns

raw_pitch_data.pitch_type.unique()

raw_pitch_data.pitch_type.value_counts().plot(kind ='bar')

#Correlation plot
corr = raw_pitch_data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

#Distribution plots
boxplot_features = ['start_speed','end_speed','spin_rate', 'spin_dir',
                    'break_angle', 'break_length','break_y',
                    'pfx_x','pfx_z', 'x0', 'y', 'z0', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az']

fig, axs = plt.subplots(ncols=2,
                        nrows=int(len(boxplot_features)/2),
                        figsize=(20, 30))

for iteration, feature in enumerate(boxplot_features):
    
    if iteration > 8:
        col = 1
    else:
        col = 0
        
    row = iteration % int(len(boxplot_features)/2)
    
    sns.boxplot(x='pitch_type', y=feature, data=raw_pitch_data, ax=axs[row, col])


#Data Preparation
unwanted_pitch_types = ['SC', 'UN', 'FA', 'PO', 'FO', 'EP', 'IN', 'KN', 'FS', 'KC']
unwanted_pitches = raw_pitch_data['pitch_type'].isin(unwanted_pitch_types)
pitches = raw_pitch_data[-unwanted_pitches]

pitches.isnull().sum()
pitches = pitches.dropna()

inde = pitches[boxplot_features]

le = LabelEncoder()
depe = le.fit_transform(pitches['pitch_type'])

code_lookup = pd.DataFrame({'type': pitches['pitch_type'], 'code': depe}).drop_duplicates()
code_lookup = pd.Series(code_lookup.type.values ,index=code_lookup.code.values).to_dict()

X_train, X_test, y_train, y_test = train_test_split(inde, depe, test_size=0.2, random_state=1)

#Create 1 or two classifications models

##xgboost

def xgboostClassifier(X_train, y_train, X_test, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {
        'max_depth': 3,  # the maximum depth of each tree
        'eta': 0.3,  # the training step for each iteration
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'num_class': 10,
        'early_stopping_rounds':10}  # the number of classes that exist in this datset

    num_round = 999  # the number of training iterations

    bst = xgb.train(param, dtrain, num_round)
    preds = bst.predict(dtest)
    
    return bst, np.asarray([np.argmax(line) for line in preds])

#random forest

def RFClassifier(X_train, y_train, X_test, y_test):
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    return clf, clf.predict(X_test)

rf_model, rf_pred = RFClassifier(X_train, y_train, X_test, y_test)
xgb_model, xgb_pred = xgboostClassifier(X_train, y_train, X_test, y_test)

#Plot results/cm or whatever



#Create robust evaluation metrics and plot these

def modelEvaluation(actual, pred):
    actual = [code_lookup[code] for code in actual]
    pred = [code_lookup[code] for code in pred]

    accuracy = accuracy_score(actual, pred)
    balanced_accuracy = balanced_accuracy_score(actual,pred)
    precision = precision_score(actual, pred, average='macro')
        
    print("Accuracy: %.2f%%" % (accuracy * 100.0),
          "Balanced Accuracy: %.2f%%" % (balanced_accuracy * 100.0),
          "Precision: %.2f%%" % (precision * 100.0),
          sep='\n')
    
    print(classification_report(actual, pred))
    
    print(confusion_matrix(actual, pred))
            
modelEvaluation(y_test, rf_pred)
modelEvaluation(y_test, xgb_pred)

#Compare agaist a dummy classifier

#Create API to query incoming stream data7

#Documentation
pfxz, x0, y0, z0, vx0, vy0, vz0, ax0, ay0, az0
pfxz — The vertical (up-down) movement of the pitch during the last 40 feet before the front of home plate, as compared to a theoretical pitch thrown at the same speed with no spin-induced movement.
x0 — The horizontal (left-right) location of the pitch 50 feet before the front of home plate. Positive numbers are toward the 1B side, while negative numbers are toward the 3B side, relative to a straight line drawn from the tip of home plate to the center of the rubber.
y0 — The distance from home plate, along a straight line drawn from the tip of home plate to the center of the rubber, 50 feet before the front of home plate. By definition this will always be 50 feet.
z0 — The height of the pitch, relative to home plate, 50 feet before the front of home plate.
vx0 — The velocity (speed) of the pitch in the left-right direction, 50 feet before the front of home plate.
vy0 — The velocity (speed) of the pitch in the direction toward home plate, 50 feet before the front of home plate.
vz0 — The velocity (speed) of the pitch in the up-down direction, 50 feet before the front of home plate.
ax0 — The acceleration (how much speed is changing) of the pitch in the left-right direction, 50 feet before the front of home plate.
ay0 — The acceleration (how much speed is changing) of the pitch in the direction toward home plate, 50 feet before the front of home plate.
az0 — The acceleration (how much speed is changing) of the pitch in the up-down direction, 50 feet before the front of home plate.
