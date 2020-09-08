# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 20:54:03 2020

@author: Tom
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Data Ingest
raw_pitch_data = pd.read_csv('pitches.csv', nrows = 10000)

#Data Exploration
raw_pitch_data.columns

raw_pitch_data.pitch_type.unique()

raw_pitch_data.pitch_type.value_counts().plot(kind ='bar')

corr = raw_pitch_data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

fig, axs = plt.subplots(nrows=3)
sns.boxplot(x='pitch_type', y='break_length', data=raw_pitch_data, ax=axs[0])
sns.boxplot(x='pitch_type', y='break_length', data=raw_pitch_data, ax=axs[1])
sns.boxplot(x='pitch_type', y='break_length', data=raw_pitch_data, ax=axs[2])

#Data Preparation
unwanted_pitch_types = ['PO', 'FO', 'EP']
unwanted_pitches = raw_pitch_data['pitch_type'].isin(unwanted_pitch_types)
pitches = raw_pitch_data[-unwanted_pitches]

#Plot Distributions

#Plot correlations

#Plot a pitch?

#Create 1 or two classifications models

#Plot results/cm or whatever

#Create robust evaluation metrics and plot these

#Create API to query incoming stream data
