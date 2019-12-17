#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 13:33:02 2019

@author: bnawan
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError

def feature(sentence):
    tfidfconverter = TfidfVectorizer(max_features=10000, min_df=5, max_df=0.7)
    fitur = tfidfconverter.fit_transform(sentence).toarray()
    return fitur

def RandomForest(x,y):
    text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)  
    return text_classifier.fit(x,y)  

def visualize_confusion_matrix(model,classes,X_train,y_train,X_test,y_test):
    # The ConfusionMatrix visualizer taxes a model
    cm = ConfusionMatrix(model, classes=classes)
    # Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
    cm.fit(X_train, y_train)
    # To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
    # and then creates the confusion_matrix from scikit-learn.
    cm.score(X_test, y_test)
    return cm.show()

def visualize_classification(model,classes,X_train,y_train,X_test,y_test):
    # Instantiate the classification model and visualizer
    visualizer = ClassificationReport(model, classes=classes, support=True)
    # Fit the training data to the visualizer
    visualizer.fit(X_train, y_train)
    # Evaluate the model on the test data
    visualizer.score(X_test, y_test)
    # Draw visualization
    return visualizer.show()

def visualize_pred_error(model,classes,X_train,y_train,X_test,y_test):
    # Instantiate the classification model and visualizer
    visualizer = ClassPredictionError(model, classes=classes)
    # Fit the training data to the visualizer
    visualizer.fit(X_train, y_train)
    # Evaluate the model on the test data
    visualizer.score(X_test, y_test)
    # Draw visualization
    return visualizer.show()

def main():
    data = pd.read_csv('output/prepo_output.csv',error_bad_lines=False)
    data.columns = ['date','username','tweet','hashtags','likes_count','sentiment']
    print(data.head(10))
 
    data['tweet'] = data['tweet'].fillna("")

    X = feature(data['tweet'])
    y = data.iloc[:, 5].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)    
    model = RandomForest(X_train, y_train)
    predictions = model.predict(X_test)

    # print(confusion_matrix(y_test,predictions))  
    # print(classification_report(y_test,predictions))  
    print(accuracy_score(y_test, predictions))

    visualize_confusion_matrix(model,["negative","neutral","positive"],X_train,y_train,X_test,y_test)
    visualize_classification(model,["negative","neutral","positive"],X_train,y_train,X_test,y_test)
    visualize_pred_error(model,["negative","neutral","positive"],X_train,y_train,X_test,y_test)


if __name__== "__main__":
    main()