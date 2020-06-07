# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 08:34:42 2020

@author: mohdf
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle  as pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utilities import shuffle_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC

#from flask import Flask,template_rendered,request
def main():
    
    pick  =  open("C:/Users/mohdf/OneDrive/Desktop/CardioVascularDisease/xgb_classifier.pkl",'rb')
    classifier =  pkl.load(pick)
    bc =  pd.read_csv('C:/Users/mohdf/OneDrive/Desktop/CardioVascularDisease/cardio_train.csv')
    print(bc.head())
    s =  pd.Series(bc['cardio'])
    df,s = shuffle_data(bc,s)
    print(df,s)             
    print("First 12 columns of dataset:\n", df[df.columns[range(12)]].head(), "\n\nTargets of the dataset:\n", s.head(), end="\n\n")
    seed =7
    
    test_size =  0.3
    model = 'XGB'
    filename =  'xgb_classifier.pkl'
    X_test, Y_test=   split_and_save(test_size, model, filename, seed, df, s)
    
    #load model and predict
    result =  classifier.score(X_test, Y_test)
    y_true = Y_test
    y_pred = classifier.predict(X_test)
    print(type(result))
    print("Accuracy is :" +str((result*100)))
    df2 = confusion_matrix_func(y_true,y_pred)
    print(df2)
    report = classification_report(Y_test, y_pred)
	#calculates precision | recall | f-1 score | support for each class.
	#print("Classification Report:"+ str(report))
    print(report)
    
    print("No Cardio Disease" if(y_pred[0] == 0) else "Having Cardio Disease")
   
    
    
def confusion_matrix_func(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    show_matrix(cm)
    TN = cm[0,0]
    TP = cm[1,1]
    FN = cm[1,0]
    FP = cm[0,1]
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1_Score = 2*(Recall * Precision) / (Recall + Precision)
    df2=pd.DataFrame([[Precision, Recall, F1_Score]],columns=["Precision", "Recall", "F1 Score"], index=["Results"])
    return df2

def show_matrix(cm):
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,fmt=".0f", annot=True,linewidths=0.2, linecolor="purple", ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Grand Truth")
    plt.show()
        
def split_and_save(test_size, model, filename, seed, df, s):
    
	#Create model dictionary
	models = {"LOR": LogisticRegression(), 
			  "SVC": SVC(), 
              "RF" : RandomForestClassifier(n_estimators =100)
              "XGB" : XGBClassifier(n_estimators=100)
			 }

	#Split data
	X_train, X_test, Y_train, Y_test = train_test_split(df, s, test_size=test_size, random_state=seed)

	#Select and train model
	filename = filename
	model = models[model]
	model.fit(X_train, Y_train)
	#Save model
	pkl.dump(model, open(filename, 'wb'))
	print("Model saved to file succesfully!", end="\n\n")
	return X_test, Y_test



if __name__ == "__main__":
	main()
