# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:11:10 2022

@author: sayan
"""

#Importing the Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#importing the dataset
dataset = pd.read_csv('breast cancer kaggle.csv')


#Dealing with missing data
dataset.isnull().values.any()   #True

dataset.isnull().values.sum()   #569

dataset.columns[dataset.isnull().any()]
len(dataset.columns[dataset.isnull().any()])

dataset['Unnamed: 32'].count()          #no values here

dataset = dataset.drop(columns ='Unnamed: 32')

dataset.shape


dataset.isnull().values.any()       #checking again - False

#Dealing with Categorical data
dataset.select_dtypes(include = 'object').columns
dataset['diagnosis'].unique()
dataset['diagnosis'].nunique()


#OneHotEncoding
dataset = pd.get_dummies(dataset, drop_first= True)

#dataset.head()

#Countplot
sns.countplot(dataset['diagnosis_M'], label = 'Count')

#how many Benign(B) values
(dataset.diagnosis_M == 0).sum()            #357

#how many Malignant(M) values
(dataset.diagnosis_M == 1).sum()            #212

#Correlation matrix and heatmap
dataset_2 = dataset.drop(columns = 'diagnosis_M')
dataset_2.corrwith(dataset['diagnosis_M']).plot.bar(
    figsize = (20,10), title = 'Correlated with Diagnosis_M', rot = 45, grid = True
    )


#correlation matrix
corr = dataset.corr()

#heatmap
plt.figure(figsize = (20,10))
sns.heatmap(corr,annot = True)



#Spliting the dataset into training and test set
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)





'''

BUILDING THE MODEl

'''
#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state=0)

classifier_lr.fit(x_train,y_train)

y_pred_lr = classifier_lr.predict(x_test)      #prediction

#Scoring the prediction for Logistic Regression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
rec_lr = recall_score(y_test, y_pred_lr)


#Results
results_lr = pd.DataFrame([['Logistic Regression',acc_lr,f1_lr,prec_lr,rec_lr]],
                       columns = ['Model', 'Accuracy_Score', 'F1_Score', 'Precision_Score', 'Recall_Score'])



#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier_rm = RandomForestClassifier(random_state=0)

classifier_rm.fit(x_train,y_train)

y_pred_rm = classifier_rm.predict(x_test)      #prediction


#Scoring the prediction for Random Forest Regression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
acc_rm = accuracy_score(y_test, y_pred_rm)
f1_rm = f1_score(y_test, y_pred_rm)
prec_rm = precision_score(y_test, y_pred_rm)
rec_rm = recall_score(y_test, y_pred_rm)



#Result
results_rm = pd.DataFrame([['Random Forest Regression',acc_rm,f1_rm,prec_rm,rec_rm]],
                       columns = ['Model', 'Accuracy_Score', 'F1_Score', 'Precision_Score', 'Recall_Score'])




results = results_lr.append(results_rm, ignore_index=True)


print(results)


cm_lr = confusion_matrix(y_test, y_pred_lr)         #logistic Reg
cm_rm = confusion_matrix(y_test, y_pred_rm)         #random forest reg

print(cm_lr)        #logistic Reg
print(cm_rm)        #random forest reg

#Cross val Score of Logistic Regression
from sklearn.model_selection import cross_val_score
accuracies_lr = cross_val_score(estimator= classifier_lr, X=x_train, y=y_train, cv = 10)
print('Accuracy is {:.2f}%'.format(accuracies_lr.mean()*100))
print('Standard Deviation is {:.2f}%'.format(accuracies_lr.std()*100))


#Cross val Score of Random Forest Regression
from sklearn.model_selection import cross_val_score
accuracies_rm = cross_val_score(estimator= classifier_rm, X=x_train, y=y_train, cv = 10)
print('Accuracy is {:.2f}%'.format(accuracies_rm.mean()*100))
print('Standard Deviation is {:.2f}%'.format(accuracies_rm.std()*100))



#Randomized parameters to find the best parameters
from sklearn.model_selection import RandomizedSearchCV
parameters = {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
              'C' : [0.25, 0.05, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']            
              
              }

random_search = RandomizedSearchCV(estimator= classifier_lr , param_distributions = parameters, n_iter = 5, 
                                   scoring = 'roc_auc', n_jobs = -1, cv =10, verbose= 3)

random_search.fit(x_train,y_train)

random_search.best_score_

random_search.best_params_

random_search.best_estimator_



#Final model : Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=2.0, random_state=0, solver='liblinear')  #need to run more times

classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
acc_rm = accuracy_score(y_test, y_pred_rm)
f1_rm = f1_score(y_test, y_pred_rm)
prec_rm = precision_score(y_test, y_pred_rm)
rec_rm = recall_score(y_test, y_pred_rm)

results2 = pd.DataFrame([['Logistic Regression',acc_lr,f1_lr,prec_lr,rec_lr]],
                       columns = ['Model', 'Accuracy_Score', 'F1_Score', 'Precision_Score', 'Recall_Score'])

results = results_lr.append(results2, ignore_index=True)

#Cross val Score of Logistic Regression
from sklearn.model_selection import cross_val_score
accuracies_lr = cross_val_score(estimator= classifier_lr, X=x_train, y=y_train, cv = 10)
print('Accuracy is {:.2f}%'.format(accuracies_lr.mean()*100))
print('Standard Deviation is {:.2f}%'.format(accuracies_lr.std()*100))






