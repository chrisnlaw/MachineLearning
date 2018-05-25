#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 20:11:02 2018
@author: christian
"""

import itertools
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.metrics import roc_curve
import numpy


url =  '/Users/christian/Downloads/HR_DataSet.csv'
names = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours','time_spend_company','Work_accident','left','promotion_last_5years','sales','salary']
dataset = pandas.read_csv(url)

dataset.loc[dataset.salary =='low','salary'] = 0
dataset.loc[dataset.salary =='medium','salary'] = 1
dataset.loc[dataset.salary =='high','salary'] = 2

"""
KNN
"""
data = dataset.select_dtypes(include=[np.number]).interpolate().dropna()

y = dataset.left
X = dataset.drop(["left","sales","average_montly_hours","Work_accident","promotion_last_5years","salary"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=.3, train_size = 0.7)


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


#Courbe ROC

fpr_cl = dict()
tpr_cl = dict()

y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)

fpr_cl["classe 0"], tpr_cl["classe 0"], _ = roc_curve(
    y_test == 0, y_proba[:, 0].ravel())
fpr_cl["classe 1"], tpr_cl["classe 1"], _ = roc_curve(
    y_test, y_proba[:, 1].ravel())  # y_test == 1

prob_pred = numpy.array([y_proba[i, 1 if c else 0]
                         for i, c in enumerate(y_pred)])
fpr_cl["tout"], tpr_cl["tout"], _ = roc_curve(
    (y_pred == y_test).ravel(), prob_pred)

plt.figure()
for key in fpr_cl:
    plt.plot(fpr_cl[key], tpr_cl[key], label=key)

lw = 2
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Proportion mal classee")
plt.ylabel("Proportion bien classee")
plt.title('ROC(s) avec predict_proba')
plt.legend(loc="lower right")
plt.show()

print('Resultat k-NN : ')
print(knn.score(X_test, y_test))

#Confusion Matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrixknn = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrixknn, classes=['Stay', 'Left'],
                      title='Confusion matrix, without normalization')

plt.show()
 
