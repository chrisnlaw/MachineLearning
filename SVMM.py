#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 20:14:48 2018

@author: christian
"""

import itertools
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.metrics import roc_curve

url =  '/Users/christian/Downloads/HR_DataSet.csv'
names = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours','time_spend_company','Work_accident','left','promotion_last_5years','sales','salary']
dataset = pandas.read_csv(url)

dataset.loc[dataset.salary =='low','salary'] = 0
dataset.loc[dataset.salary =='medium','salary'] = 1
dataset.loc[dataset.salary =='high','salary'] = 2

"""
SVM
"""

y = dataset.left
X = dataset.drop(["left","sales","Work_accident","promotion_last_5years","salary"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=.3, train_size = 0.7)
from sklearn.svm import SVC
clf1 = SVC()
clf1.fit(X_train,y_train)
y_predd = clf1.predict(X_test)
print('Resultat SVM : ')
print(clf1.score(X_test, y_test))

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
cnf_matrixsvm = confusion_matrix(y_test, y_predd)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrixsvm, classes=['Stay', 'Left'],
                      title='Confusion matrix, without normalization')


plt.show()

#courbe ROC
from sklearn.metrics import auc

svm3 = SVC(C=1, kernel='rbf', gamma=1)
svm3.fit(X_train, y_train)

svm4 = SVC(C=1, kernel='rbf', gamma=50)
svm4.fit(X_train, y_train)

y_train_score3 = svm3.decision_function(X_train)
y_train_score4 = svm4.decision_function(X_train)

y_train_score3 = svm3.decision_function(X_train)
y_train_score4 = svm4.decision_function(X_train)


fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,6))
y_test_score3 = svm3.decision_function(X_test)
y_test_score4 = svm4.decision_function(X_test)

false_pos_rate3, true_pos_rate3, _ = roc_curve(y_test, y_test_score3)
roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

false_pos_rate4, true_pos_rate4, _ = roc_curve(y_test, y_test_score4)
roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

ax2.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
ax2.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
ax2.set_title('Test Data')
for ax in fig.axes:
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

plt.show()