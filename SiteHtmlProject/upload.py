import webbrowser
# -*-coding:Latin-1 -*
# -*- coding: utf-8 -*-
# -*- coding: iso-8859-1 -*-
# -*- coding: utf-8 -*-

import itertools
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
import dominate
from dominate.tags import *
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np 
import sys
from sklearn.metrics import roc_curve
reload(sys)
sys.setdefaultencoding('utf-8')
url =  '/Users/christian/Downloads/HR_DataSet.csv'
names = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours','time_spend_company','Work_accident','left','promotion_last_5years','sales','salary']
dataset = pandas.read_csv(url)


#Object to Numeric Data 

dataset.loc[dataset.salary =='low','salary'] = 0
dataset.loc[dataset.salary =='medium','salary'] = 1
dataset.loc[dataset.salary =='high','salary'] = 2

#Bivariate analysis
sns.boxplot(x='number_project' ,y='last_evaluation', data=dataset, palette='GnBu_d').set_title("number_project Vs last_evaluation")
plt.savefig('/Users/christian/Desktop/HTLM /lastNum.jpg')
plt.close()
sns.countplot(x='left' ,hue='sales', data=dataset, palette='GnBu_d').set_title("Left Vs Sales")
plt.savefig('/Users/christian/Desktop/HTLM /wooker.jpg')
plt.close()
sns.countplot(x='salary' ,hue='left', data=dataset, palette='GnBu_d').set_title("Left Vs Salary")
plt.savefig('/Users/christian/Desktop/HTLM /Salaty.jpg')
plt.close()
sns.boxplot(x='number_project' ,y='average_montly_hours', data=dataset, palette='GnBu_d').set_title("number_project Vs average_montly_hours")
plt.savefig('/Users/christian/Desktop/HTLM /numb.jpg')
plt.close()
dataset.plot(kind='box',y='time_spend_company')
plt.savefig('/Users/christian/Desktop/HTLM /Outliers.jpg')
plt.close()

#Matrix Correlation
matrice=dataset.corr()
sns.heatmap(matrice)
plt.savefig('/Users/christian/Desktop/HTLM /Heats.jpg')
plt.close()


#Object to Numeric Data 
dataset.loc[dataset.sales =='sales','sales'] = 0
dataset.loc[dataset.sales =='accounting','sales'] = 1
dataset.loc[dataset.sales =='hr','sales'] = 2
dataset.loc[dataset.sales =='technical','sales'] = 3
dataset.loc[dataset.sales =='support','sales'] = 4
dataset.loc[dataset.sales =='management','sales'] = 5
dataset.loc[dataset.sales =='IT','sales'] = 6
dataset.loc[dataset.sales =='product_mng','sales'] = 7
dataset.loc[dataset.sales =='marketing','sales'] = 8
dataset.loc[dataset.sales =='RandD','sales'] = 9

#Features Importance  

data = dataset.select_dtypes(include=[np.number]).interpolate().dropna()

y = dataset.left
X = dataset.drop(["left"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=.3, train_size = 0.7)
clf1 = RandomForestRegressor(n_jobs=2, n_estimators=1000)
model = clf1.fit(X_train, y_train)

import pandas as pd
import seaborn as sns
sns.set(color_codes=True)

headers = ["name", "score"]
values = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1] * -1)
print(values, headers)
df = pd.DataFrame(values)
plt.scatter(df[0], df[1])
plt.title('Importance des features')
locs, labels = plt.xticks() 
plt.setp(labels, rotation=90)
plt.savefig('/Users/christian/Desktop/HTLM /Corr.jpg')
plt.close()




from sklearn.linear_model import LogisticRegression


"""
LOG
"""
y = dataset.left
X = dataset.drop(["left","sales","average_montly_hours","Work_accident","promotion_last_5years","salary"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=.3, train_size = 0.7)

logreg= LogisticRegression(random_state = 0)
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print('Resultat Logistic Regression : ')
a=logreg.score(X_test, y_test)

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
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Stay', 'Left'],
                      title='Confusion matrix, without normalization')

plt.savefig('/Users/christian/Desktop/HTLM /LogMatt.jpg')
plt.close()
plt.close()

 

fpr_cl = dict()
tpr_cl = dict()

y_pred = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)

fpr_cl["classe 0"], tpr_cl["classe 0"], _ = roc_curve(
    y_test == 0, y_proba[:, 0].ravel())
fpr_cl["classe 1"], tpr_cl["classe 1"], _ = roc_curve(
    y_test, y_proba[:, 1].ravel())  # y_test == 1

import numpy
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
plt.savefig('/Users/christian/Desktop/HTLM /ROCLOG.jpg')
plt.close()

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
"""
on determine le nombre de neighbor 
"""
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

fpr_cl = dict()
tpr_cl = dict()

y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)

fpr_cl["classe 0"], tpr_cl["classe 0"], _ = roc_curve(
    y_test == 0, y_proba[:, 0].ravel())
fpr_cl["classe 1"], tpr_cl["classe 1"], _ = roc_curve(
    y_test, y_proba[:, 1].ravel())  # y_test == 1

import numpy
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
plt.savefig('/Users/christian/Desktop/HTLM /ROCKNN.jpg')
plt.close()
 

print('Resultat k-NN : ')
b=knn.score(X_test, y_test)


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

plt.savefig('/Users/christian/Desktop/HTLM /KnMat.jpg')
plt.close()
 


"""
RF
"""
y = dataset.left
X = dataset.drop(["left","sales","Work_accident","promotion_last_5years"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=.3, train_size = 0.7)
y_train=y_train.ravel()
y_test=y_test.ravel()
rclf = RandomForestClassifier()
rclf.fit(X_train,y_train)
y_pred = rclf.predict(X_test)
print('Resultat Random Forest : ')
d=rclf.score(X_test,y_test)
print(accuracy_score(y_test,y_pred))
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
cnf_matrixrf = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrixrf, classes=['Stay', 'Left'],
                      title='Confusion matrix, without normalization')


plt.savefig('/Users/christian/Desktop/HTLM /RfMat.jpg')
plt.close()
 

fpr_cl = dict()
tpr_cl = dict()

y_pred = rclf.predict(X_test)
y_proba = rclf.predict_proba(X_test)

fpr_cl["classe 0"], tpr_cl["classe 0"], _ = roc_curve(
    y_test == 0, y_proba[:, 0].ravel())
fpr_cl["classe 1"], tpr_cl["classe 1"], _ = roc_curve(
    y_test, y_proba[:, 1].ravel())  # y_test == 1

import numpy
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

plt.savefig('/Users/christian/Desktop/HTLM /ROCRF.jpg')
plt.close()

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
bj =clf1.score(X_test, y_test)
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


plt.savefig('/Users/christian/Desktop/HTLM /SvmMatt.jpg')
plt.close()
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

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
plt.savefig('/Users/christian/Desktop/HTLM /ROCSVM.jpg')
plt.close()
for ax in fig.axes:
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")


f = open('projet2.html','w')

message = html()
with message.add(body()):
    link(rel='stylesheet', href='project.css')    
with message.add(body()).add(div(id='bloc ')):
    with message.add(body()).add(header(id='bloc-page')):
          h1('Machine Learning-Gestion Predictive')
          h5('Launay - Chandelier - Chantharath')
          p('Ce site Html nous servira de couverture pour permettre la visualisation du travail accomplit durant ces 3 derniers mois')
   
with message.add(body()).add(div(id='analysedonne')):       
    h1('Exploration des donnees')        
    p('''Un moyen rapide d'avoir une idee de la distribution de chaque attribut est de regarder les histogrammes.''')
    p(''' Les histogrammes regroupent les donnees dans des groupes et nous fournissent le nombre d'observations dans chaque groupe. A partir de la forme des bacs,on peux rapidement observer les minimums ,maximums ect.. Cela peut egalement nous aider a voir les valeurs aberrantes . .''')
    img(src='Distrib.jpg',id='logo',style="width:600px;height:600px;") 
    p('''pour comprendre notre dataset et prendre les meilleurs decision il est necessaire de realiser des analyse univarie et bivarie de notre distribution ''')
    p('''Nous pourront par exemple observe la distribution des secteur depuis lesquelles les employes ont quitte l'entreprise , ou encore le nombre de projets qui ont pu leurs etres affecte''')
    img(src='lastNum.jpg',id='logo',style="width:300px;height:200px;") 
    img(src='wooker.jpg',id='logo',style="width:300px;height:200px;")
    img(src='numb.jpg',id='logo',style="width:300px;height:200px;")
    img(src='Salaty.jpg',id='logo',style="width:300px;height:200px;")
    
with message.add(body()).add(div(id='variable')):       
    h1('Classement et choix des attributs')        
    p('''pour effectuer notre choix de variables nous allons observer les correlations relier a notre variables de decision .Pour ce faire nous devons tout d abord determiner quelles sont les features qui affectent le plus notre variable de decision left Nous allons donc ici realiser un Heatmap''')
    img(src='Heats.jpg',id='logo',style="width:500px;height:400px;")
    p('''Il n'y a aucune valeur predictive fortement correlees superieure a 0.8 qu'on aurait donc du supprimer. ''')
  
with message.add(body()).add(div(id='variable')):       
    p(''' nous devons tout d'abords ici determiner quelles sont les attributs qui affectent le plus notre variable de decision LEFT''')
    p('''Voici les attributs les plus susceptible de faire varier notre variable cible.''')
    p('''Le  niveau de satisfaction de l'employe , son nombre d'accident du travail et son salaire sont les 3 facteurs les plus importants. Dans un ordre d'importance nous avons :''')
    h = ul()
with h:
    li('''' niveau de satisfaction avec un coefficient de :-0,441730''')
    li(''' salaire avec un coefficient de :-0,161970 ''')
   
    li('''accident du travail avec un coefficient : -0,158292 ''')
with message.add(body()).add(div(id='attribut')): 
    p('''Avec ces informations nous avons un bon point de depart pour pouvoir choisir nos variables decisionels mais ce n'est pas suffisant''' )
    p('''nous allons donc maintenant utiliser la regression lineaire avec random forest pour avoir un autre outils a notre disposition nous permettant de realiser une comparaison  ''')
    img(src='Corr.jpg',id='logo',style="width:500px;height:400px;")
    p('''A partir de la regression lineaire nous obtenons que les features les plus important sont :''')
    p(''' "satisfaction_lvl","time_spend_company" et "last_evaluation" suivi de tres pres par "number_project" et "average_montly_hours"''')
    p('''il nous faudra donc ici faire une synthese entre la regression lineaire et la correlation pour choisir nos features, mais seul l'etude finale nous permettera de determiner lesquelles etaient reelement les facteurs decisif''')
with message.add(body()).add(div(id='algo')):       
    h1('Algorithme et performances')    
    h2('Mise en place et selection du model')
    p('''On ne sait pas quelles algorithmes vont etre efficace pour le problemes que nous traitons ou quelles configuration nous allons devoir utiliser .''')
    p('''Nous avons tout de meme une idee grace aux plot que nous avons realiser precedement et du aux observation que certaine classe sont lineairement separable , donc nous attendons de bons resultats .''')
    p('''Nous allons evaluer 4 algorithmes:''') 
    h = ul()
with h:
    li(''' SVM''')
    li('''Logistique Regression ''')
    li('''KNN''') 
    li('''Random Forest''')
with message.add(body()).add(div(id='algo')):   
    p(''' on va affecter aux different algortithme le meme nombre de donnees dans la partie test et la partie prediction pour qu'ils soient directement comparables''')
    p('''nous allons donc maintennat mettre en place et evaluer nos differents modeles ''')
    p('''pour la regression logistique nous avons :''',a) 
    p('''pour KNN nous avons :''',b) 
    p('''pour Random Forest avons :''',d) 
    p('''pour SVM nous avons :''',bj) 

    p('''Nos resultats peuvent bien evidement changer en fonction des attribut que l'on fournit aux algorithmes''')
    p('''Neanmoins ici l'algorithme le plus efficaces est le Random Forest''')
with message.add(body()).add(div(id='algo')):   
    h2('Evaluation de nos modeles')   
    p('''l'algorithme du Random Forest est donc le model le plus precis , mais maintenant nous voulons avoir une idee de son eficacite sur notre jeu de validation , les 30% restant ''')
    p('''Cela nous donnera une derniere maniere d'evaluer nos modeles de machine learning. Les outils que nous allons utiliser vont nous permettre de quantifier le niveau de la prediction et donner une idee plus ou moins precise de la qualite du pattern obtenu.''')
    p('''Cela va nous servir a verifier notre jeu de validation et etre sur que nous n'avons pas fait d'erreurs durant l'entrainement de nos donnees comme par exemple donner trop de donnees a notre jeu d'entrainement ou encore peut etre perdu d'informations''')
    p('''pour observer cela nous utiliseront donc les outils d'evaluation des problemes de classification: les matrices de confusion et courbes ROC!''')
    h2('Indicateurs de performance en classification')
    h3('a)MATRICE DE CONFUSION')
    h4('Matrice de confusion SVM')
    img(src='SvmMatt.jpg',id='logo',style="width:300px;height:200px;")
    p('''par exemple ici ,parmis les ,''',cnf_matrixsvm[0,0 ]+ cnf_matrixsvm[0,1],  '''personne predites pour rester dans l'entreprise ''', cnf_matrixsvm[0,0],''' sont effectivement rester dans l'entreprise  ''')
    p(''' et sur les''',cnf_matrixsvm[1,1 ]+ cnf_matrixsvm[1,0],  '''  predites pour quitter l'entrepriser ''',cnf_matrixsvm[1,1],'''  ont effectivement quitter l'entreprise''')
    h4('Matrice de confusion Logistique Regression')
    img(src='LogMatt.jpg',id='logo',style="width:300px;height:200px;")
    p('''par exemple ici ,parmis les ,''',cnf_matrix[0,0 ]+ cnf_matrix[0,1],  '''personne predites pour rester dans l'entreprise ''', cnf_matrix[0,0],''' sont effectivement rester dans l'entreprise  ''')
    p(''' et sur les''',cnf_matrix[1,1 ]+ cnf_matrix[1,0],  '''  predites pour quitter l'entrepriser ''',cnf_matrix[1,1],'''  ont effectivement quitter l'entreprise''')
    h4('Matrice de confusion KNN')
    img(src='KnMat.jpg',id='logo',style="width:300px;height:200px;")
    p('''par exemple ici ,parmis les ,''',cnf_matrixknn[0,0 ]+ cnf_matrixknn[0,1],  '''personne predites pour rester dans l'entreprise ''', cnf_matrixknn[0,0],''' sont effectivement rester dans l'entreprise  ''')
    p(''' et sur les''',cnf_matrixknn[1,1 ]+ cnf_matrixknn[1,0],  '''  predites pour quitter l'entrepriser ''',cnf_matrixknn[1,1],'''  ont effectivement quitter l'entreprise''')
    h4('Matrice de confusion Random Forest')
    img(src='RfMat.jpg',id='logo',style="width:300px;height:200px;")
    p('''par exemple ici ,parmis les ,''',cnf_matrixrf[0,0 ]+ cnf_matrixrf[0,1],  '''personne predites pour rester dans l'entreprise ''', cnf_matrixrf[0,0],''' sont effectivement rester dans l'entreprise  ''')
    p(''' et sur les''',cnf_matrixrf[1,1 ]+ cnf_matrixrf[1,0],  '''  predites pour quitter l'entrepriser ''',cnf_matrixrf[1,1],'''  ont effectivement quitter l'entreprise''')
with message.add(body()).add(div(id='algo')):
    h3('b)COURBE ROC')
    p('''Dans le cas d'un classifieur binaire, il est possible de visualiser les performances du classifieur sur ce que l'on appelle une courbe ROC. La courbe ROC est une representation du taux de vrais positifs en fonction du taux de faux positifs. Son interet est de s'affranchir de la taille des donnees de test dans le cas ou les donnees sont desequilibrees.
      Cette representation met en avant un nouvel indicateur qui est l'aire sous la courbe. Plus elle se rapproche de 1, plus le classifieur est performant.''')
    h4('ROC SVM')
    img(src='ROCSVM.jpg',id='logo',style="width:300px;height:200px;")
    p('la librairie SVM na pas de fonction predict nous ne pouvons donc pas realiser de courbe ROC pour elle ')
    h4('ROC Logistque Regression')
    img(src='ROCLOG.jpg',id='logo',style="width:300px;height:200px;")
    h4('ROC KNN')
    img(src='ROCKNN.jpg',id='logo',style="width:300px;height:200px;")
    h4('ROC Random Forest')
    img(src='ROCRF.jpg',id='logo',style="width:300px;height:200px;")
with message.add(body()).add(div(id='conclu')):       
    h1('Conclusion')        
    p(' En mettant en relation toutes nos demarche , en commencant par la precision , puis les matrices de confusion et ROC , nous arrivons a la conclusion que le model le plus addapter et efficace pour resoudre notre probleme sera le Random Forest ')


  
f.write(str(message))
f.close()

filename = 'file:///Users/christian/Desktop/HTLM /' + 'projet2.html'
webbrowser.open_new_tab(filename)
