#!/usr/bin/env python2
# -*- coding: utf-8 -*-
 


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


url =  '/Users/christian/Downloads/HR_DataSet.csv'
names = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours','time_spend_company','Work_accident','left','promotion_last_5years','sales','salary']
emp = pd.read_csv(url)

 
#on change les valeurs qui sont des strings en entier pour que les commandes les prennent en compte
emp.loc[emp.salary == 'low','salary'] = 0
emp.loc[emp.salary == 'medium','salary'] = 1
emp.loc[emp.salary == 'high','salary'] = 2

emp.loc[emp.sales == 'IT','sales'] = 1
emp.loc[emp.sales == 'RandD','sales'] = 2
emp.loc[emp.sales == 'accounting','sales'] = 3
emp.loc[emp.sales== 'hr','sales'] = 4
emp.loc[emp.sales == 'management','sales'] = 5
emp.loc[emp.sales == 'marketing','sales'] = 6
emp.loc[emp.sales == 'product_mng','sales'] = 7
emp.loc[emp.sales == 'sales','sales'] = 8
emp.loc[emp.sales == 'support','sales'] = 9
emp.loc[emp.sales == 'technical','sales'] = 10




#on selectionne notre dataset avec seulement les employés ayant démissionés
empquitte = emp.loc[emp['left']==1]

f1 = empquitte['satisfaction_level'].values
f2 = empquitte['last_evaluation'].values
f3 = empquitte['number_project'].values
f4 = empquitte['average_montly_hours'].values
f5 = empquitte['time_spend_company'].values
f6 = empquitte['Work_accident'].values
f7 = empquitte['left'].values
f8 = empquitte['promotion_last_5years'].values
f9 = empquitte['sales'].values
f10 = emp['salary'].values

#on crée un numpy array contenant nos données
X = np.array(list(zip(f1, f2,f3,f4,f5,f6,f7,f8,f9,f10)))

#On utilise K means avec 3 groupes
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


centers = kmeans.cluster_centers_

#j'ai fais des test en plotant les barycentres dans un premier temps

plt.scatter(centers[:, 3], centers[:, 2], c='black', s=200, alpha=0.5);
plt.title('nmbre projet X heures par mois')
plt.xlabel('Heure par mois')
plt.ylabel('Nombre de projet')
plt.show()
plt.clf()

plt.scatter(centers[:, 2], centers[:, 1], c='black', s=200, alpha=0.5);
plt.title('evaluation X nombre projet')
plt.xlabel('Nombre de projet')
plt.ylabel('Evaluation')
plt.show()
plt.clf()

plt.scatter(centers[:, 3], centers[:, 1], c='black', s=200, alpha=0.5);
plt.title('evaluation X heure mois')
plt.xlabel('Heure par mois')
plt.ylabel('Evaluation')
plt.show()
plt.clf()

#J'ai utilisé cette partie en commentaire pour faire des test des des 100 combinaisons possibles avec une boucle

"""
#on utilise une boucle pour repérer les combinaisons qui affichent un graphe avec des groupes distincts
indice = np.arange(0,10)
for i, k in enumerate(indice):
    print(i)
    plt.scatter(X[:, i], X[:, 6], c=y_kmeans, s=100, alpha=1);
#Pour observer toute les combinaisons il faut modifier le numéro dans le X[:,6] de 0 à 9    
    plt.show()
    plt.clf()
"""

#Je plot les graphes avec une répartition des clusters que je trouve interressantes

plt.scatter(X[:, 3], X[:, 1], c=y_kmeans, s=100, alpha=1);
plt.title('evaluation en fonction des heures par mois')
plt.xlabel('Heure par mois')
plt.ylabel('Evaluation')
plt.show()
plt.clf()

#PROBLEME pour celui là
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, alpha=1)
plt.title('evaluation en fonction de la satisfaction')
plt.xlabel('Satisfaction')
plt.ylabel('Evaluation')
plt.show()
plt.clf()
plt.scatter(X[:, 3], X[:, 0], c=y_kmeans, s=100, alpha=1);
plt.title('Satisfaction en fonctin des heure par mois')
plt.xlabel('Heure par mois')
plt.ylabel('Satisfaction')
plt.show()
plt.clf()

#Pas convaincu pour celui là
plt.scatter(X[:, 3], X[:, 2], c=y_kmeans, s=100, alpha=1);
plt.title('nombre projet en fonction des heures par mois')
plt.xlabel('Heure par mois')
plt.ylabel('Nombre de projet')
plt.show()
plt.clf()
