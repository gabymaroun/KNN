## k-voisins.py
## Introduction à Python
## Squelette du code recherche des k plus proches voisins
## (c) Eric Gouardères, novembre 2019
## Gaby Maroun et Brunel Ebata

import numpy as np
import matplotlib.pyplot as plt
import random as rd
from math import sqrt
import sys
from scipy.spatial import distance
import timeit 
from numba import jit
import numba
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances

iris = datasets.load_iris()
def sklearn_to_df(sklearn_dataset):
# Création dataFrame, colonnes caractéristiques (valeurs et étiquettes)
    df = pd.DataFrame(data=sklearn_dataset.data, columns=sklearn_dataset.feature_names)
# Ajout colonne variable à estimer
    df['target'] = sklearn_dataset.target
    return df
# Chargement du jeu de données et création d'une structure DataFrame Pandas
df = sklearn_to_df(iris)

"""Distance euclidienne entre deux points (espace à deux dimensions)

    Un point est représenté par un couple de valeurs flottantes.
    """
#4eme methode
def distMulti (a,b):
    somme=0
    for i in range(len(b)):
        diff=(a[i]-b[i])**2
        somme+=diff
    
    return sqrt(somme)
    
def affiche1(x, a, n , k_voisins, predic ):

    # Création des deux tableaux (axe x et y) pour affichage des points de a (tableaux numpy)
    x2=a[0]
    y2=a[1]

    # Création des deux tableaux (axe x et y) pour affichage des points de k_voisins (tableaux numpy)
    x1=[]
    y1=[]
    for i in k_voisins:
        x1.append(i[0])
        y1.append(i[1])
    
    # Initialisation des propriétés de l'affichage
    plt.scatter(x2[lab==0],y2[lab==0],c='g')
    plt.scatter(x2[lab==1],y2[lab==1],c='r')
    plt.scatter(x2[lab==2],y2[lab==2],c='b')
    #plt.scatter(x2,y2,s=25,c=couleur(x2,y2,x1,y1))
    plt.scatter(x1,y1,s=25,c='gray')
    plt.scatter(x[0],x[1],s=25,c='magenta')
    # Affichage
    
    #Affichage résultats
    txt="Résultat : "
    if predic==0:
      txt=txt+"setosa"
    if predic==1:
      txt=txt+"virginica"
    if predic==2:
      txt=txt+"versicolor"
    #plt.text(3,0.5, f"largeur : {round(x[0],2)} cm longueur : {round(x[1],2)} cm", fontsize=12)
    plt.text(3,0.3, f"k : {k}", fontsize=12)
    plt.text(3,0.1, txt, fontsize=12)
    #fin affichage résultats
    
    
    #plt.show()
def affiche2(x, a, k):
    
    x2=a[0]
    y2=a[1]
  
    #graphique
    plt.scatter(x2[lab==0],y2[lab==0], color='g')
    plt.scatter(x2[lab==1],y2[lab==1], color='r')
    plt.scatter(x2[lab==2],y2[lab==2], color='b')
    plt.scatter(x[0],x[1], color='k')
    #plt.legend()
    #fin graphique

    #algo knn
    d=list(zip(x2,y2))
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(d,lab)
    prediction= model.predict([[x[0],x[1]]])
    #fin algo knn

    #Affichage résultats
    txt="Résultat : "
    if prediction[0]==0:
      txt=txt+"setosa"
    if prediction[0]==1:
      txt=txt+"virginica"
    if prediction[0]==2:
      txt=txt+"versicolor"
    plt.text(3,0.5, f"largeur : {round(x[0],2)} cm longueur : {round(x[1],2)} cm", fontsize=12)
    plt.text(3,0.3, f"k : {k}", fontsize=12)
    plt.text(3,0.1, txt, fontsize=12)
    #fin affichage résultats

    #plt.show()
    
def cherche_k_voisins6(k, x, a): 

    A=list(zip(a[0],a[1]))
    dis=list(map(lambda l:distMulti(l,x),A))
    dicts=dict(zip(zip(range(0,len(dis)),lab.values,dis),A))
    proches_voisins=[]
    p=0
    for j in sorted(dicts.keys(), key = lambda x : x[2]):
        proches_voisins.append(dicts[j])
        p+=1
        if p==k:
            break

    seto=0
    virg=0
    virs=0
    r=list(dicts.keys())
    for i in proches_voisins:
        for j in dicts.values():
            if i==j :
                l=list(dicts.values())
                ind=l.index(j)
                if r[ind][1]==0:
                    seto+=1
                if r[ind][1]==1:
                    virg+=1
                if r[ind][1]==2:
                    virs+=1
                break
                
    if max(seto,virg,virs)==seto:
        predic=0
    elif max(seto,virg,virs)==virg:
        predic=1
    else:
        predic=2
    return(proches_voisins,predic)


# Programme principal

# Création des données
X = iris.data[:, :]
xy_train, xy_ttest, lab_train, lab_test = train_test_split(X, lab,stratify=lab, test_size=0.2, random_state=0)
x1=df.loc[:,"petal length (cm)"]
y1=df.loc[:,"petal width (cm)"]
lab=df.loc[:,"target"]
x5=(x1[19],y1[19])


n =len(iris.data)
a = (x1,y1)
x = (7*np.random.uniform()+1,2.5*np.random.uniform()+1)

k2=121 

#affichage des k plus proches voisins 
print('le point x(',round(x[0],3),',',round(x[1],3),')')
#affiche(x, a, n, k_voisins)
affiche1(x, a, n, k_voisins, predict)



#notre methode
def precision (k1, x, a):
    right=0
    fault=0
    score=0
    for i in range(k1):
        ckv7,predicti=cherche_k_voisins6(i, x, a)
        if predicti==lab[19]:
            right+=1
    score=(right/(k1-1))*100
    return score

%timeit s=precision(k2, x5, a)
print(precision(k2, x5, a))
#affiche1(x5, a, n, ckv7, predicti)


#methode scikit
X = iris.data[:, :]
xy_train, xy_test, lab_train, lab_test = train_test_split(X, lab,stratify=lab, test_size=0.2, random_state=0)


model = KNeighborsClassifier(n_neighbors=120)
model.fit(xy_train, lab_train)
prediction= model.predict(xy_test)


# Calcul précision (mean accuracy)
%timeit model.score(xy_test, lab_test)
model.score(xy_test, lab_test)*100
#print(accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))
