# -*- coding: utf-8 -*-
# préparation des données pour le clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as st
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import dendrogram

# Prépararion, centrage et réduction
def centrage_reduction(dataset):
    Xf = dataset.values 
    paysf = dataset.index

    std_scalef = preprocessing.StandardScaler().fit(Xf)
    X_crf = std_scalef.transform(Xf)

    Zf = linkage(X_crf, method='ward')
    
    return Xf, paysf, std_scalef, X_crf, Zf

def plot_dendogram(Zf,paysf,hauteur,index):
    plt.figure(figsize=(12,25), dpi=300)
    plt.title('Dendogramme de classification ascendante hiérarchique (CAH)')
    plt.xlabel('distance')
    plt.grid(False)
    dendrogram(
        Zf,
        labels = paysf,
        orientation = "right",
        color_threshold=hauteur
    )
    plt.savefig('exports/dendogram_CAH_'+str(index)+'.png')
    plt.show()
    
def clustering(dataset, Zf, nb_clust):
    clusters_cahf = fcluster(Zf, nb_clust, criterion='maxclust')
    
    #index triés des groupes
    idgf = np.argsort(clusters_cahf)
    
    #affichage des pays et leurs groupes
    df_groupage_f = pd.DataFrame(columns=["Cluster"+str(nb_clust),"Zone"])
    df_groupage_f["Zone"] = dataset.index[idgf]
    df_groupage_f["Cluster"+str(nb_clust)] = clusters_cahf[idgf]
    
    return df_groupage_f

# Calculs des Kmeans
def kmeans(n_clust, X_crf):
    kmf = KMeans(n_clusters=n_clust)
    kmf.fit(X_crf)
    clusters_kmf = kmf.labels_
    pcaf = decomposition.PCA().fit(X_crf)
    X_projectedf = pcaf.transform(X_crf)
    
    return kmf, clusters_kmf, pcaf, X_projectedf

# ACP - Eboulis des valeurs propres
def eboulis(pcaf):
    varexplf = pcaf.explained_variance_ratio_*100
    
    plt.figure(figsize=(12,8))
    plt.bar(np.arange(len(varexplf))+1, varexplf)
    plt.plot(np.arange(len(varexplf))+1, varexplf.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
    
    print(varexplf)
        




    