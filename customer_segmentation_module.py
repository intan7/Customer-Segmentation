# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 09:27:52 2022

@author: intan
"""
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization

from tensorflow.keras import Sequential,Input
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import numpy as np
import pandas as pd


class EDA:
    def plot_con_graph(self,con_col,df):
       #continuous
        for i in con_col:
            plt.figure()
            sns.distplot(df[i])
            plt.show()

    def plot_cat_graph(self,cat_col,df):
        #categorical
        for i in cat_col:
          plt.figure()
          sns.countplot(df[i])
          plt.show()

    def cramers_corrected_stat(self,confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
          
class ModelDevelopment:
    def simple_MD_model(self,X_shape,nb_class,nb_node=128, dropout_rate=0.3):
        
        model=Sequential()
        model.add(Input(shape=X_shape)) #9
        model.add(Dense(nb_node,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_node,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_node,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_class,activation='softmax'))
        model.summary()
        
        return model
    
class ModelEvaluation:
    def Plot_Hist(self,hist,loss=0,vloss=2):
        a=list(hist.history.keys())
        plt.figure()
        plt.plot(hist.history[a[loss]])
        plt.plot(hist.history[a[vloss]])
        plt.legend(['training_'+ str(a[loss]), a[vloss]])
        plt.show()
