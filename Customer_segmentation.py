# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:30:49 2022

@author: intan
"""

from customer_segmentation_module import ModelDevelopment,ModelEvaluation,EDA

from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer

from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import pickle
import os

#%%
CSV_PATH=os.path.join(os.getcwd(),'dataset','train.csv')
LOGS_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
BEST_MODEL_PATH = os.path.join(os.getcwd(),'model','best_model.h5')
#%%Step 1) Data loading

df=pd.read_csv(CSV_PATH)

#%%Step 2) Data inspection

df.info()
df.describe().T

cat_col=list(df.columns[df.dtypes=='object'])
cat_col.extend(['term_deposit_subscribed','day_of_month'])

con_col=df.drop(labels=cat_col,axis=1).columns
#%%plotting running slow

eda=EDA()
eda.plot_cat_graph(cat_col,df)
eda.plot_con_graph(con_col,df)

df.groupby(['term_deposit_subscribed','job_type']).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')
df.groupby(['term_deposit_subscribed','marital']).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')
df.groupby(['term_deposit_subscribed','education']).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')

#blue-collar,management,technician are are having trend of not subscribing
#those married and having secondary education are having trend of not subscribing
#%% Step 3) Data cleaning
for i in cat_col:
    if i=='term_deposit_subscribed':
        continue
    else:
        le=LabelEncoder()
        temp=df[i]
        temp[temp.notnull()]=le.fit_transform(temp[df[i].notnull()])
        df[i]=pd.to_numeric(df[i],errors='coerce')
        PICKLE_SAVE_PATH=os.path.join(os.getcwd(),'model',i+'_encoder.pkl')
        with open(PICKLE_SAVE_PATH,'wb') as file:
            pickle.dump(le,file)

df.info()
df.isna().sum()

#dropping days_since_prev_campaign_contact since too many NaN and ID since it's insignificance for this
df=df.drop(labels=['id','days_since_prev_campaign_contact'],axis=1)

df_col=df.columns

knn_im=KNNImputer()
df=knn_im.fit_transform(df)
df=pd.DataFrame(df)
df.columns=df_col

df.isna().sum() #no longer contain NaNs

df['customer_age']=np.floor(df['customer_age']).astype(int)
df['housing_loan']=np.floor(df['housing_loan']).astype(int)

df.duplicated().sum() #no duplicated data

#%%Ste 4) Features selection

X=df.drop(labels='term_deposit_subscribed',axis=1)
y=df['term_deposit_subscribed'].astype(int)

con_col=['customer_age','balance','last_contact_duration',
         'num_contacts_in_campaign','num_contacts_prev_campaign']

cat_dol=list(df.drop(labels=con_col,axis=1))
cat_col.remove('id')

#cont vs cat
selected_features=[]

for i in con_col:
    lr=LogisticRegression()
    lr.fit(np.expand_dims(X[i],axis=-1),y)
    print(i)
    print(lr.score(np.expand_dims(X[i],axis=-1),y))
    if lr.score(np.expand_dims(X[i],axis=-1),y)>0.8:
        selected_features.append(i)

print(selected_features)

#cat vs cat

for i in cat_col:
    print(i)
    matrix=pd.crosstab(df[i],y).to_numpy()
    print(eda.cramers_corrected_stat(matrix))
    if eda.cramers_corrected_stat(matrix)> 0.3:
        selected_features.append(i)
 
print(selected_features)
    
#from above analysis only customer_age,balance,last_contact_duration,
#num_contacts_in_campaign,num_contacts_prev_campaign,last_contact_duration,
#prev_campaign_outcome are higly correlated to term_deposit_subscribed 
    
df=df.loc[:,selected_features]
X=df.drop(labels='term_deposit_subscribed',axis=1)
y=df['term_deposit_subscribed'].astype(int)

#%%Step 5) Model Preprocessing

mms=MinMaxScaler()
X=mms.fit_transform(X)
MMS_PATH=os.path.join(os.getcwd(),'model','mms.pkl')
with open(MMS_PATH,'wb') as file:
    pickle.dump(mms,file)

#OHE
ohe=OneHotEncoder(sparse=False)
y=ohe.fit_transform(np.expand_dims(y,axis=-1))

X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.3,
                                               random_state=123)

#%% Model Development

shape_x=np.shape(X_train)[1:]
output=len(np.unique(y,axis=0))

md=ModelDevelopment()
model=md.simple_MD_model(shape_x,output,nb_node=128,dropout_rate=0.3)

plot_model(model,show_shapes=(True))

#%% Model Training

tensorboard_callback=TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)
early_callback=EarlyStopping(monitor='val_loss',patience=3)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.save(BEST_MODEL_PATH)

hist = model.fit(X_train, y_train, 
                 epochs=20,
                 validation_data=(X_test, y_test),
                 callbacks=[tensorboard_callback,early_callback])
#%% Model Evaluation

print(hist.history.keys())

me=ModelEvaluation()
me.Plot_Hist(hist,0,2) #to look for loss & val_loss

me=ModelEvaluation()
me.Plot_Hist(hist,1,3) #to look for acc & val_acc

y_pred = np.argmax(model.predict(X_test), axis=1)
y_test = np.argmax(y_test, axis=1)
print(classification_report(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred)

labels=['not subscribed','subscribed']
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()