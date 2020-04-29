''' Programar para prever notas de matemática do banco de dados teste utilizando banco de dados de treino'''

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC  
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVR
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing



#Leitura do banco de dados
db_test = pd.read_csv("D:\\Codenation\\Data_Science\\Desafio\\test.csv")
db_train = pd.read_csv("D:\\Codenation\\Data_Science\\Desafio\\train.csv")

#Verificação se banco de dados de teste tem as mesmas colunas que o bando de treino exceto notas de matemática 
print(set(db_test.columns).issubset(set(db_train.columns)))

#Selecionar somente dados numericos dos bancos de dados
#db_train = db_train.select_dtypes(include=['int64', 'float64'])

#Seleciona possiveis dados significativo
'''
NU_IDADE        =   Idade       
NU_NOTA_REDACAO =   Nota da prova de Redação
NU_NOTA_CN	    =   Nota da prova de Ciências da Natureza
NU_NOTA_CH	    =   Nota da prova de Ciências Humanas
NU_NOTA_LC	    =   Nota da prova de Linguagens e Códigos
NU_NOTA_MT	    =   Nota da prova de Matemática
'''
a = db_train[['NU_IDADE','NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_MT','NU_NOTA_REDACAO']]

#Correlação das notas e idade em relação a prova de matemática
corr_a = a.corr(method ='pearson')

#Idade não importa tanto para a nota de matemática, então foi retirado da lista
features = ['NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_MT','NU_NOTA_REDACAO']
a=a[features]

#substitui 0 por notas nulas
a = a.replace(0, np.nan)
#Retira registros no qual tem todas as notas nulas
a1 = a.dropna(axis = 0, how ='all') 
#Retira registros no qual tem somente a nota de matemática nula
a1 = a1.dropna(axis = 0, subset=['NU_NOTA_MT'])

#Substituir valores 0 pela media da linha
med = a1.mean(axis=1)
for i, col in enumerate(a1):
    a1.iloc[:, i] = a1.iloc[:, i].fillna(med)

#Substituir valores nulos pela media da coluna
#a1.fillna(a1.mean(),inplace=True)

#verificar se existe algum valor nulo
a1.isnull().sum()

y = a1['NU_NOTA_MT']
train = ['NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO']
x = a1[train]

scaler = preprocessing.StandardScaler().fit(x)
x_scaler = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, test_size = 0.15)  

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
r2_score(y_test, y_pred)

b = db_test[['NU_INSCRICAO','NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO']]
b = b.replace(0, np.nan)
#Retira registros no qual tem todas as notas nulas
b1 = b.dropna(axis = 0, how ='all', subset=['NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO'])

# Sub valores nulos pela media da linha
medb = b1.mean(axis=1)
for i, col in enumerate(b1):
    b1.iloc[:, i] = b1.iloc[:, i].fillna(medb)

b1.isnull().sum()

n = b1[['NU_INSCRICAO']]
b1 = b1.drop(['NU_INSCRICAO'], axis=1)

scalerb = preprocessing.StandardScaler().fit(b1)
x_scalerb = scaler.transform(b1)

y_pred_test = lr.predict(x_scalerb)
nota = np.column_stack((n, y_pred_test))

notas = pd.DataFrame({'NU_INSCRICAO': nota[:, 0], 'NU_NOTA_MT': nota[:, 1]})
notas = notas.sort_values(by=['NU_NOTA_MT'], ascending=False)
notas.to_csv(r'D:\\Codenation\\Data_Science\\Desafio\\answer.csv', index=False, header=True)