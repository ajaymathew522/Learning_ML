import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('mlr02.xls')
X = np.array(df)

plt.scatter(X[:,1], X[:,0])
plt.show()

plt.scatter(X[:,2], X[:,0])
plt.show()


df['ones'] = 1
Y = df['X1']
X = df[['X2', 'X3', 'ones']]
X2 = df[['X2', 'ones']]
X3 = df[['X3','ones']]

def get_r2(X,Y):
    w = np.linalg.solve(np.dot(X.T,X), np.dot(X.T,Y))
    Yhat = np.dot(X, w)
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1)/ d2.dot(d2)
    return r2
print('X2 only:'+str(get_r2(X2, Y)))
print('X3 only:'+str(get_r2(X3, Y)))
print('Both:'+str(get_r2(X, Y)))