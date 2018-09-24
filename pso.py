import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
f = pd.read_csv('MLL_train.csv')
y = f.iloc[:,f.shape[1]-1]

for i in range(0,f.shape[1]-1):
    X=f.iloc[:,i]
    X_test = np.array(X.iloc[0: 5]).reshape(-1,1)
    y_test = np.array(y.iloc[0: 5]).reshape(-1,1)
    X_train = np.array(X.drop(X.index[[0, 1, 2, 3, 4]])).reshape(-1,1)
    y_train = np.array(y.drop(y.index[[0, 1, 2, 3, 4]])).reshape(-1,1)
    m=0
    for k in range(3,20):
        b = 0
        for j in range(1,k):
            knn = KNeighborsClassifier(n_neighbors = j)
            knn.fit(X_train, y_train)
            b+=knn.score(X_test, y_test)
        if m<b/k:
            m=b/k
            kopt=k
    if m>=0.9:
        print(i,kopt,m)