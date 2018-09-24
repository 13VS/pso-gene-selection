import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
f = pd.read_csv('data.csv')
X = f.iloc[:,:f.shape[1]-1]
y = f.iloc[:,f.shape[1]-1]
k_range = range(1,len(X.index))
scores = []
scores2 = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = 2)
    X_test = np.array(X.iloc[k - 1,:]).reshape(1,-1)
    y_test = np.array(y.iloc[k - 1]).reshape(1,-1)
    X_train=X.drop(X.index[k-1])
    y_train=y.drop(y.index[k-1])
    knn.fit(X_train, y_train)
    a = knn.score(X_test, y_test)
    if a == 1:
        scores.append(k - 1)
    sv = svm.SVC()
    sv.fit(X_train, y_train)
    a=sv.score(X_test, y_test)
    if a == 1:
        scores2.append(k - 1)
print("knn")
for s in scores:
    print(s,"- 100%")
print("svm")
for s in scores2:
    print(s,"- 100%")
