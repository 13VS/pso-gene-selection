import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
f = pd.read_csv('data.csv')
X = f.iloc[:,:f.shape[1]-1]
y = f.iloc[:,f.shape[1]-1]
k_range = range(1,int(len(X.index)/5))
scores = []
scores2 = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = 2)
    X_test = X.iloc[(k - 1)*5:k*5]
    y_test = y.iloc[(k - 1)*5:k*5]
    j=(k-1)*5
    X_train=X.drop(X.index[[j,j+1,j+2,j+3,j+4]])
    y_train=y.drop(y.index[[j,j+1,j+2,j+3,j+4]])
    knn.fit(X_train, y_train)
    a=knn.score(X_test, y_test)
    scores.append([k-1,a])
    sv = svm.SVC()
    sv.fit(X_train, y_train)
    a = sv.score(X_test, y_test)
    scores2.append([k-1,a])
scores=sorted(scores, key=lambda x: x[1], reverse=True)
scores2=sorted(scores2, key=lambda x: x[1], reverse=True)
print("knn")
print(scores)
print("svm")
print(scores2)
