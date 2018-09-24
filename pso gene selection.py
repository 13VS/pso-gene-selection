import random
import copy    # array-copying convenience
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# ------------------------------------

def show_vector(vector):
  for i in range(len(vector)):
   print("%.4f" % vector[i], end=" ") # 4 decimals
  print("\n")

def error(position,f):
    err = 0.0
    T=pd.DataFrame()
    for i in range(len(position)):
        if(position[i]>0):
            a=f.iloc[i]
            T=T.append(a)

    y = T.iloc[:, T.shape[1] - 1]
    c=0
    m1=0
    for i in range(0, T.shape[1] - 1):
        X = T.iloc[:, i]
        k_range = range(1, int(len(X.index) / 5))
        m = 0
        k1 = 1
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=2)
            j = (k - 1) * 5
            X_test = np.array(X.iloc[j:j + 5]).reshape(-1, 1)
            y_test = np.array(y.iloc[j: j + 5]).reshape(-1, 1)
            X_train = np.array(X.drop(X.index[[j, j + 1, j + 2, j + 3, j + 4]])).reshape(-1, 1)
            y_train = np.array(y.drop(y.index[[j, j + 1, j + 2, j + 3, j + 4]])).reshape(-1, 1)
            knn.fit(X_train, y_train)
            a = knn.score(X_test, y_test)
            if a > m:
                m = a
                k1 = k
        j = (k1 - 1) * 5
        X_test = np.array(X.iloc[j:j+5]).reshape(-1, 1)
        y_test = np.array(y.iloc[j: j+5]).reshape(-1, 1)
        X_train = np.array(X.drop(X.index[[j, j + 1, j + 2, j + 3, j + 4]])).reshape(-1, 1)
        y_train = np.array(y.drop(y.index[[j, j + 1, j + 2, j + 3, j + 4]])).reshape(-1, 1)
        m = 0
        for k in range(3, min(X_train.shape[0],20)):
            b = 0
            for j in range(1, k):
                knn = KNeighborsClassifier(n_neighbors=j)
                knn.fit(X_train, y_train)
                b += knn.score(X_test, y_test)
            if m < b / k:
                m = b / k
        if m >= 0.9:
            c+=1
            print(i,end=" ")
            print("is a selected gene")
        if m>m1:
            m1=m
    print()
    if c!=0:
        return m1+1/c
    else:
        return -1
# ------------------------------------

class Particle:
  def __init__(self, dim, minx, maxx, seed,f):
    self.rnd = random.Random(seed)
    self.position = [0.0 for i in range(dim)]
    self.velocity = [0.0 for i in range(dim)]
    self.best_part_pos = [0.0 for i in range(dim)]
    for i in range(dim):
      self.position[i] = ((maxx - minx) * self.rnd.random() + minx)
      self.velocity[i] = ((maxx - minx) * self.rnd.random() + minx)
    print("Particle",end=" ")
    print(seed)
    self.error = error(self.position,f) # curr error
    self.best_part_pos = copy.copy(self.position)
    self.best_part_err = self.error # best error

def Solve(max_epochs, n, dim, minx, maxx,f):
  rnd = random.Random(0)
  swarm = [Particle(dim, minx, maxx, i,f) for i in range(n)]
  best_swarm_pos = [0.0 for i in range(dim)] # not necess.
  best_swarm_err = 0 # swarm best
  for i in range(n): # check each particle
    if swarm[i].error > best_swarm_err:
      best_swarm_err = swarm[i].error
      best_swarm_pos = copy.copy(swarm[i].position)
  epoch = 0
  w = 0.729    # inertia
  c1 = 1.49445 # cognitive (particle)
  c2 = 1.49445 # social (swarm)
  while epoch < max_epochs:
    print("Epoch = " + str(epoch) +
        " best error = %.3f" % best_swarm_err)
    for i in range(n): # process each particle
      for k in range(dim):
        r1 = rnd.random()    # randomizations
        r2 = rnd.random()
        swarm[i].velocity[k] = ( (w * swarm[i].velocity[k]) +
          (c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k])) +
          (c2 * r2 * (best_swarm_pos[k] - swarm[i].position[k])) )
       # print(swarm[i].velocity[k],end=" ")
        if swarm[i].velocity[k] < minx:
          swarm[i].velocity[k] = minx
        elif swarm[i].velocity[k] > maxx:
          swarm[i].velocity[k] = maxx
      print("")
      # compute new position using new velocity
      for k in range(dim):
        swarm[i].position[k] += swarm[i].velocity[k]
      print("particle",end=" ")
      print(i)
      # compute error of new position
      swarm[i].error = error(swarm[i].position,f)
      # is new position a new best for the particle?
      if swarm[i].error > swarm[i].best_part_err:
        swarm[i].best_part_err = swarm[i].error
        swarm[i].best_part_pos = copy.copy(swarm[i].position)
      # is new position a new best overall?
      if swarm[i].error > best_swarm_err:
        best_swarm_err = swarm[i].error
        best_swarm_pos = copy.copy(swarm[i].position)
    # for-each particle
    epoch += 1
  # while
  return best_swarm_pos

f = pd.read_csv('mll.csv')
dim = f.shape[0]
num_particles = f.shape[1]-1
max_epochs = 10
print("Setting num_particles = " + str(num_particles))
print("Setting max_epochs    = " + str(max_epochs))
print("\nStarting PSO algorithm\n")
best_position = Solve(max_epochs, num_particles, dim, -10.0, 10.0,f)
print("\nPSO completed\n")
print("\nBest solution found:")
show_vector(best_position)
err = error(best_position,f)
print("Error of best solution = %.6f" % err)
