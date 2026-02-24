import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math



def compute_cost(X,y,w,b):
    cost=0.0
    m=X.shape[0]
    for i in range(m):
        cost+=(np.dot(X[i],w)+b-y[i])**2
    cost/=2*m
    return cost

def compute_gradient(X,y,w,b):
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0.0
    for i in range(m):
        err=np.dot(X[i],w)+b-y[i]
        dj_db+=err
        for j in range(n):
            dj_dw[j]+=err*X[i][j]
    dj_dw/=m
    dj_db/=m
    return dj_dw,dj_db

def gradient_descent(X,y,w_in,b_in,compute_cost,compute_gradient,alpha,num_iters):
    w=copy.deepcopy(w_in)
    b=b_in
    J_history=[]
    for i in range(num_iters):
        dj_dw,dj_db=compute_gradient(X,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
        if(i<10000):
            J_history.append(compute_cost(X,y,w,b))
        if(i%math.ceil(num_iters/10)==0):
            print(f"Iterations:{i:4d} ,Cost:{float(J_history[-1]):8.2f}")
    return w,b,J_history



X_train,y_train=load_data()
X_train=X_train.reshape(-1,1)
print(f"X_train shape:{X_train.shape}")
print(f"y_train shape:{y_train.shape}")

plt.scatter(X_train,y_train,label="target",marker='x',c='r')
plt.xlabel("Population in 10,000s")
plt.ylabel("Profit in 10,000$")
plt.legend()
plt.title("Profits vs. Population per city")
plt.show()


initial_w = 0.
initial_b = 0.

iterations = 1500
alpha = 0.01

w,b,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                     compute_cost, compute_gradient, alpha, iterations)
m=X_train.shape[0]
yp=np.zeros(m)
for i in range(m):
    yp[i]=X_train[i]@w+b
plt.plot(X_train[:,0],yp,label="predict",c='r')
plt.scatter(X_train[:,0],y_train,label="target",marker='x',c='b')
plt.legend()
plt.xlabel("Population of city in 10,000s")
plt.ylabel("Profit in 10,000$")
plt.title("Profit vs. Population per city")
plt.show()

































