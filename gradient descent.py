import copy, math
import numpy as np
import matplotlib.pyplot as plt

def compute_cost(X,y,w,b):
    cost=0.
    m=X.shape[0]
    for i in range(m):
        cost+=(np.dot(X[i],w)+b-y[i])**2
    cost/=2*m
    return cost

def compute_gradient(X,y,w,b):
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0.
    for i in range(m):
        err=np.dot(X[i],w)+b-y[i]
        for j in range(n):
            dj_dw[j]+=err*X[i,j]
        dj_db+=err
    dj_dw/=m
    dj_db/=m
    return dj_dw,dj_db

def gradient_descent(X,y,w_in,b_in,cost_function,gradient_function,alpha,num_iters):
    J_history=[]
    w=copy.deepcopy(w_in)
    b=b_in
    for i in range(num_iters):
        dj_dw,dj_db=gradient_function(X,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
        if(i<100000):
            J_history.append(cost_function(X,y,w,b))
        if(i%math.ceil(num_iters/10)==0):
            print(f"Iteration {i:4d}: Cost {J_history[-1]:0.2f}")
    return w,b,J_history
    
    

X_train=np.array([[2104,5,1,45],
                  [1416,3,2,40],
                  [852,2,1,35]])
y_train=np.array([460,232,178])
print(np.mean(X_train,axis=0))

m,n=X_train.shape
initial_w=np.zeros((n,))
initial_b=0
alpha=5.0e-7
iteration=1000
w_final,b_final,J_hist=gradient_descent(X_train,y_train,initial_w,
                 initial_b,compute_cost,compute_gradient,
                 alpha,iteration)
print(f"b,w found by gradient descent:{b_final:0.2f},{w_final}")
for i in range(m):
    print(f"prediction: {np.dot(X_train[i],w_final)+b_final:0.2f}, target value: {y_train[i]}")


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()







