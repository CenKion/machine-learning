import copy, math
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from lab_utils_common import  dlc, plot_data, plt_tumor_data, sigmoid, compute_cost_logistic
from plt_quad_logistic import plt_quad_logistic, plt_prob
plt.style.use('./deeplearning.mplstyle')

def compute_cost_logistic(X,y,w,b):
    m=X.shape[0]
    cost=0.0
    for i in range(m):
        z_i=np.dot(X[i],w)+b
        f_wb=sigmoid(z_i)
        cost+=-y[i]*np.log(f_wb)-(1-y[i])*np.log(1-f_wb)
    cost/=m
    return cost

def compute_gradient_logistic(X,y,w,b):
    m,n=X.shape
    dj_dw=np.zeros(n)
    dj_db=0.0
    for i in range(m):
        z_i=np.dot(X[i],w)+b
        f_wb=sigmoid(z_i)
        err=f_wb-y[i]
        for j in range(n):
            dj_dw[j]+=err*X[i][j]
        dj_db+=err
    dj_dw/=m
    dj_db/=m
    return dj_db,dj_dw

def gradient_descent(X,y,w_in,b_in,alpha,num_iters):
    w=copy.deepcopy(w_in)
    b=b_in
    J_history=[]
    for i in range(num_iters):
        dj_db,dj_dw=compute_gradient_logistic(X, y, w, b)
        w-=alpha*dj_dw
        b-=alpha*dj_db
        if(i<100000):
            J_history.append(compute_cost_logistic(X, y, w, b))
        if(i%math.ceil(num_iters/10)==0):
            print(f"Iterations {i:4d}: Cost {J_history[-1]:0.2f}")
    return w,b,J_history


X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")


fig,ax = plt.subplots(1,1,figsize=(5,4))
plt_prob(ax, w_out, b_out)


ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')   
ax.axis([0, 4, 0, 3.5])
plot_data(X_train,y_train,ax)


x0 = -b_out/w_out[1]
x1 = -b_out/w_out[0]
ax.plot([0,x0],[x1,0], c=dlc["dlblue"], lw=1)
plt.show()


x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])


from mpl_toolkits import mplot3d
w_range = np.array([-1, 7])
b_range = np.array([1, -14])
quad = plt_quad_logistic( x_train, y_train, w_range, b_range )







