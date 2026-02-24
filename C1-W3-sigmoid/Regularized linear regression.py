import numpy as np
import matplotlib.pyplot as plt
from plt_overfit import overfit_example, output
from lab_utils_common import sigmoid
np.set_printoptions(precision=8)

def compute_cost_linear_reg(X,y,w,b,lambda_=1):
    m,n=X.shape
    sum_w=0.0
    cost=0.0
    for i in range(n):
        sum_w+=w[i]**2
    for i in range(m):
        cost+=(np.dot(X[i],w)+b-y[i])**2
    sum_w=sum_w*lambda_
    cost+=sum_w
    cost/=2*m
    return cost

def compute_cost_logistic_reg(X,y,w,b,lambda_=1):
    m,n=X.shape
    cost=0.0
    sum_w=0.0
    for i in range(n):
        sum_w+=w[i]**2
    sum_w*=lambda_/2/m
    for i in range(m):
        f_wb=sigmoid(np.dot(X[i],w)+b)
        cost+=-y[i]*np.log(f_wb)-(1-y[i])*np.log(1-f_wb)
    cost/=m
    cost+=sum_w
    return cost
        
def compute_gradient_logistic_reg(X,y,w,b,lambda_):
    m,n=X.shape
    dj_dw=np.zeros(n)
    dj_db=0
    for i in range(m):
        err=sigmoid(np.dot(X[i],w)+b)-y[i]
        for j in range(n):
            dj_dw[j]+=err*X[i,j]
        dj_db+=err
    for i in range(n):
        dj_dw[i]+=lambda_*w[i]
    dj_dw/=m
    dj_db/=m
    return dj_db,dj_dw
            
np.random.seed(1)
X_tmp=np.random.rand(5,3)
y_tmp=np.array([0,1,0,1,0])
w_tmp=np.random.rand(3)
b_tmp=0.5
lambda_tmp=0.7
dj_db,dj_dw=compute_gradient_logistic_reg(X_tmp,y_tmp,w_tmp,b_tmp,lambda_tmp)
print(f"dj_db: {dj_db}", )
print(f"Regularized dj_dw:\n {dj_dw.tolist()}", )




