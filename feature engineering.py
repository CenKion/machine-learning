import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2)


x=np.arange(0,20,1)
y=x**2
X=np.c_[x,x**2,x**3]
mu=np.mean(X,axis=0)
sig=np.std(X,axis=0)
X=(X-mu)/sig
model_w,model_b=run_gradient_descent_feng(X, y,iterations=100000,alpha=1.0e-1)
plt.scatter(x,y,marker='x',c='r',label="target")
plt.plot(x,X@model_w+model_b,label="predicted")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("normaolized x,x**2,x**3 features")
plt.show()








