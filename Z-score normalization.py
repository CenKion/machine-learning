import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; 
plt.style.use('./deeplearning.mplstyle')
from lab_utils_multi import  load_house_data, compute_cost, run_gradient_descent 
from lab_utils_multi import  norm_plot, plt_contour_multi, plt_equal_scale, plot_cost_i_w

def zscore_normalize_features(X):
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    X_norm=(X-mu)/sigma
    return (X_norm,mu,sigma)



X_train,y_train=load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

mu=np.mean(X_train,axis=0)
sig=np.std(X_train,axis=0)
X_mean=X_train-mu
X_norm=(X_mean)/sig

fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_train[:,i])
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count")
fig.suptitle("distribution of features before normalization")
plt.show()
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_norm[:,i])
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count")
fig.suptitle("distribution of features after normalization")
plt.show()

w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )

m=X_norm.shape[0]
yp=np.zeros(m)

for i in range(m):
    yp[i]=np.dot(X_norm[i],w_norm)+b_norm
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train,color=dlblue,label="target")
    ax[i].scatter(X_train[:,i],yp,color=dlorange,label="predict")
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price");ax[0].legend()
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

plt_equal_scale(X_train, X_norm, y_train)












