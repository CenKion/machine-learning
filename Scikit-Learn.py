import numpy as np
np.set_printoptions(precision=2)
from sklearn.linear_model import SGDRegressor,LinearRegression
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import  load_house_data
import matplotlib.pyplot as plt
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; 
plt.style.use('./deeplearning.mplstyle')


X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']
m,n=X_train.shape

scaler=StandardScaler()
X_norm=scaler.fit_transform(X_train)
print(f"X_train mean:{scaler.mean_},sigma:{scaler.scale_}")
print(f"peak to peak before normoalization:{np.ptp(X_train,axis=0)} , after:{np.ptp(X_norm,axis=0)}")

sgdr=SGDRegressor(max_iter=1000)
sgdr.fit(X_norm,y_train)
print(f"iterations: {sgdr.n_iter_}, updates: {sgdr.t_}")
print(f"b,w found by gradient descent:{sgdr.intercept_},{sgdr.coef_}")
y_pred=sgdr.predict(X_norm)
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(n):
    ax[i].scatter(X_train[:,i],y_train,color=dlorange,label="target")
    ax[i].scatter(X_train[:,i],y_pred,label="predict")
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("y")
ax[0].legend()
fig.suptitle("target and predic")
plt.show()










