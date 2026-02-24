import numpy as np
import matplotlib.pyplot as plt
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh
plt.style.use('./deeplearning.mplstyle')


def sigmoid(z):
    return 1/(1+np.exp(-z))

z_tmp=np.arange(-10,11)
y=sigmoid(z_tmp)
fig,ax=plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp,y,c='b')
ax.set_title("Sigmoid function")
ax.set_xlabel("z")
ax.set_ylabel("sigmoid(z)")
draw_vthresh(ax, 0)

