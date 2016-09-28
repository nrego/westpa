import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import h5py




def mueller(x, y):
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]

    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]
    V1 = 0
    for j in range(4):
        V1 += AA[j] * np.exp(aa[j] * (x - XX[j])**2 + \
              bb[j] * (x - XX[j]) * (y - YY[j]) + cc[j] * (y - YY[j])**2)

    return V1


def plotmueller(beta=1):
    xx, yy = np.mgrid[-1.5:1.2:0.01, -0.2:2.0:0.01]

    v = mueller(xx, yy)

    plt.contourf(xx, yy, v.clip(max=200), 40)


f = h5py.File('strings.h5')

all_strings = f['strings'][...]
plotmueller()
for i in range(all_strings.shape[0]):
    if i % 10 == 0:
        string = all_strings[i]
        #plt.plot(string[:,0], string[:,1], '-o', label='{}'.format(i))

#plt.legend()
plt.colorbar()
plt.show()
    