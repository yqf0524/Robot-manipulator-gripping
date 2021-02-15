import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-10,10,0.2)
y = np.arange(-10,10,0.2)
f_x_y=np.power(x,2)+np.power(y,2)
fig = plt.figure()
ax = plt.gca(projection='3d')
ax.plot(x,y,f_x_y)