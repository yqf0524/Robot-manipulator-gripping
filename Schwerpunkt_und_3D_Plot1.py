import numpy as np
import trimesh
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

mesh = trimesh.load('cardoor_refined_3845_vertices.stl')

x_sp = np.average(mesh.vertices[:, 0])
y_sp = np.average(mesh.vertices[:, 1])
z_sp = np.average(mesh.vertices[:, 2])

mesh.vertices = mesh.vertices - [x_sp, y_sp, z_sp]

fig = plt.figure()
ax = fig.gca(projection='3d')

scat = ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2])

X = mesh.vertices[:, 0]
Y = mesh.vertices[:, 1]
Z = mesh.vertices[:, 2]

# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([xb], [yb], [zb], 'w')

plt.grid()
plt.show()
