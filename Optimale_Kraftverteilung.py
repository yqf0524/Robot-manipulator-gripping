import numpy as np
import trimesh
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import decomposition


def z2polar(x, y):
    return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)


mesh = trimesh.load('cardoor_refined_3845_vertices.stl')  # cardoor_refined_3845_vertices.stl

n_greifer = 6

# x_sp = np.average(mesh.vertices[:, 0])
# y_sp = np.average(mesh.vertices[:, 1])
# z_sp = np.average(mesh.vertices[:, 2])
#
# mesh.vertices = mesh.vertices - [x_sp, y_sp, z_sp]

pca = decomposition.PCA(n_components=3)
pca.fit(mesh.vertices)
mesh.vertices = pca.transform(mesh.vertices)

X = mesh.vertices[:, 0]
Y = mesh.vertices[:, 1]
Z = mesh.vertices[:, 2]

r, w = z2polar(X, Y)
index_w = np.argsort(w)

index_sort = np.append(index_w[round(np.size(w) * (1 - 1 / (2 * n_greifer))):None],
                       index_w[0:round(np.size(w) * (1 - 1 / (2 * n_greifer)))])
index_split = np.array_split(index_sort, n_greifer)

x_sp = np.empty(n_greifer)
y_sp = np.empty(n_greifer)
z_sp = np.empty(n_greifer)

for i in range(n_greifer):
    x_sp[i] = np.average(X[index_split[i]])
    y_sp[i] = np.average(Y[index_split[i]])
    z_sp[i] = np.average(Z[index_split[i]])

fig = plt.figure()
ax = fig.gca(projection='3d')

scat_sp = ax.scatter(x_sp, y_sp, z_sp, 'r', alpha=1)
scat_obj = ax.scatter(X, Y, Z, alpha=0.2)

# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([xb], [yb], [zb], 'w')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.figure()
plt.scatter(X, Y)
plt.scatter(x_sp, y_sp)

plt.grid()
plt.show()
