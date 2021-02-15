import numpy as np
import trimesh
import matplotlib.pyplot as plt
from sklearn import decomposition
from scipy.spatial.distance import cosine
import fill_holes
from shapely.geometry import MultiPoint
from shapely.prepared import prep
from scipy.spatial import KDTree
from mpl_toolkits import mplot3d


def z2polar(x, y):
    return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)


def distance_between_points(sp):
    sum_distances = 0
    for m in range(np.size(sp, axis=0)):
        for n in range(np.size(sp, axis=0)):
            sum_distances = sum_distances + np.linalg.norm(sp[m, :] - sp[n, :])

    return sum_distances


def distance_moved(sp, gp):
    moved_sum = 0
    for m in range(np.size(sp, axis=0)):
        for n in range(np.size(sp, axis=0)):
            moved_sum = moved_sum + abs(np.linalg.norm(sp[m, :] - sp[n, :]) - np.linalg.norm(gp[m, :] - gp[n, :]))

    return moved_sum


def exclude_hole_and_edge(v, d, tol, d_hole):
    shapely_v = MultiPoint(v)

    # Rand und Löcher als shaply-Multipolygone
    holes = fill_holes.return_holes(v, max_circumradius=d_hole)
    edge = shapely_v.convex_hull

    edge_prep = prep(edge.buffer(- d - tol))
    incl = np.where(np.asarray(list(map(edge_prep.contains, shapely_v))))
    excl = np.delete(np.arange(np.size(v, axis=0)), incl)

    for i in range(len(holes)):
        holes[i] = holes[i].buffer(d + tol)  # Vergrößerung der Löcher
        holes_prep = prep(holes[i])
        excl = np.concatenate(
            (excl, np.asarray(np.where(np.asarray(list(map(holes_prep.contains, shapely_v))))[0])), axis=0)

    incl = np.delete(np.arange(np.size(v, axis=0)), np.unique(excl))

    return incl


def sp_pieces(v, n, w_s):
    r, w = z2polar(v[:, 0], v[:, 1])
    w = np.mod(w + w_s, 2 * np.pi)
    index_w = np.argsort(w)  # Polarkoordinaten nach Winkel sortieren
    index_split = np.array_split(index_w, n)
    sp_split = np.empty([n, 3])
    for i in range(n):
        sp_split[i, :] = np.average(v[index_split[i][:], :], axis=0)

    sum_dist = distance_between_points(sp_split[:, 0:2])  # Summe aller Punkte zu allen Punkten

    # plt.figure()
    # plt.grid()
    # plt.axis('equal')
    # plt.triplot(v[:, 0], v[:, 1], triangles=mesh.faces, alpha=0.5, zorder=1)
    # plt.scatter(v[index_split[0][:], 0], v[index_split[0][:], 1], s=1, c='r', zorder=2)

    return sp_split, sum_dist


def best_idx(v, incl, sp_split, tol_p):
    _, w = z2polar(v[incl, 0], v[incl, 1])
    _, w_sp = z2polar(sp_split[:, 0], sp_split[:, 1])
    w_range = 5 * np.pi / 180

    nearest_p = np.empty(np.size(sp_split, axis=0)).astype(int)

    for i in range(np.size(sp_split, axis=0)):
        w_zw_sp = np.mod(w - w_sp[i], 2 * np.pi)
        idx_w_sp = np.argwhere(np.logical_or(w_zw_sp < w_range, w_zw_sp > 2 * np.pi - w_range))
        kdtree_zw = KDTree(v[incl[idx_w_sp[:, 0]], :])
        _, nearest_p_zw = kdtree_zw.query(sp_split[i])
        nearest_p[i] = idx_w_sp[nearest_p_zw, 0]

    sp_moved = np.average(v[incl[nearest_p]], axis=0)

    n_it = 0
    final_idx = np.copy(nearest_p)

    while np.linalg.norm(sp_moved[0:2]) > tol_p and n_it < 3:
        for n in range(np.size(final_idx)):
            v_zw = v[incl, :] - v[incl[final_idx[n]], :] + 1/2*sp_moved
            v_abs_zw = np.abs(v_zw)
            x_abs_max = 1 / 2 * np.abs(sp_moved[0])
            y_abs_max = 1 / 2 * np.abs(sp_moved[1])
            idx_zw = np.where(np.logical_and(v_abs_zw[:, 0] <= x_abs_max + tol_p, v_abs_zw[:, 1] <= y_abs_max + tol_p))[0]
            kdtree = KDTree(v_zw[idx_zw])
            _, pt = kdtree.query(-1/2*sp_moved)
            final_idx[n] = idx_zw[pt]
        n_it = n_it + 1
        sp_moved = np.average(v[incl[final_idx]], axis=0)

    # kdtree = KDTree(v[incl, :])
    #
    # valid = False
    # incl_idx = np.arange(np.size(nearest_p))
    #
    # while not valid:
    #     try_p = v[incl[nearest_p[incl_idx]]] - sp_moved * (
    #             1 + (n_greifer - np.size(nearest_p[incl_idx])) / n_greifer)
    #     _, try_np = kdtree.query(try_p)
    #     try_dist = np.empty(np.size(incl_idx))
    #
    #     valid = True
    #     excl_idx = []
    #     for i in range(np.size(incl_idx)):
    #         try_dist[i] = np.linalg.norm(v[incl[try_np[i]], 0:2] - try_p[i, 0:2])
    #         if try_dist[i] > tol_p:
    #             excl_idx.append(i)
    #             valid = False
    #
    #     incl_idx = np.delete(incl_idx, excl_idx)
    #
    # final_idx = np.empty(np.size(nearest_p)).astype(int)
    # for i in range(np.size(nearest_p)):
    #     if np.any(incl_idx == i):
    #         final_idx[i] = try_np[np.where(incl_idx == i)]
    #     else:
    #         final_idx[i] = nearest_p[i]

    moved = distance_moved(sp_split[:, 0:2], v[incl[final_idx], 0:2])

    return final_idx, moved


def eval_curvature(ms, r, tol, std_max_val):
    normals = np.abs(ms.vertex_normals)
    av_std = np.empty(np.size(normals, 0))

    for m in range(np.size(normals, 0)):
        v_zw = ms.vertices - ms.vertices[m, :]
        r_zw, w_zw = z2polar(v_zw[:, 0], v_zw[:, 1])
        in_area = np.where(r_zw < r + tol)
        av_normal_v = np.average(normals[in_area, :], axis=1)
        n_sum = 0
        for n in range(np.size(in_area)):
            n_sum = n_sum + cosine(normals[in_area[0][n], :], av_normal_v)**2
        av_std[m] = np.sqrt(n_sum/(np.size(in_area)-1))

    std_incl_idx = np.where(av_std < std_max_val)

    return std_incl_idx, av_std


def plot_XYZ_projection(xyz, f):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    for m in range(3):
        if m == 0:
            a = x
            b = y
        if m == 1:
            a = x
            b = z
        if m == 2:
            a = y
            b = z
        plt.figure()
        plt.grid()
        plt.axis('equal')
        plt.triplot(a, b, triangles=f)


def plot_excl_normal(xy, f, excl):
    x = xy[:, 0]
    y = xy[:, 1]

    x_excl = x[excl]
    y_excl = y[excl]

    plt.figure()
    plt.grid()
    plt.axis('equal')
    plt.triplot(x, y, triangles=f, alpha=0.3, zorder=1)
    plt.scatter(x_excl, y_excl, s=1, c='r', zorder=2)


def plot_2D(xy, f, xy_incl, opt, final_gp, d):
    x = xy[:, 0]
    y = xy[:, 1]

    x_incl = xy_incl[:, 0]
    y_incl = xy_incl[:, 1]

    plt.figure()
    plt.grid()
    plt.axis('equal')
    plt.triplot(x, y, triangles=f, alpha=0.3, zorder=1)
    # plt.triplot(x_incl, y_incl, color='w', zorder=2)
    plt.scatter(x_incl, y_incl, s=1, c='g', alpha=0.6, zorder=2)
    plt.scatter(opt[:, 0], opt[:, 1], c='w', s=50, zorder=3)
    plt.scatter(opt[:, 0], opt[:, 1], alpha=0.4, c='k', s=50, zorder=4)
    plt.scatter(final_gp[:, 0], final_gp[:, 1], c='r', s=25, zorder=5)
    for n in range(np.size(final_gp, 0)):
        circle = plt.Circle(final_gp[n, :], d, fill=False, color='r', zorder=6)
        plt.gcf().gca().add_artist(circle)
    plt.scatter(0, 0, c='w', s=50, zorder=7)
    plt.scatter(0, 0, c='k', alpha=0.4, s=50, zorder=8)


def plot_3D(xyz, f, final_gp):
    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(X, Y, Z, triangles=f, alpha=0.5, zorder=1)
    ax.scatter(final_gp[:, 0], final_gp[:, 1], final_gp[:, 2], c='r', zorder=2)
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


def plot_color(xy, f, std_plot):
    x = xy[:, 0]
    y = xy[:, 1]

    plt.figure()
    plt.grid()
    plt.axis('equal')
    plt.triplot(x, y, triangles=f, alpha=0.2, zorder=1)
    plt.scatter(x, y, c=np.cbrt(std_plot), s=4, cmap='Reds', zorder=2)


mesh = trimesh.load('cardoor.stl')  # platte.stl

n_greifer = 4  # Anzahl der Greifer
r_greifer = 9  # Radius des Greifers
tol_greifer = 1  # Zusätzlicher Sicherheitsabstand zum Rand des Greifers
d_detect_hole = 10  # Mindestabstand Einstufung als Loch
d_edge = 18  # Abstand zum Rand des Objekts
tol_edge = 2  # Zusätzliche Toleranz
tol_dist_pt = 5  # Toleranz bei der Punktplatzierung
std_max = 0.0004  # Maximale Standardabweichung der Normalen im Saugbereich

# Principal Component Analyse
# 1. Verlagert den Schwerpunkt auf den Nullpunkt
# 2. X-Achse = größte Varianz, Z-Achse = kleinste Varianz
# --> Projektion in XY-Ebene und Anhaltspunkt für den Startpunkt für die Aufteilung der Gebiete

pca = decomposition.PCA(n_components=3)
pca.fit(mesh.vertices)
mesh.vertices = pca.transform(mesh.vertices)

incl_max_std, std_for_scatter = eval_curvature(mesh, r_greifer, tol_greifer, std_max)
excl_max_std = np.setdiff1d(np.arange(np.size(mesh.vertices, axis=0)), incl_max_std)

incl_hole_edge = exclude_hole_and_edge(mesh.vertices, d_edge, tol_edge, d_detect_hole)

inclusion = np.intersect1d(incl_max_std, incl_hole_edge)

w_area = 2 * np.pi / n_greifer
n_w = np.round(w_area / (5 * np.pi / 180)).astype(int)

final = np.empty([n_w, n_greifer]).astype(int)
rating = np.empty(n_w)
sp = np.empty([n_w, n_greifer, 3])

for i in range(n_w):
    sp[i, :, :], sum_dist_sp = sp_pieces(mesh.vertices, n_greifer, i * (5 * np.pi / 180))
    final[i, :], sum_moved = best_idx(mesh.vertices, inclusion, sp[i, :, :], tol_dist_pt)
    rating[i] = sum_dist_sp - sum_moved
    # rating[i] = sum_dist_sp

opt_config = np.argmax(rating)

plot_2D(mesh.vertices, mesh.faces, mesh.vertices[inclusion, :], sp[opt_config, :, :],
        mesh.vertices[inclusion[final[opt_config]]], r_greifer)
plot_3D(mesh.vertices, mesh.faces, mesh.vertices[inclusion[final[opt_config]]])
plot_excl_normal(mesh.vertices, mesh.faces, excl_max_std)
plot_color(mesh.vertices, mesh.faces, std_for_scatter)
# plot_XYZ_projection(mesh.vertices, mesh.faces)

plt.show()
