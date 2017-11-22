import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D


def show_surface(fis, x_label, y_label, z_label, n_pts=25, x_range=None,
                 y_range=None, z_range=None, ax=None, title=None):
    if ax is None:
        fig = plt.figure(figsize=(4, 8))
        _ax = fig.gca(projection='3d')
    else:
        _ax = ax

    def get_range_lv(label):
        for r in fis.rules:
            for a in r.antecedents:
                if a.lv_name.name == label:
                    return a.lv_name.in_range

    lv_x_range = x_range if x_range is not None else get_range_lv(x_label)
    lv_y_range = y_range if y_range is not None else get_range_lv(y_label)

    X = np.linspace(*lv_x_range, n_pts)
    Y = np.linspace(*lv_y_range, n_pts)
    X, Y = np.meshgrid(X, Y)

    def f(x, y):
        return fis.predict({x_label: x, y_label: y})[z_label]

    f_vect = np.vectorize(f)
    v_out = f_vect(X, Y)

    # Plot the surface.
    surf = _ax.plot_surface(X, Y, v_out, cmap=cm.viridis,
                            linewidth=0, antialiased=True)

    # Customize the z axis.
    _ax.set_xlabel(x_label)
    _ax.set_ylabel(y_label)
    _ax.set_zlabel(z_label)

    if z_range is not None:
        _ax.set_zlim(*z_range)

    _ax.xaxis.set_major_locator(MaxNLocator(5))
    _ax.yaxis.set_major_locator(MaxNLocator(5))

    if title is not None:
        _ax.set_title(title)

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    if ax is None:
        plt.show()
