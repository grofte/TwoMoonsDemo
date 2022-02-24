import matplotlib.pyplot as plt
from matplotlib import colors


def plot_confidence(confidences, X_train, t_train, X_test, t_test, x_limit, y_limit, x_grid, y_grid, num_points,
                    title=None, plot_points=True):
    fig, axis = plt.subplots(1, 1, figsize=(15, 12))
    axis.set_aspect('equal')
    if plot_points:
        plot_two_moons(axis, X_train, t_train, X_test, t_test, x_limit=x_limit, y_limit=y_limit)
    pc = axis.pcolormesh(x_grid, y_grid, confidences.reshape(num_points, num_points), cmap=plt.cm.RdBu_r,
                         vmin=min(confidences), vmax=max(confidences))
    fig.colorbar(pc)
    plt.title(title)
    plt.show()


def plot_two_moons(ax, X_train, t_train, X_test, t_test, x_limit, y_limit):
    t_train_T = t_train.reshape(t_train.shape[0])
    t_test_T = t_test.reshape(t_test.shape[0])

    ax.plot(X_train[t_train_T == 0, 0], X_train[t_train_T == 0, 1], 'bo', label="Class 0 - train")
    ax.plot(X_train[t_train_T == 1, 0], X_train[t_train_T == 1, 1], 'go', label="Class 1 - train")
    ax.plot(X_test[t_test_T == 0, 0], X_test[t_test_T == 0, 1], 'bx', label="Class 0 - test", markersize=10)
    ax.plot(X_test[t_test_T == 1, 0], X_test[t_test_T == 1, 1], 'gx', label="Class 1 - test", markersize=10)
    ax.set_ylim(y_limit)
    ax.set_xlim(x_limit)
    ax.legend(ncol=2, loc='lower right')
    ax.set_xlabel('Feature $x_1$')
    ax.set_ylabel('Feature $x_2$')
