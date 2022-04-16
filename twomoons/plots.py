import matplotlib.pyplot as plt
from matplotlib import axis, colors


def plot_confidence(confidences, X_train, t_train, X_test, t_test, x_limit, y_limit, x_grid, y_grid, num_points,
                    title=None):
    fig, axis = plt.subplots(1, 1, figsize=(15, 12))
    axis.set_aspect('equal')
    plot_two_moons(axis, X_train, t_train, X_test, t_test, x_limit=x_limit, y_limit=y_limit)
    pc = axis.pcolormesh(x_grid, y_grid, confidences.reshape(num_points, num_points), cmap=plt.cm.plasma,
                         vmin=min(confidences), vmax=max(confidences))
    fig.colorbar(pc)
    plt.title(title)
    plt.show()


def plot_two_moons(ax, X_train, t_train, X_test, t_test, x_limit, y_limit):
    train_0 = [all(ele) for ele in t_train == [1., 0.]]
    ax.plot(X_train[train_0, 0], X_train[train_0, 1], 'bo', label="Class 0 - train")
    train_1 = [all(ele) for ele in t_train == [0., 1.]]
    ax.plot(X_train[train_1, 0], X_train[train_1, 1], 'go', label="Class 1 - train")
    train_degenerate = [all(ele) for ele in t_train == [0.5, 0.5]]
    # print(X_train[train_degenerate, :])
    ax.plot(X_train[train_degenerate, 0], X_train[train_degenerate, 1], 'w*', label="Degenerate - train")
    test_0 = [all(ele) for ele in t_test == [1., 0.]]
    ax.plot(X_test[test_0, 0], X_test[test_0, 1], 'bx', label="Class 0 - test", markersize=10)
    test_1 = [all(ele) for ele in t_test == [0., 1.]]
    ax.plot(X_test[test_1, 0], X_test[test_1, 1], 'gx', label="Class 1 - test", markersize=10)
    ax.set_ylim(y_limit)
    ax.set_xlim(x_limit)
    ax.legend(ncol=2, loc='lower right')
    ax.set_xlabel('Feature $x_1$')
    ax.set_ylabel('Feature $x_2$')
