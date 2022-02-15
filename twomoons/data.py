from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def to_softmax_format(t_set):
    N = len(t_set)
    t_set_sm = np.zeros((N, 2))
    for idx, class_ in enumerate(t_set):
        t_set_sm[idx, class_] = 1.
    return t_set_sm


def get_two_moons_np(size=300, train_size=250):
    X, y = datasets.make_moons(n_samples=size, shuffle=True, noise=0.1, random_state=1234)
    y = y.reshape(-1, 1)
    X_train, X_test, t_train, t_test = train_test_split(X, y, train_size=train_size)
    return X_train, X_test, t_train, t_test


def get_mesh_np(x_limit, y_limit, x1_len=75, x2_len=75):
    vectors = [np.linspace(*x_limit, x1_len), np.linspace(*y_limit, x2_len)]
    X = np.reshape(np.meshgrid(*vectors), (2, -1)).T
    return X


def prepare_grid(x_lim, y_lim, num_points=100):
    x_min, x_max = x_lim
    y_min, y_max = y_lim

    x_grid = np.linspace(x_min, x_max, num_points)
    y_grid = np.linspace(y_min, y_max, num_points)

    XX, YY = np.meshgrid(x_grid, y_grid)
    XX = np.column_stack((XX.ravel(), YY.ravel()))

    return x_grid, y_grid, XX, num_points


def get_torch_loaders(X_train_np, X_test_np, t_train_np, t_test_np, batch_size=16):
    X_train = torch.Tensor(X_train_np)
    t_train = torch.Tensor(t_train_np)
    X_test = torch.Tensor(X_test_np)
    t_test = torch.Tensor(t_test_np)

    train_dataset = TensorDataset(X_train, t_train)
    test_dataset = TensorDataset(X_test, t_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader
