import numpy as np
import torch

from twomoons.data import get_two_moons_np, to_softmax_format, get_torch_loaders, prepare_grid
from twomoons.model import get_model, train_laplace_approximation
from twomoons.plots import plot_confidence

x_limit = (-4, 4)
y_limit = (-3.5, 4)

if __name__ == '__main__':
    # Prepare train/test sets for training
    X_train, X_test, t_train, t_test = get_two_moons_np()
    t_train_sm = to_softmax_format(t_train)
    t_test_sm = to_softmax_format(t_test)
    train_dataloader, val_dataloader = get_torch_loaders(X_train, X_test, t_train_sm, t_test_sm, batch_size=10)

    model = get_model()
    n_epochs = 100
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    for i in range(n_epochs):
        for X, y in train_dataloader:
            optimizer.zero_grad()
            pred = torch.sigmoid(model(X))
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    # Prepare mesh to investigate models predictions outside train/test set
    x_grid, y_grid, XX, num_points = prepare_grid(x_limit, y_limit, num_points=500)
    _ = np.empty(XX.shape[0])
    mesh_dataloader, _ = get_torch_loaders(XX, np.array([]), _, np.array([]), batch_size=100)

    # Compute MAP confidence
    with torch.no_grad():
        mesh_preds = []
        for X, _ in mesh_dataloader:
            f = torch.softmax(model(X), dim=1).numpy()
            v = np.max(f, axis=1)
            mesh_preds.append(v)
    confidences = np.hstack(mesh_preds)

    plot_confidence(confidences, X_train, t_train, X_test, t_test, x_limit, y_limit, x_grid, y_grid, num_points,
                    "NN prediction confidence")

    # Train BNN
    model_BNN = train_laplace_approximation(model, train_dataloader, val_dataloader)

    # Compute BNN confidence
    with torch.no_grad():
        mesh_preds_BNN = []
        for X, _ in mesh_dataloader:
            f = model_BNN(X)
            v = np.max(f.numpy(), axis=1)
            mesh_preds_BNN.append(v)
    confidences_BNN = np.hstack(mesh_preds_BNN)

    plot_confidence(confidences_BNN, X_train, t_train, X_test, t_test, x_limit, y_limit, x_grid, y_grid, num_points,
                    "Laplace approx. BNN prediction confidence")
