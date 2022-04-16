# %%
from matplotlib.pyplot import axis
import numpy as np
import torch

from data import get_two_moons_np, to_softmax_format, get_torch_loaders, prepare_grid
from model import get_model, train_laplace_approximation
from plots import plot_confidence

x_limit = np.array([-4, 4])
y_limit = np.array([-3.5, 4])
root_n_outliers = 5 # 0 for no outliers, 5 for 25 outliers etc

if __name__ == '__main__':
    # Prepare train/test sets for training
    X_train, X_test, t_train, t_test = get_two_moons_np()
    print("Length of train set:", len(X_train))
    t_train_sm = to_softmax_format(t_train)
    t_test_sm = to_softmax_format(t_test)
    if root_n_outliers > 0:
    # Add a bit of background data
        x_grid, y_grid, XX, num_points = prepare_grid(x_limit*0.9, y_limit*0.9, num_points=root_n_outliers)
        X_train = np.concatenate((X_train, XX), axis=0)
        t_train_sm = np.concatenate((t_train_sm, np.zeros((num_points**2, 2)) + 0.5), axis=0)
    train_dataloader, val_dataloader = get_torch_loaders(X_train, X_test, t_train_sm, t_test_sm, batch_size=10)

    model = get_model()
    n_epochs = 250
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    for i in range(n_epochs):
        for X, y in train_dataloader:
            optimizer.zero_grad()
            pred = torch.softmax(model(X), dim=1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
    print(model.eval())
    # Prepare mesh to investigate models predictions outside train/test set
    x_grid, y_grid, XX, num_points = prepare_grid(x_limit, y_limit, num_points=500)
    _ = np.empty(XX.shape[0])
    mesh_dataloader, _ = get_torch_loaders(XX, np.array([]), _, np.array([]), batch_size=500)

    # Compute MAP confidence
    with torch.no_grad():
        mesh_preds = []
        for X, _ in mesh_dataloader:
            f = torch.softmax(model(X), dim=1).numpy()
            # v = np.max(f, axis=1)
            v = f[:, 0]
            mesh_preds.append(v)
    confidences = np.hstack(mesh_preds)

    plot_confidence(confidences, X_train, t_train_sm, X_test, t_test_sm, x_limit, y_limit, x_grid, y_grid, num_points,
                    "NN prediction confidence")
    # %%
    # Train BNN
    model_BNN = train_laplace_approximation(model, train_dataloader, val_dataloader)
    print("Done training BNN")

    # Compute BNN confidence
    with torch.no_grad():
        mesh_preds_BNN = []
        for X, _ in mesh_dataloader:
            f = model_BNN(X, pred_type='glm', link_approx='mc')
            # v = np.max(f.numpy(), axis=1)
            v = f[:, 0]
            mesh_preds_BNN.append(v)
    confidences_BNN = np.hstack(mesh_preds_BNN)

    plot_confidence(confidences_BNN, X_train, t_train_sm, X_test, t_test_sm, x_limit, y_limit, x_grid, y_grid, num_points,
                    "Laplace approx. BNN prediction confidence")

print("Done.")
# %%

# # Prepare mesh to investigate models predictions outside train/test set
# x_grid, y_grid, XX, num_points = prepare_grid(x_limit, y_limit, num_points=500)
# _ = np.empty(XX.shape[0])
# mesh_dataloader, _ = get_torch_loaders(XX, np.array([]), _, np.array([]), batch_size=500)

# # Compute BNN confidence
# with torch.no_grad():
#     mesh_preds_BNN = []
#     for X, _ in mesh_dataloader:
#         f = model_BNN(X, pred_type='nn') 
#         v = np.max(f.numpy(), axis=1)
#         mesh_preds_BNN.append(v)
# confidences_BNN = np.hstack(mesh_preds_BNN)

# plot_confidence(confidences_BNN, X_train, t_train, X_test, t_test, x_limit, y_limit, x_grid, y_grid, num_points,
#                 "Laplace approx. BNN prediction confidence")


# %%
