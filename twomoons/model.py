import torch
from laplace import Laplace


def get_model():
    torch.manual_seed(711)
    return torch.nn.Sequential(
        torch.nn.Linear(2, 70), torch.nn.Mish(),
        torch.nn.Linear(70, 30), torch.nn.Mish(),
        torch.nn.Linear(30, 2)
    )


def train_laplace_approximation(model, train_dataloader, val_loader):
    la = Laplace(model, 'classification',
                 subset_of_weights='last_layer',
                 hessian_structure='kron')
    la.fit(train_dataloader)
    la.optimize_prior_precision(method='CV', val_loader=val_loader)
    # la.optimize_prior_precision(val_loader=val_loader)
    return la
