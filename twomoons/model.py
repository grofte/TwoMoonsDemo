import torch
from laplace import Laplace


def get_model():
    torch.manual_seed(711)
    return torch.nn.Sequential(
        torch.nn.Linear(2, 1000), torch.nn.Tanh(),
        torch.nn.Linear(1000, 2)
    )


def train_laplace_approximation(model, train_dataloader, val_loader):
    la = Laplace(model, 'classification',
                 subset_of_weights='all',
                 hessian_structure='full')
    la.fit(train_dataloader)
    la.optimize_prior_precision(method='CV', val_loader=val_loader)
    return la
