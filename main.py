import torch

from data import df2mat, read_data
from expimpl import GradientDescentExperiment, ALSExperiment
from landscape import cut_1d_section
from wmf import WeightedMF

df = read_data('ml-100k/u.data')
mat, ule, ile = df2mat(df)

print(mat.shape)

device = torch.device('cpu')

k = 20
users0 = torch.randn(mat.shape[0], k, requires_grad=True, device=device)
items0 = torch.randn(k, mat.shape[1], requires_grad=True, device=device)
mat = torch.from_numpy(mat.todense()).to(device)

wmf = WeightedMF()

# cut_1d_section(wmf, mat,
#                users0, torch.randn(mat.shape[0], k, requires_grad=False, device=device),
#                items0)

# gde = GradientDescentExperiment(wmf, mat, users0, items0,
#                                 optim=torch.optim.Adam, optim_kwargs={'lr': 0.01})

als = ALSExperiment(wmf, mat, users0, items0)

# gde.resume()
als.resume()
