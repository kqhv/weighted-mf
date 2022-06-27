import numpy as np
from tqdm import tqdm

from wmf import WeightedMF


def cut_1d_section(wmf: WeightedMF, mat, users0, users1, items, n=1000):
  assert users0.shape == users1.shape

  # turn off grad to keep low memory footprint
  mat = mat.clone().detach()
  users0 = users0.clone().detach()
  users1 = users1.clone().detach()
  items = items.clone().detach()

  alphas = np.array(range(n)) / n
  losses = []
  pbar = tqdm(alphas, position=0, leave=True)
  for alpha in pbar:
    users_hat = alpha * users0 + (1 - alpha) * users1
    loss = wmf.forward(mat=mat, users=users_hat, items=items)
    losses.append(loss)

    pbar.set_postfix(loss=f'{loss:,.3f}')

  return alphas, losses
