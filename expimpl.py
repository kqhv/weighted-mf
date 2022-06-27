import copy
import time
from abc import ABC

import torch
from tqdm import tqdm

from wmf import WeightedMF


class Experiment(ABC):
  def __init__(self):
    self.full_losses = []
    self.batch_losses = []
    self.users_history = []
    self.items_history = []
    self.runtimes = []
    return


class GradientDescentExperiment(Experiment):
  def __init__(self, wmf: WeightedMF, mat, users, items,
               optim=torch.optim.Adam, optim_kwargs: dict = None):
    super().__init__()

    mat.clone().float().detach().requires_grad_(True)
    users = users.clone().detach().requires_grad_(True)
    items = items.clone().detach().requires_grad_(True)
    optim_kwargs = optim_kwargs if optim_kwargs is not None else {}

    self._wmf = wmf
    self._mat = mat
    self._users = users
    self._items = items

    self._optim = optim([users, items], **optim_kwargs)

    return

  def resume(self, batch_size=None, n_iters=1000):
    # init
    full_losses, batch_losses = self.full_losses, self.batch_losses
    users_history, items_history = self.users_history, self.items_history
    runtimes = self.runtimes

    wmf = self._wmf
    mat, users, items = self._mat, self._users, self._items
    optim = self._optim

    start_iter = len(runtimes)
    print(f"Continue at {start_iter}-th iteration.")
    pbar = tqdm(range(start_iter, start_iter + n_iters, 1), position=0, leave=True)
    for _ in pbar:
      starttime = time.time()

      idx = torch.randperm(len(users))[:batch_size] if batch_size is not None else None
      batch_loss = wmf.forward(mat=mat, users=users, items=items, idx=idx, debug=True)

      # keep track of the loss values
      full_losses.append(wmf.full_loss.item())
      batch_losses.append(batch_loss.item())
      pbar.set_postfix(batch_loss=f'{batch_loss:,.3f}', full_loss=f'{wmf.full_loss:,.3f}')

      # update params
      optim.zero_grad()
      batch_loss.backward()
      optim.step()

      users_history.append(copy.deepcopy(users.detach().numpy()))
      items_history.append(copy.deepcopy(items.detach().numpy()))

      runtimes.append(time.time() - starttime)

    return None


class ALSExperiment(Experiment):
  def __init__(self, wmf: WeightedMF, mat, users, items):
    super().__init__()

    mat = mat.clone().float().detach()
    users = users.clone().detach()
    items = items.clone().detach()

    self._wmf = wmf
    self._mat = mat
    self._users = users
    self._items = items
    return

  def resume(self, n_iters=50):
    # init
    full_losses = self.full_losses
    users_history, items_history = self.users_history, self.items_history
    runtimes = self.runtimes
    wmf = self._wmf
    mat, users, items = self._mat, self._users, self._items

    if len(users_history) >= 1:
      users = users_history[-1].clone().detach()
    if len(items_history) >= 1:
      items = items_history[-1].clone().detach()

    start_iter = len(runtimes)
    print(f"Continue at {start_iter}-th iteration.")
    pbar = tqdm(range(start_iter, start_iter + n_iters, 1), position=0, leave=True)
    for _ in pbar:
      starttime = time.time()

      # keep track of the loss values
      loss = wmf.forward(mat=mat, users=users, items=items, idx=None, debug=False)
      full_losses.append(loss.item())
      pbar.set_postfix(full_loss=f'{loss:,.3f}')

      # update params
      users, items = wmf.als_step(mat=mat, users=users, items=items)
      users_history.append(users)
      items_history.append(items)

      runtimes.append(time.time() - starttime)

    return None
