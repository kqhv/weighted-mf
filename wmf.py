import torch


class WeightedMF:
  """
  The loss function in "Collaborative filtering for implicit feedback datasets".
  """

  SCALERS = {'sqrt': torch.sqrt}

  def __init__(self, alp=1.0, eps=1.0, lam=0.1, scaler=None):
    self.alp = alp
    self.eps = eps
    self.lam = lam

    self.scaler = self.SCALERS.get(scaler)

    self.full_loss = torch.tensor(-1.0)
    return

  def als_step(self, mat, users, items):
    """Take a step in the ALS version."""
    alp = self.alp
    eps = self.eps
    lam = self.lam

    p = (mat > 0).float()
    c = 1 + alp * torch.log(1 + mat / eps)

    users = users.clone().detach()
    items = items.clone().detach()

    m, n = mat.shape
    k = users.shape[1]

    eyes = torch.eye(k)

    # update users
    yyt = items @ items.T
    for i in range(m):
      cu = torch.diag(c[i])
      users[i] = torch.inverse(
        yyt + items @ (cu - torch.eye(n)) @ items.T + lam * eyes) @ items @ cu @ p[i]

    del yyt, cu

    # update items
    xtx = users.T @ users
    for i in range(n):
      ci = torch.diag(c[:, i])
      items[:, i] = torch.inverse(
        xtx + users.T @ (ci - torch.eye(m)) @ users + lam * eyes) @ users.T @ ci @ p[:, i]

    return users, items

  def forward(self, mat, users, items, idx=None, debug: bool = True):

    # loss type
    _loss = self._compute_weighted_loss
    kwargs = {'alp': self.alp, 'eps': self.eps, 'lam': self.lam,
              'scaler': self.scaler}

    if debug:
      self.full_loss = _loss(mat=mat, users=users, items=items, **kwargs)

    if idx is None:
      idx = torch.arange(start=0, end=len(users))

    loss = _loss(mat=mat[[idx]], users=users[idx], items=items, **kwargs)

    return loss

  @staticmethod
  def _compute_weighted_loss(mat, users, items, alp, eps, lam, scaler):
    p = (mat > 0).float()
    c = 1 + alp * torch.log(1 + mat / eps)
    loss = ((c * (p - users @ items)) ** 2).sum() + lam * ((users ** 2).sum() + (items ** 2).sum())
    loss = scaler(loss) if scaler is not None else loss  # can scale down to prevent blow-up
    return loss
