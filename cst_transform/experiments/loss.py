import torch as th

class MLCrossEntropy(th.nn.Module):

    def __init__(self, binary_loss=False, weight=None):
        super().__init__()
        self.norm = th.nn.LogSoftmax(dim=-1)
        self.binary_loss = binary_loss
        self.weight = weight
        self.use_weight = weight is not None

    def forward(self, logit, y):

        if self.binary_loss:
            y = y / y.sum(dim=-1, keepdim=True)
            y[y != y] = 0

        x = self.norm(logit)
        w = 1
        if self.use_weight:
            w = self.weight.to(logit.device)

        nll = (-w * y * x).sum(dim=-1)
        nll = nll.mean(dim=0)

        return nll


class HingeLoss(th.nn.Module):

    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.margin = margin

    def forward(self, p, y):

        mask = y != 0

        y = y.float()

        p = th.nn.Tanh()(p)
        L = th.nn.functional.relu(self.margin - y * p)
        L = L[mask]
        return reduce(L, self.reduction)


def reduce(L, reduction):

    if reduction == 'mean':
        return L.mean()
    if reduction == 'sum':
        return L.sum()
    return L
