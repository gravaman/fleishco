import torch.nn.functional as F


class MYELoss:
    def __init__(self, standard_stats, reduction='mean'):
        self.standard_stats = standard_stats
        self.reduction = reduction

    def __call__(self, y_pred, y):
        mse = F.mse_loss(y_pred, y, reduction=self.reduction)

        y_pred_adj = self.standard_stats.destandardize(
            y_pred.detach(), is_target=True).exp()
        y_adj = self.standard_stats.destandardize(
            y.detach(), is_target=True).exp()
        mye = F.mse_loss(y_pred_adj, y_adj, reduction=self.reduction)
        return mse, mye
