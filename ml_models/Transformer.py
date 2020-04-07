import torch  # noqa
from torch import nn


class Transcoder(nn.Module):
    """
    Attention-based Transcoder
    """
    def __init__(self, device=None):
        """
        params
        device: tensor device
        """
        super(Transcoder, self).__init__()

        self.device = device

    def forward(self, X):
        """
        params
        X (batch_size, T, D_in): input minibatch

        return
        y_pred (batch_size, T, D_in): output prediction
        """
        pass
