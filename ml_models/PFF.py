import torch.nn as nn


class PFF(nn.Module):
    """
    Position-Wise Feed Forward Network akin to 1D convolution
    """
    def __init__(self, D_embed, H=None):
        """
        params
        D_embed (int): embedded input dimension
        H (int): hidden dimension (if none defaults to 4*D_embed)
        """
        super(PFF, self).__init__()

        H = 4*D_embed if H is None else H

        # FFN layers
        self.fcin = nn.Linear(D_embed, H)
        self.relu = nn.ReLU()
        self.fcout = nn.Linear(H, D_embed)

    def forward(self, X):
        """
        Pass input thru PFF

        params
        X (batch_size, T, D_embed): input tensor

        returns
        (batch_size, T, D_embed): output tensor
        """
        X = self.relu(self.fcin(X))
        X = self.fcout(X)
        return X
