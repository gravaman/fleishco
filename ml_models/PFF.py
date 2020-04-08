import torch.nn as nn


class PFF(nn.Module):
    """
    Position-Wise Feed Forward Network akin to 1D convolution
    """
    def __init__(self, D_in, H=None):
        """
        params
        D_in (scalar): input dimension
        H (scalar): hidden dimension (if none defaults to 8*D_in)
        """
        super(PFF, self).__Init__()

        self.D_in = D_in
        self.H = H if H is not None else 8*D_in

        # FFN layers
        self.fcin = nn.Linear(D_in, H)
        self.relu = nn.ReLU()
        self.fcout = nn.Linear(H, D_in)

    def forward(self, X):
        """
        Pass input thru PFF

        params
        X (batch_size, T, D_in): input tensor

        returns
        (batch_size, T, D_in): output tensor
        """
        X = self.relu(self.fcin(X))
        X = self.fcout(X)
        return X
