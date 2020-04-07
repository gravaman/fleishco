import torch.nn as nn
from MHA import MHA
from PFF import PFF


class Encoder(nn.Module):
    """
    Multi-head attention encoder based on 'Attention is All You Need' paper
    Attention output is summed with the input residual and then normalized.
    Resultant output is passed through a position-wise FFN.
    """
    def __init__(self, D_in, Q, V, H, local_attn_size=None, fwd_attn=True,
                 dropout=0.5, device=None):
        super(Encoder, self).__init__()

        self.MHA = MHA(D_in, Q, V, H, local_attn_size=local_attn_size,
                       fwd_attn=fwd_attn, device=device)
        self.PFF = PFF(D_in)
        self.dropout = nn.Dropout(p=dropout)
        self.lnorm1 = nn.LayerNorm(D_in)
        self.lnorm2 = nn.LayerNorm(D_in)

    def forward(self, X):
        """
        params
        X (batch_size, T, D_in): input tensor

        returns
        (batch_size, T, D_in): output tensor
        """
        # multi-head attention segment
        R = X
        X = self.MHA(X)
        X = self.dropout(X)
        X = self.lnorm1(R+X)

        # position-wise feed-forward segment
        R = X
        X = self.PFF(X)
        X = self.dropout(X)
        X = self.lnorm2(R+X)
        return X
