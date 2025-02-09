import torch.nn as nn
from ml.models.MHA import MHA
from ml.models.PFF import PFF


class Encoder(nn.Module):
    """
    Multi-head attention encoder based on 'Attention is All You Need' paper.
    Attention output is summed with the input residual and then normalized.
    Resultant output is passed through a position-wise FFN.
    """
    def __init__(self, D_embed, Q, V, H, local_attn_size=None, fwd_attn=False,
                 dropout=0.5, device=None):
        super(Encoder, self).__init__()

        self.attn = MHA(D_embed, Q, V, H, local_attn_size=local_attn_size,
                        fwd_attn=fwd_attn, device=device)
        self.ffwd = PFF(D_embed)
        self.dropout = nn.Dropout(p=dropout)
        self.lnorm1 = nn.LayerNorm(D_embed)
        self.lnorm2 = nn.LayerNorm(D_embed)

    def forward(self, X):
        """
        params
        X (batch_size, T, D_embed): input tensor

        returns
        (batch_size, T, D_embed): output tensor
        """
        # multi-head attention segment
        R = X
        X = self.attn(q_in=X, k_in=X, v_in=X)
        X = self.dropout(X)
        X = self.lnorm1(R+X)

        # position-wise feed-forward segment
        R = X
        X = self.ffwd(X)
        X = self.dropout(X)
        X = self.lnorm2(R+X)
        return X
