import torch.nn as nn
from MHA import MHA
from PFF import PFF


class Decoder(nn.Module):
    """
    Multi-head attention decoder based on 'Attention is All You Need' paper.
    [1] Masked multi-head attention of outputs
    [2] Multi-head attention of encoder and [1]
    [3] Position-wise FFN
    """
    def __init__(self, D_in, Q, V, H, local_attn_size=None, fwd_attn=True,
                 dropout=0.5, device=None):
        """
        params
        D_in (scalar): input feature dimensions
        Q (scalar): query matrix dimension
        V (scalar): value matrix dimension
        H (scalar): number of attention heads
        local_attn_size (scalar): local attention mask size
        fwd_attn (bool): forward attention mask indicator
        device (torch.device): tensor device
        """
        super(Decoder, self).__init__()

        self.attn1 = MHA(D_in, Q, V, H, local_attn_size=local_attn_size,
                         fwd_attn=fwd_attn, device=device)
        self.attn2 = MHA(D_in, Q, V, H, local_attn_size=local_attn_size,
                         fwd_attn=fwd_attn, device=device)
        self.ffwd = PFF(D_in)

        self.dropout = nn.Dropout(p=dropout)
        self.lnorm1 = nn.LayerNorm(D_in)
        self.lnorm2 = nn.LayerNorm(D_in)
        self.lnorm3 = nn.LayerNorm(D_in)

    def forward(self, X, encoded_attn):
        """
        params
        X (batch_size, T, D_in): input tensor
        encoded_attn (batch_size, T, D_in): encoded attention tensor

        returns
        (batch_size, T, D_in): output tensor
        """
        # multi-head attention segment for input
        R = X
        X = self.attn1(q_in=X, k_in=X, v_in=X)
        X = self.dropout(X)
        X = self.lnorm1(R+X)

        # multi-head attention composite of encoder and prior attn outputs
        R = X
        X = self.attn2(q_in=X, k_in=encoded_attn, v_in=encoded_attn)
        X = self.dropout(X)
        X = self.lnorm2(R+X)

        # position-wise feed-forward
        R = X
        X = self.ffwd(X)
        X = self.dropout(X)
        X = self.lnorm3(R+X)

        return X
