import torch
import torch.nn as nn
import numpy as np


class MHA(nn.Module):
    """
    Multi-Head Attention model based on 'Attention Is All You Need' paper.

    Generates self-attention tensor given requisite Query, Key, Value tensors.
    """
    def __init__(self, D_embed, Q, V, H, local_attn_size=None, fwd_attn=True,
                 device=None):
        """
        params
        D_embed (scalar): input embedding feature dimension
        Q (scalar): query matrix dimension
        V (scalar): value matrix dimension
        H (scalar): number of heads
        local_attn_size: number of prior local elements attention will consider
        fwd_attn: indicator for whether to include forward attention mask
        device: tensor device
        """
        super(MHA, self).__init__()

        self.H = H
        self.local_attn_size = local_attn_size
        self.fwd_attn = fwd_attn
        self.device = device
        self.scores = None

        # multi-head linear projection layers
        self.W_q = nn.Linear(D_embed, Q*H)
        self.W_k = nn.Linear(D_embed, Q*H)
        self.W_v = nn.Linear(D_embed, V*H)

        # softmax layer
        self.smax = nn.Softmax(dim=-1)

        # output layer
        self.W_o = nn.Linear(V*H, D_embed)

    def forward(self, q_in, k_in, v_in):
        """
        Feed forward input thru multi-head attention.

        params
        q_in (batch_size, T, D_embed): input tensor used to compute queries
        k_in (batch_size, T, D_embed): input tensor used to compute keys
        v_in (batch_size, T, D_embed): input tensor used to compute values

        returns
        attn (batch_size, T, D_embed): self attention tensor
        """
        T = q_in.size(1)

        # [1] multi-head linear projection
        # queries: (batch_size*H, T, Q)
        # keys: (batch_size*H, T, Q)
        # values: (batch_size*H, T, V)
        queries = torch.cat(self.W_q(q_in).chunk(self.H, dim=-1), dim=0)
        keys = torch.cat(self.W_k(k_in).chunk(self.H, dim=-1), dim=0)
        values = torch.cat(self.W_v(v_in).chunk(self.H, dim=-1), dim=0)

        # [2] scaled dot-product
        # scores based on scaled dot product QK/sqrt(T)
        # scores: (batch_size*H, T, T)
        self.scores = torch.bmm(queries, keys.transpose(1, 2)) / (T**0.5)

        # [3] conditionally apply local and forward masks
        if self.local_attn_size is not None:
            i, j = np.indices((T, T))
            mask = np.abs(i-j) > self.local_attn_size
            mask = torch.BoolTensor(mask).to(self.device)
            self.scores = self.scores.masked_fill(mask, -np.inf)

        if self.fwd_attn:
            mask = torch.triu(torch.ones((T, T), device=self.device),
                              diagonal=1).bool()
            self.scores = self.scores.masked_fill(mask, -np.inf)

        # [4] update scores with softmax (must come after inf fill)
        self.scores = self.smax(self.scores)

        # [5] concatenated attention heads
        # attn: (batch_size, T, V*H)
        attn = torch.bmm(self.scores, values)
        attn = torch.cat(attn.chunk(self.H, dim=0), dim=-1)

        # [6] linear output of attention heads
        # attn: (batch_size, T, D_embed)
        attn = self.W_o(attn)

        return attn
