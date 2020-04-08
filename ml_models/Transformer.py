import torch
from torch import nn
from Encoder import Encoder
from Decoder import Decoder
from utils import pos_encodings


class Transcoder(nn.Module):
    """
    Attention-based Transcoder based on 'All You Need Is Attention' paper.
    """
    def __init__(self, D_in, D_embed, D_out, Q, V, H, N,
                 local_attn_size=None, fwd_attn=True, dropout=0.3,
                 P=4, device=None):
        """
        params
        D_in (scalar): input feature dimension
        D_embed (scalar): embedding dimension
        D_out (scalar): output feature dimension
        Q (scalar): query matrix dimension
        V (scalar): value matrix dimension
        H (scalar): number of attention heads
        N (scalar): number of encoding and decoding layers
        local_attn_size (scalar): number of attention heads
        fwd_attn (bool): forward attention mask indicator
        P (int): periods for positional encoding
        device: tensor device
        """
        super(Transcoder, self).__init__()
        self.D_embed = D_embed
        self.P = P
        self.device = device

        self.encoding_layers = nn.ModuleList([
            Encoder(D_in=D_in, Q=Q, V=V, H=H,
                    local_attn_size=local_attn_size,
                    fwd_attn=fwd_attn, device=device) for _ in range(N)
        ])
        self.decoding_layers = nn.ModuleList([
            Decoder(D_in=D_in, Q=Q, V=V, H=H,
                    local_attn_size=local_attn_size,
                    fwd_attn=fwd_attn, device=device) for _ in range(N)
        ])

        self.embed_layer = nn.Linear(D_in, D_embed)
        self.out_layer = nn.Linear(D_embed, D_out)

    def forward(self, X):
        """
        params
        X (batch_size, T, D_in): input minibatch

        return
        y_pred (batch_size, T, D_in): output prediction
        """
        T = X.size(1)

        # embed and positionally encode input
        X = self.embed_layer(X)
        PE = pos_encodings(T, self.P, self.D_embed).to(self.device)
        X = X.add_(PE)

        # pass thru encoding layers
        for layer in self.encoding_layers:
            X = layer(X)

        # pass thru decoding layers
        X = X.add_(PE)
        for layer in self.decoding_layers:
            X = layer(X)

        # generate output
        X = torch.sigmoid(self.out_layer(X))

        return X
