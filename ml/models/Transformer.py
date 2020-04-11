import torch
from torch import nn
from ml.models.Encoder import Encoder
from ml.models.Decoder import Decoder
from ml.models.utils import pos_encodings


class Transformer(nn.Module):
    """
    Attention-based Transformer based on 'All You Need Is Attention' paper.
    """
    def __init__(self, D_in, D_embed, D_ctx, D_out, Q, V, H, N,
                 local_attn_size=None, dropout=0.3, P=4,
                 device=None):
        """
        params
        D_in (int): input feature dimension
        D_embed (int): embedding dimension
        D_ctx (int): corp_tx dimension
        D_out (int): output feature dimension
        Q (int): query matrix dimension
        V (int): value matrix dimension
        H (int): number of attention heads
        N (int): number of encoding and decoding layers
        local_attn_size (int): number of attention heads
        P (int): periods for positional encoding
        device: tensor device
        """
        super(Transformer, self).__init__()
        self.D_embed = D_embed
        self.P = P
        self.device = device

        self.embedding_layer = nn.Linear(D_in, D_embed)
        self.encoding_layers = nn.ModuleList([
            Encoder(D_embed=D_embed, Q=Q, V=V, H=H,
                    local_attn_size=local_attn_size,
                    fwd_attn=False, device=device) for _ in range(N)
        ])
        self.decoding_layers = nn.ModuleList([
            Decoder(D_embed=D_embed, Q=Q, V=V, H=H,
                    local_attn_size=local_attn_size,
                    fwd_attn=True, device=device) for _ in range(N)
        ])
        self.ctx_layer = nn.Linear(Q*D_embed+D_ctx, 2*(Q*D_embed+D_ctx))
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(2*(Q*D_embed+D_ctx), D_out)

    def forward(self, X_fin, X_ctx):
        """
        params
        X_fin (batch_size, T, D_in): input financials minibatch
        X_ctx (batch_size, 1, D_ctx): input corp_tx minibatch

        return
        y_pred (batch_size, T, D_out): output prediction
        """
        # positionally encode and embed input
        batch_size, T, _ = X_fin.size()
        PE = pos_encodings(T, self.P, self.D_embed).to(self.device)

        X_fin = self.embedding_layer(X_fin)
        X_fin = X_fin.add_(PE)

        # pass inputs thru encoding layers
        for layer in self.encoding_layers:
            X_fin = layer(X_fin)

        # positionally encode outputs
        encoded_attn = X_fin
        X_fin = X_fin.add_(PE)

        # pass outputs thru decoding layers
        # X_fin: (batch_size, T, D_embed)
        for layer in self.decoding_layers:
            X_fin = layer(X_fin, encoded_attn=encoded_attn)

        # combine with corp_tx inputs and generate output
        X_fin = X_fin.reshape(batch_size, -1)
        X_out = torch.cat((X_fin, X_ctx.reshape(batch_size, -1)), dim=1)
        X_out = self.ctx_layer(X_out)
        X_out = self.relu(X_out)
        X_out = self.output_layer(X_out)
        return X_out
