from torch import nn
from ml.models.Encoder import Encoder
from ml.models.Decoder import Decoder
from ml.models.utils import pos_encodings


class Transformer(nn.Module):
    """
    Attention-based Transformer based on 'All You Need Is Attention' paper.
    """
    def __init__(self, D_in, D_embed, D_out, Q, V, H, N,
                 local_attn_size=None, dropout=0.3, P=4,
                 device=None):
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
        self.output_layer = nn.Linear(Q*D_embed, D_out)

    def forward(self, X):
        """
        params
        X (batch_size, T, D_in): input minibatch

        return
        y_pred (batch_size, T, D_out): output prediction
        """
        # positionally encode and embed input
        batch_size, T, _ = X.size()
        PE = pos_encodings(T, self.P, self.D_embed).to(self.device)

        X = self.embedding_layer(X)
        X = X.add_(PE)

        # pass inputs thru encoding layers
        for layer in self.encoding_layers:
            X = layer(X)

        # positionally encode outputs
        encoded_attn = X
        X = X.add_(PE)

        # pass outputs thru decoding layers
        # X: (batch_size, T, D_embed)
        for layer in self.decoding_layers:
            X = layer(X, encoded_attn=encoded_attn)

        # reshape and generate output
        X = X.reshape(batch_size, -1)
        X = self.output_layer(X)
        return X
