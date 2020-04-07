from torch import nn
from Encoder import Encoder
from Decoder import Decoder


class Transcoder(nn.Module):
    """
    Attention-based Transcoder based on 'All You Need Is Attention' paper.
    """
    def __init__(self, D_in, D_embed, D_out, Q, V, H, N,
                 local_attn_size=None, fwd_attn=True, dropout=0.3,
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
        fwd_attn (bool): forward attention mask indicator
        device: tensor device
        """
        super(Transcoder, self).__init__()

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
        pass
