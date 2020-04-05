import torch
from torch import nn


class LSTM(nn.Module):
    """
    Basic LSTM model.
    """
    def __init__(self, D_in, H, D_out, L=1, dropout=0.5, device=None):
        """
        params
        D_in: input feature count
        H: hidden state feature count
        D_out: output feature count
        L: number of layers
        dropout: dropout probability for each Dropout layer on LSTM outputs
        device: tensor device
        """
        super(LSTM, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.L = L
        self.dropout = dropout if L > 1 else 0
        self.device = device

        self.lstm = nn.LSTM(input_size=self.D_in, hidden_size=self.H,
                            num_layers=self.L, dropout=self.dropout)
        self.fcout = nn.Linear(self.H, self.D_out)

    def forward(self, X):
        """
        params
        X (batch_size, T, D_in): input minibatch

        return
        y_pred (batch_size, D_out): output prediction
        """
        # initialize hidden state and cell
        # lstm expects input shape (T, batch_size, D_in)
        X = X.permute(1, 0, 2)
        T, batch_size, D_in = X.size()
        hidden = X.new_zeros((self.L, batch_size, self.H),
                             dtype=torch.float,
                             device=self.device)
        cell = hidden.clone()

        # ffwd each time step
        for t in range(T):
            self.lstm.flatten_parameters()
            x = X[t].unsqueeze(0)
            _, (hidden, cell) = self.lstm(x, (hidden, cell))

        # ffwd fc output layer
        y_pred = self.fcout(hidden.squeeze(0))
        return y_pred
