import torch
from torch import nn


class RNN(nn.Module):
    """
    Vanilla RNN
    """
    def __init__(self, D_in, H, D_out, L=1, nonlinearity='tanh',
                 dropout=0.0, device=None):
        """
        params
        D_in: input feature count
        H: hidden state feature count
        D_out: output feature count
        L: number of layers
        nonlinearity: nonlinearity to use (either tanh or relu)
        dropout: dropout probability for each Dropout layer on RNN outputs
        device: tensor device
        """
        super(RNN, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.L = L
        self.nonlinearity = nonlinearity
        self.dropout = dropout if L > 1 else 0
        self.device = device

        self.rnn = nn.RNN(input_size=self.D_in, hidden_size=self.H,
                          num_layers=self.L, nonlinearity=self.nonlinearity,
                          dropout=self.dropout)
        self.fcout = nn.Linear(self.H, self.D_out)

    def forward(self, X):
        """
        params
        X (batch_size, T, D_in): input minibatch

        return
        y_pred (batch_size, D_out): output prediction
        """
        # init hidden state and permute input to match rnn
        X = X.permute(1, 0, 2)
        T, batch_size, D_in = X.size()
        hidden = X.new_zeros((self.L, batch_size, self.H),
                             dtype=torch.float,
                             device=self.device)

        # ffwd each time step
        for t in range(T):
            self.rnn.flatten_parameters()
            _, hidden = self.rnn(X[t].unsqueeze(0), hidden)

        # ffwd fc output layer
        y_pred = self.fcout(hidden.squeeze(0))
        return y_pred
