import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    The first stage of the DARNN model representing an encoder with
    input attention.
    """
    def __init__(self, D_in, H, T, device=torch.device('cpu')):
        # D_in: input dims for the driving series
        # H: hidden dims of the encoder
        # T: time window
        # device: pytorch device
        super(Encoder, self).__init__()

        self.D_in = D_in
        self.H = H
        self.T = T
        self.device = device
        self.dtype = torch.float

        self.attn = nn.Linear(in_features=2*H+T-1, out_features=1)
        self.lstm = nn.LSTM(input_size=D_in, hidden_size=H, num_layers=1)

    def forward(self, data):
        """
        Attention Mechanism

        params:
        data (batch_size, T-1, D_in): batched tenor input data

        return:
        input_weighted (batch_size, T-1, D_in): weighted input features
        input_encoded (batch_size, T-1, D_in): encoded input features
        """
        # data: (batch_size, T-1, D_in)
        batch_size, Tend, D_in = data.size()
        input_weighted = data.new_zeros((batch_size, Tend, D_in),
                                        dtype=self.dtype, device=self.device)
        input_encoded = data.new_zeros((batch_size, Tend, self.H),
                                       dtype=self.dtype, device=self.device)

        # hidden, cell: initial states with dimension H
        # (1, batch_size, H)
        hidden = data.new_zeros((1, batch_size, self.H),
                                dtype=self.dtype, device=self.device)
        cell = hidden.clone()

        # iterate over each time step in window
        for t in range(Tend):
            # [1] Input Attention Mechanism
            # The input attention mechanism computes attention weights across
            # driving series (input features given time step) conditioned
            # on the prior hidden encoder state.

            # concatenate hidden states with each predictor
            # W[h_prior;s_prior;x]
            # attn_weights: (batch_size, D_in, 2*H+T-1)
            attn_weights = torch.cat((
                hidden.repeat(self.D_in, 1, 1).permute(1, 0, 2),
                cell.repeat(self.D_in, 1, 1).permute(1, 0, 2),
                data.permute(0, 2, 1)), dim=2)

            # feed attn weights thru linear layer and then reshape
            # softmax of the attn weights is taken to ensure sums to 1
            # new input derived by multiplying attention weights with input
            # tanh taken on attention weights per the paper
            # attn_weights: (batch_size*D_in, 2*H+T-1) => (batch_size, D_in)
            # weighted_input: (batch_size, D_in)
            attn_weights = attn_weights.view(-1, self.H*2+Tend)
            attn_weights = self.attn(attn_weights)
            attn_weights = attn_weights.view(-1, D_in)
            attn_weights = F.softmax(attn_weights, dim=1)
            weighted_input = torch.mul(attn_weights, data[:, t, :])
            weighted_input = torch.tanh(weighted_input)

            # [2] Temporal Attention Layer - LSTM
            # lstm input: (seq_len, batch_size, D_in)
            self.lstm.flatten_parameters()
            _, lstm_states = self.lstm(weighted_input.unsqueeze(0),
                                       (hidden, cell))
            hidden, cell = lstm_states
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded


class Decoder(nn.Module):
    def __init__(self, H_en, H_de, T, device=torch.device('cpu')):
        super(Decoder, self).__init__()

        self.T = T
        self.H_en = H_en
        self.H_de = H_de
        self.device = device
        self.dtype = torch.float

        self.attn = nn.Sequential(
            nn.Linear(2*H_de+H_en, H_en),
            nn.Tanh(),
            nn.Linear(H_en, 1))
        self.lstm = nn.LSTM(input_size=1, hidden_size=H_de)
        self.fc1 = nn.Linear(H_en+1, 1)
        self.fc2 = nn.Linear(H_de+H_en, 1)

        self.fc1.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: (batch_size, T-1, H_en)
        # y_history: (batch_size, T-1)

        # init hidden and cell
        # (1, batch_size, H_de)
        hidden = input_encoded.new_zeros((1, input_encoded.size(0), self.H_de),
                                         dtype=self.dtype, device=self.device)
        cell = hidden.clone()

        for t in range(self.T-1):
            # compute attn weights
            # (batch_size, T, 2*H_de+H_en)
            x = torch.cat((hidden.repeat(self.T-1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T-1, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)
            # (batch_size,T-1) row sums to 1
            x = self.attn(x.view(-1, self.H_de*2+self.H_en))
            x = F.softmax(x.view(-1, self.T-1), dim=1)

            # compute context vector
            # (batch_size, H_en)
            ctx = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]
            if t < self.T-1:
                # (batch_size, 1)
                y_tilde = self.fc1(
                    torch.cat((ctx, y_history[:, t].unsqueeze(1)), dim=1)
                )
                # lstm
                # (1,batch_size, H_de)
                self.lstm.flatten_parameters()
                _, lstm_out = self.lstm(y_tilde.unsqueeze(0), (hidden, cell))
                hidden, cell = lstm_out

            # final output
            y_pred = self.fc2(torch.cat((hidden[0], ctx), dim=1))
            return y_pred


class DARNN:
    """
    DA-RNN: Dual-Stage Attention-Based Recurrent Neural Network
    reference: https://arxiv.org/pdf/1704.02971.pdf

    DA-RNN is inspired by theories of human attention which argue that
    behavioral results are best modeled by a two-stage attention where the
    first stage selects elementary stimulus features while the second stage
    uses categorical information to decode the stimulus.

    The DA-RNN model's first stage adaptively extracts relevant driving series
    at each time step by referring to the prior encoder hidden state thereby
    focusing the model on relevant input features.

    The second stage uses a temporal attention mechanism to select relevant
    encoder hidden states across all time steps thereby focusing the model
    on long-term temporal dependencies.

    A square loss is used for the objective function.

    Both stages are integrated within a LSTM-based RNN and jointly trained
    using back propagation. By incorporating both stages the model aims to
    select the most relevant input features as well as capture long-term
    temporal dependencies.
    """
    def __init__(self, data_path, H_en=64, H_de=64, T=10, lr=0.001,
                 batch_size=128, parallel=False, nrows=None,
                 logger_name='darnn', train_split=0.7, gamma=0.95):
        # data_path: path to underlying data
        # H_en: encoder hidden dims
        # H_de: decoder hidden dims
        # T: time steps
        # lr: learning rate
        # batch_size: batch size
        # parallel: indicator for data parallel
        # nrows: rows to read from data file (default None pulls all)
        # logger_name: name of logger to use (if not prior setup creates)
        # train_split: training size pct
        # gamma: learning rate gamma
        self.T = T
        self.batch_size = batch_size

        # setup logger
        self._setup_logger(logger_name)
        self.logger.info('Launching Dual-Stage Attention-Based RNN')

        # pull data from file
        df = pd.read_csv(data_path, nrows=nrows, index_col='Date')
        self.logger.info(f'Data shape: {df.shape}')

        # standardize data
        self.X = df.loc[:, df.columns != 'Adj Close'].values
        N, D_in = self.X.shape
        self.y = df['Adj Close'].values

        # device and dtype setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.logger.info(f'Device Type: {self.device.type}')
        self.dtype = torch.float

        self.encoder = Encoder(D_in=D_in, H=H_en, T=T,
                               device=self.device)
        self.decoder = Decoder(H_en=H_en, H_de=H_de, T=T,
                               device=self.device)

        if self.device.type == 'cuda':
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        if parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.en_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.en_scheduler = optim.lr_scheduler.StepLR(self.en_optimizer,
                                                      1.0, gamma=gamma)
        self.de_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.de_scheduler = optim.lr_scheduler.StepLR(self.de_optimizer,
                                                      1.0, gamma=gamma)

        self.train_size = int(N*train_split)
        self.y = self.y - self.y[:self.train_size].mean()

    def train(self, epochs=10, schedule_interval=10000, log_interval=1):
        # epochs: number of training epochs
        steps_per_epoch = int(
            np.ceil(np.float(self.train_size)/self.batch_size)
        )
        self.logger.info(f'steps per epoch: {steps_per_epoch}')

        self.step_losses = np.zeros(epochs*steps_per_epoch)
        self.epoch_losses = np.zeros(epochs)
        self.loss_fn = nn.MSELoss()

        step = 0
        for i in range(epochs):
            perm_idx = np.random.permutation(self.train_size-self.T)
            j = 0
            while j < steps_per_epoch:
                batch_idx = perm_idx[j:(j+self.batch_size)]
                batch_size = len(batch_idx)
                X = np.zeros((batch_size, self.T-1, self.X.shape[1]))
                y_hist = np.zeros((batch_size, self.T-1))
                y_target = self.y[batch_idx+self.T]

                for k in range(batch_size):
                    X[k, :, :] = self.X[batch_idx[k]:batch_idx[k]+self.T-1, :]
                    y_hist[k, :] = self.y[batch_idx[k]:(batch_idx[k]+self.T-1)]
                loss = self._train_step(X, y_hist, y_target)
                self.step_losses[i*steps_per_epoch+j] = loss
                j += 1
                step += 1

                # learning rate scheduler step
                if step % schedule_interval == 0 and step > 0:
                    self.en_scheduler.step()
                    self.de_scheduler.step()

            # store loss per epoch
            sidx, eidx = i*steps_per_epoch, (i+1)*steps_per_epoch
            self.epoch_losses[i] = self.step_losses[sidx:eidx].mean()

            if i % log_interval == 0:
                self.logger.info(f'epoch {i} | '
                                 f'loss: {self.epoch_losses[i]:.3f}')

    def _train_step(self, X, y_hist, y_target):
        # zero out gradient
        self.en_optimizer.zero_grad()
        self.de_optimizer.zero_grad()

        # data setup
        X_en = torch.tensor(X, dtype=self.dtype, device=self.device)
        y_de = torch.tensor(y_hist, dtype=self.dtype, device=self.device)
        y = torch.tensor(y_target, dtype=self.dtype, device=self.device)

        # take step
        input_weight, input_encoded = self.encoder(X_en)
        y_pred = self.decoder(input_encoded, y_de)
        loss = self.loss_fn(y_pred.squeeze(1), y)
        loss.backward()
        self.en_optimizer.step()
        self.de_optimizer.step()
        return loss.item()

    def predict(self):
        y_pred = np.zeros(self.X.shape[0]-self.train_size)
        i = 0
        N = y_pred.shape[0]
        y_rng = np.arange(N)
        while i < N:
            batch_idx = y_rng[i:i+self.batch_size]
            bsz = batch_idx.shape[0]
            X = np.zeros((bsz, self.T-1, self.X.shape[1]))
            y_hist = np.zeros((bsz, self.T-1))
            for j in range(bsz):
                sidx = batch_idx[j]+self.train_size-self.T
                eidx = batch_idx[j]+self.train_size-1
                X[j, :, :] = self.X[sidx:eidx, :]
                y_hist[j, :] = self.y[sidx:eidx]

            y_hist = torch.tensor(y_hist, dtype=self.dtype,
                                  device=self.device)
            X = torch.tensor(X, dtype=self.dtype,
                             device=self.device)
            _, input_encoded = self.encoder(X)
            y_pred[i:i+self.batch_size] = self.decoder(
                input_encoded,
                y_hist).data.numpy()[:, 0]
            i += self.batch_size

        return y_pred

    def _setup_logger(self, name):
        # create logger with given name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s '
                                      '| %(message)s')

        # add formatter to console handler and console handler to logger
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)


def main():
    # launch DARNN
    data_path = 'data/equities/AAPL.csv'
    model = DARNN(data_path)
    model.train()


if __name__ == '__main__':
    main()
