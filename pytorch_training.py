import numpy as np
import torch


def np_two_layer_nn():
    # numpy two layer nn
    # N: batch size
    # D_in: input dimensiokn
    # H: hidden dimension
    # D_out: output dimension
    N, D_in, H, D_out = 64, 1000, 100, 10

    # create random input and output data
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    # randomly initialize weights
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)

    lr = 1e-6
    steps = 500
    for t in range(steps):
        # fwd pass: compute predicted y
        # h: (N,D_in)x(D_in,H) => (N,H)
        # h_relu: (N,H)
        # y_pred: (N,D_out)
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # compute and print loss
        loss = np.square(y_pred-y).sum()
        if t % 100 == 99:
            print(f'step: {t}, loss: {loss}')

        # compute gradients
        dy_pred = 2*(y_pred-y)
        dw2 = h_relu.T.dot(dy_pred)

        dh_relu = dy_pred.dot(w2.T)
        dh = dh_relu.copy()
        dh[dh < 0] = 0
        dw1 = x.T.dot(dh)

        # update weights
        w1 -= lr*dw1
        w2 -= lr*dw2


def torch_two_layer_nn(device_type="cpu"):
    # pytorch two layer nn
    # device: cpu or cuda:0
    dtype = torch.float
    device = torch.device(device_type)

    # N: batch size
    # D_in: input dim
    # H: hidden dim
    # D_out: output dim
    N, D_in, H, D_out = 64, 1000, 100, 10

    # create random input and output data
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # randomly initialize weights
    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype)

    lr = 1e-6
    steps = 500
    for t in range(steps):
        # fwd pass: compute predicted y
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)

        # compute and print loss
        loss = (y_pred-y).pow(2).sum().item()
        if t % 100 == 99:
            print(f'step: {t}, loss: {loss}')

        # compute gradients
        dy_pred = 2*(y_pred-y)
        dw2 = h_relu.t().mm(dy_pred)

        dh_relu = dy_pred.mm(w2.t())
        dh = dh_relu.clone()
        dh[h < 0] = 0
        dw1 = x.t().mm(dh)

        # update weights
        w1 -= lr*dw1
        w2 -= lr*dw2


def long_two_layer_nn(device_type='cpu'):
    # pytorch two layer nn with autograd

    # device: cpu, cuda:0, or cuda:1
    dtype = torch.float
    device = torch.device(device_type)

    # N: batch size
    # D_in: input dim
    # H: hidden dim
    # D_out: output dim
    N, D_in, H, D_out = 64, 1000, 100, 10

    # create random input and output data
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # randomly initialize weights
    w1 = torch.randn(D_in, H, device=device, dtype=dtype,
                     requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype,
                     requires_grad=True)

    lr = 1e-6
    steps = 500
    for t in range(steps):
        # fwd pass: copute predicted y
        y_pred = x.mm(w1).clamp(min=0).mm(w2)

        # compute and print loss
        loss = (y_pred-y).pow(2).sum()
        if t % 100 == 99:
            print(f'step: {t}, loss: {loss.item()}')

        # backward pass
        loss.backward()

        # update weights without tracking via autograd
        with torch.no_grad():
            w1 -= lr*w1.grad
            w2 -= lr*w2.grad

            # zero out gradients after update
            w1.grad.zero_()
            w2.grad.zero_()


def two_layer_nn(device_type='cpu'):
    # two layer nn
    # device_type: cpu, cuda:0, or cuda:1
    dtype = torch.float
    device = torch.device(device_type)

    # N: batch size
    # D_in: input dim
    # H: hidden dim
    # D_out: output dim
    N, D_in, H, D_out = 64, 1000, 100, 10

    # create random input and output data
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    ).cuda(device=device)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    lr = 1e-6
    steps = 500
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for t in range(steps):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(f'step: {t}, loss: {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10
dtype = torch.float
device_type = 'cuda:0'
device = torch.device(device_type)
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

model = TwoLayerNet(D_in, H, D_out).cuda(device=device)
criterion = torch.nn.MSELoss(reduction='sum')

lr = 1e-6
steps = 500
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for t in range(steps):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    if t % 100 == 99:
        print(f'step: {t}, loss: {loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
