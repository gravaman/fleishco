import logging
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from ml_models.StockDataset import (
    StockDataset, StockBatch,
    TickerDataset
)
from ml_models.RNN import RNN
from ml_models.LSTM import LSTM
from ml_models.Transformer import Transformer
from ml_models.utils import (
    list_files, path_to_ticker,
    line_plot
)


##############################
# TODO
# pull data from postgres
# add CLI parser
# visualizations
##############################

# constants
MODEL_TYPES = ['rnn', 'lstm', 'transformer']


def setup_logger(name, level='DEBUG', fmt=None):
    level = level.upper()
    logger = logging.getLogger(name)
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)

    if fmt is None:
        fmt = (
               '%(asctime)s - %(name)s - %(levelname)s '
               '| %(message)s'
        )

    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)
    return logger


def setup_dataloaders(ticker_dir, tickers_per_batch=4, mbatch_size=50,
                      num_workers=4, pmem=False, train_split=0.7,
                      val_split=0.2):
    """
    builds train, val, and test dataloaders
    """
    # data splits
    test_split = 1-train_split-val_split
    assert test_split > 0, 'train_split+val_split must be less than 1'

    ticker_paths = list_files(ticker_dir)
    data_size = len(ticker_paths)
    idxs = np.arange(data_size)
    np.random.shuffle(idxs)

    train_size = int(np.floor(data_size*train_split))
    val_size = int(np.floor(data_size*val_split))

    train_idxs = idxs[:train_size]
    val_idxs = idxs[train_size:train_size+val_size]
    test_idxs = idxs[train_size+val_size:]

    # data loaders
    shapes, loaders = [], []
    StockBatch.mbatch_size = mbatch_size
    for i, ticker_idxs in enumerate([train_idxs, val_idxs, test_idxs]):
        dataset = StockDataset(ticker_dir,
                               idxs=ticker_idxs,
                               index_col='Date')
        if i == 0:
            # get stats for training data
            stats = dataset.get_stats()
        else:
            # set stats for validation and testing data based on training
            dataset.mu, dataset.std = stats

        shapes.append(dataset.size())
        loaders.append(DataLoader(dataset, batch_size=tickers_per_batch,
                                  shuffle=True, pin_memory=pmem,
                                  num_workers=num_workers,
                                  collate_fn=StockBatch))

    return shapes, loaders, stats, test_idxs


def get_viz_dataset(ticker_path, stats):
    dataset = TickerDataset(ticker_path, index_col='Date', stats=stats)
    return dataset


def test_dataloader(loader, logger_name):
    logger = logging.getLogger(logger_name)
    logger.info('launching dataloader test')
    for i, batch in enumerate(loader):
        logger.info(f'batch idx: {i} X size: {batch.X.size()}'
                    f'y size: {batch.y.size()}')
    logger.info('dataloader test complete')


def train(model, loader, optimizer, loss_fn, device, logger_name,
          grad_norm_max=0.5, log_rate=0.25, isdataset=False):
    logger = logging.getLogger(logger_name)
    model.train()
    total_loss = 0.0
    start_time = time.time()
    total_steps = len(loader)
    log_interval = max(int(np.floor(len(loader)*log_rate)), 1)
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        if isdataset:
            X = batch[0].to(device)
            y = batch[1].to(device)
        else:
            X = batch.X.to(device)
            y = batch.y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       grad_norm_max)
        optimizer.step()

        total_loss += loss.item()
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'| step {i:5d}/{total_steps:5d} '
                f'| lr: {lr:0.4f} '
                f'| ms/step: {elapsed*1000/log_interval:5.2f} '
                f'| loss: {cur_loss:5.2f} '
            )
            total_loss = 0.0
            start_time = time.time()


def evaluate(model, loader, loss_fn, device, isdataset=False):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            if isdataset:
                X = batch[0].to(device)
                y = batch[1].to(device)
            else:
                X = batch.X.to(device)
                y = batch.y.to(device)
            y_pred = model(X)
            total_loss += loss_fn(y_pred, y)
            total_count += len(X)

    return total_loss / total_count


def project(model, dataset, loss_fn, device, title=None,
            should_show=False, savepath=None, logger_name=None):
    """
    Generates projection and conditionally plots results.

    params
    model (nn.Module): model to use for projection
    dataset (iterable): input dataset
    loss_fn (nn.Module): evaluation loss function
    device (torch.device): tensor device
    title (str): chart title
    should_show (bool): plot indicator
    savepath (str): location for storing chart
    logger_name (str): name of logger to use

    returns
    avg_loss (float): average loss over projections
    std_loss (float): standard deviation of loss over projections
    """
    # setup
    model.eval()
    Y_pred, Y, L = [], [], []
    base = dataset.base_index[-1]
    if dataset.stats is not None:
        mu, std = [stat[-1] for stat in dataset.stats]

    # generate predictions
    with torch.no_grad():
        for X, y in dataset:
            y_pred = model(X.to(device))

            # get loss
            loss = loss_fn(y_pred, y.to(device)).item()

            # conditionally revert standardization
            if dataset.stats is not None:
                loss = loss*std+mu
                y_pred, y = y_pred.cpu().numpy()*std+mu, y.numpy()*std+mu
                y_pred, y = y_pred*base, y*base

            # store loss
            L.append(loss)

            # store projections
            y_pred, y = y_pred[:, -1], y[:, -1]
            Y_pred = np.concatenate((Y_pred, y_pred), axis=0)
            Y = np.concatenate((Y, y), axis=0)

    # generate and log loss stats
    L = np.array(L)
    avg_loss, std_loss = L.mean(), L.std()

    if logger_name is not None:
        logger = logging.getLogger(logger_name)
        logger.info(f'projection stats: avg loss {avg_loss:5.5f} '
                    f' | std loss {std_loss:5.5f}')

    # generate chart
    if should_show or savepath is not None:
        X = np.arange(len(Y_pred))
        X = dataset.fwd_dates
        line_plot(X, [Y_pred, Y], labels=['pred', 'true'],
                  title=title, should_show=should_show, savepath=savepath)

    return avg_loss, std_loss


def main():
    # set random seed
    np.random.seed(0)

    # available types: rnn, lstm, tranformer
    model_type = 'transformer'

    assert model_type in MODEL_TYPES, f'unknown model_type: {model_type}'

    # logger setup
    logger_name = model_type
    logger = setup_logger(logger_name)

    # device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f'device type: {device.type}')

    # dataloaders setup
    ticker_dir = 'data/test'
    tickers_per_batch = 4
    mbatch_size = 400  # minibatch size
    num_workers = 4  # number of data loader workers
    pmem = True if device.type == 'cuda' else False

    # loader setup
    should_viz = True
    shapes, loaders, stats, tidxs = setup_dataloaders(
        ticker_dir,
        tickers_per_batch=tickers_per_batch,
        mbatch_size=mbatch_size,
        num_workers=num_workers, pmem=pmem)
    train_loader, val_loader, test_loader = loaders

    if should_viz:
        test_idx = tidxs[0]
        test_idx = 0
        ticker_path = list_files(ticker_dir)[test_idx]
        test_ticker = path_to_ticker(ticker_path)
        viz_dataset = get_viz_dataset(ticker_path, stats=stats)

    test_dataset = TickerDataset(ticker_path, index_col='Date',
                                 T_back=10, T_fwd=1, standardize=True)
    print(f'test dataset batches: {len(test_dataset)}')

    # model setup
    train_xshape, train_yshape = shapes[0]
    # D_in = train_xshape[2]  # number of input features
    D_in = 6  # from model
    D_out = 1  # from model
    # D_out = train_yshape[1]  # number of output features
    if model_type == 'transformer':
        D_embed = 512  # embedding dimension
        # Q = train_xshape[1]  # query matrix dimesion (T)
        # V = train_xshape[1]  # value matrix dimension (T)
        Q = 10  # from model
        V = 10  # from model
        H = 4  # number of heads
        N = 6  # number of encoder and decoder stacks
        attn_size = None  # local attention mask size
        dropout = 0.3  # dropout pct
        P = 5  # periodicity of input data
        model = Transformer(D_in, D_embed, D_out, Q, V, H, N,
                            local_attn_size=attn_size, dropout=dropout,
                            P=P, device=device).to(device)
    elif model_type == 'lstm':
        H = 10  # number of hidden state features
        model = LSTM(D_in, H, D_out, device=device).to(device)
    elif model_type == 'rnn':
        H = 10  # number of hidden state features
        model = RNN(D_in, H, D_out, device=device).to(device)

    # optimizer setup
    optimize_type = 'adam'
    lr = 0.0001  # learning rate
    if optimize_type == 'adam':
        wd = 0.0  # weight decay (L2 penalty)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=wd)
    elif optimize_type == 'sgd':
        sched_step_size = 1  # epochs step size for decay
        gamma = 0.95  # lr decay rate
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    sched_step_size,
                                                    gamma)

    # train model
    # TODO: change loader back to train_loader and val_loader!!!
    epochs = 10
    best_model = None
    best_val_loss = float('inf')
    train_loss_fn = nn.MSELoss()
    eval_loss_fn = nn.MSELoss(reduction='sum')
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train(model, test_dataset, logger_name=model_type, device=device,
              optimizer=optimizer, loss_fn=train_loss_fn, isdataset=True)
        val_loss = evaluate(model, test_dataset,
                            loss_fn=eval_loss_fn, device=device,
                            isdataset=True)
        epoch_time = time.time()-epoch_start_time
        print('-'*89)
        logger.info(
                f'| end of epoch {epoch:3d} | '
                f'epoch time: {epoch_time:5.2f}s | '
                f'valid loss {val_loss:5.2f} | '
        )
        print('-'*89)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        if optimize_type == 'sgd':
            scheduler.step()

    # eval model
    # TODO: change loader back to test_loader!!!
    test_loss = evaluate(best_model, test_dataset,
                         loss_fn=eval_loss_fn, device=device,
                         isdataset=True)
    print('-'*89)
    logger.info(
        f'| End of training | test loss {test_loss:5.2f} | '
    )
    print('-'*89)

    # generate and plot projections
    viz_dataset = test_dataset
    chart_path = f'charts/{model_type}_{test_ticker}.png'
    project(best_model, viz_dataset, loss_fn=train_loss_fn, device=device,
            title=test_ticker, should_show=True, savepath=chart_path,
            logger_name=logger_name)


if __name__ == '__main__':
    main()
