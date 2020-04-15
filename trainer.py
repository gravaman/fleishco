import logging
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from db.db_query import get_corptx_ids
from ml.models.StockDataset import (
    TickerDataset
)
from ml.models.CreditDataset import CreditDataset
from ml.models.RNN import RNN
from ml.models.LSTM import LSTM
from ml.models.Transformer import Transformer
from ml.models.MYELoss import MYELoss
from ml.models.utils import line_plot


##############################
# TODO
# add CLI parser
##############################

# constants
MODEL_TYPES = ['rnn', 'lstm', 'transformer']
TECH = [
    'AAPL', 'MSFT', 'INTC', 'IBM', 'QCOM', 'ORCL', 'TXN', 'MU', 'AMZN', 'GOOG',
    'NVDA', 'JNPR', 'ADI', 'ADBE', 'STX', 'AVT', 'ARW', 'KLAC', 'NTAP',
    'VRSK', 'TECD', 'MRVL', 'KEYS'
]


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


def setup_dataloaders(tickers, release_window, T, limit=None, mbatch_size=50,
                      num_workers=4, pin_memory=False, train_split=0.7,
                      val_split=0.2):
    # data splits
    test_split = 1-train_split-val_split
    assert test_split > 0, 'train_split+val_split must be less than 1'

    idxs = get_corptx_ids(tickers,
                          release_window=release_window,
                          release_count=T,
                          limit=limit)
    data_size = len(idxs)
    train_size = int(np.floor(data_size*train_split))
    val_size = int(np.floor(data_size*val_split))

    train_idxs = idxs[:train_size]
    val_idxs = idxs[train_size:train_size+val_size]
    test_idxs = idxs[train_size+val_size:]

    # data loaders
    datasets, loaders = [], []
    for i, ticker_idxs in enumerate([train_idxs, val_idxs, test_idxs]):
        if i == 0:
            dataset = CreditDataset(tickers,
                                    T=T,
                                    standardize=True,
                                    txids=ticker_idxs)
            standard_stats = dataset.standard_stats
        else:
            # set stats for validation and testing data based on training
            dataset = CreditDataset(tickers,
                                    T=T,
                                    standardize=True,
                                    txids=ticker_idxs,
                                    standard_stats=standard_stats)
        datasets.append(dataset)
        loaders.append(DataLoader(dataset, batch_size=mbatch_size,
                                  shuffle=True, pin_memory=pin_memory,
                                  num_workers=num_workers))

    return datasets, loaders


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
          grad_norm_max=0.5, log_rate=0.25, pin_memory=False):
    logger = logging.getLogger(logger_name)
    model.train()
    total_loss = 0.0
    start_time = time.time()
    total_steps = len(loader)
    log_interval = max(int(np.floor(len(loader)*log_rate)), 1)
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        X_fin, X_ctx, y = [b.to(device, non_blocking=pin_memory)
                           for b in batch]
        y_pred = model(X_fin, X_ctx)
        mse_loss, mye_loss = loss_fn(y_pred, y)
        mse_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       grad_norm_max)
        optimizer.step()

        total_loss += mye_loss.item()
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'| step {i:5d}/{total_steps:5d} '
                f'| lr: {lr:0.4f} '
                f'| ms/step: {elapsed*1000/log_interval:5.2f} '
                f'| yield loss: {cur_loss:5.2f} '
            )
            total_loss = 0.0
            start_time = time.time()


def evaluate(model, loader, loss_fn, device, logger_name,
             pin_memory=False):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            X_fin, X_ctx, y = [b.to(device, non_blocking=pin_memory)
                               for b in batch]
            y_pred = model(X_fin, X_ctx)
            _, mye = loss_fn(y_pred, y)
            total_loss += mye

    return total_loss / len(loader)


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
    loss_stats (tuple floats): avg and standard deviation of losses
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

    loss_stats = (avg_loss, std_loss)
    return loss_stats


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
    tickers = TECH
    mbatch_size = 1024
    max_data_size = (190000 // mbatch_size)*mbatch_size
    num_workers = 12  # number of data loader workers
    pin_memory = True if device.type == 'cuda' else False
    T = 8  # financial periods to pull
    release_window = T*90+10  # periods*filing req + buffer
    datasets, loaders = setup_dataloaders(tickers=tickers,
                                          release_window=release_window,
                                          T=T, limit=max_data_size,
                                          mbatch_size=mbatch_size,
                                          num_workers=num_workers,
                                          pin_memory=pin_memory)
    train_loader, val_loader, test_loader = loaders
    train_stats = datasets[0].standard_stats

    # check sample shapes of each dataset
    for ds_type, ds in zip(['train', 'val', 'test'], datasets):
        sample = next(iter(ds))
        logger.info(f'{ds_type} size: {sample[0].size()} '
                    f'{sample[1].size()} '
                    f'{sample[2].size()}')

    logger.info(f'tickers: {tickers}')
    for lt, loader in zip(['train', 'val', 'test'], loaders):
        logger.info(f'type {lt} | '
                    f'| batches {len(loader)} '
                    f'| minibatch size {mbatch_size} '
                    f'| total records {len(loader)*mbatch_size}')
        if lt == 'train':
            s = train_stats.ctx_stats
            logger.info(f' type {lt} | '
                        f'| mu: {s.target[0]:5.2f} '
                        f'| std: {s.target[1]:5.2f} ')

    # model setup
    # train_xshape, train_yshape = shapes[0]
    # D_in = train_xshape[2]  # number of input features
    D_in = train_stats.fin_stats.mu.shape[0]  # from model (23 AAPL)
    D_ctx = 14  # from model (only for corp_txs)
    D_out = 1  # from model
    # D_out = train_yshape[1]  # number of output features
    if model_type == 'transformer':
        D_embed = 512  # embedding dimension
        # Q = train_xshape[1]  # query matrix dimesion (T)
        # V = train_xshape[1]  # value matrix dimension (T)
        Q = 8  # from model (10 equities)
        V = 8  # from model (10 equities)
        H = 4  # number of heads
        N = 6  # number of encoder and decoder stacks
        attn_size = None  # local attention mask size
        dropout = 0.3  # dropout pct
        P = 4  # periodicity of input data (equities 5)
        model = Transformer(D_in, D_embed, D_ctx, D_out, Q, V, H, N,
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
    lr = 0.00001  # learning rate
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
    epochs = 1
    best_model = None
    best_val_loss = float('inf')
    train_stats = datasets[0].standard_stats
    train_loss_fn = MYELoss(standard_stats=train_stats.ctx_stats)
    eval_loss_fn = MYELoss(standard_stats=train_stats.ctx_stats)
    L = np.zeros(epochs)
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train(model, train_loader, logger_name=logger_name, device=device,
              optimizer=optimizer, loss_fn=train_loss_fn,
              pin_memory=pin_memory)
        val_loss = evaluate(model, val_loader,
                            loss_fn=eval_loss_fn, device=device,
                            logger_name=logger_name,
                            pin_memory=pin_memory)
        L[epoch-1] = val_loss
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

    # eval model on test data
    test_loss = evaluate(best_model, test_loader,
                         loss_fn=eval_loss_fn, device=device,
                         logger_name=logger_name,
                         pin_memory=pin_memory)
    test_loss = test_loss.item()

    train_stats = datasets[0].standard_stats
    y_mu_train, y_std_train = train_stats.target

    val_stats = datasets[1].get_stats()
    y_mu_val, y_std_val = val_stats.target

    test_stats = datasets[2].get_stats()
    y_mu_test, y_std_test = test_stats.target

    print('-'*89)
    logger.info(
        f'| End of training | test yield loss {test_loss:5.2f} '
        f'| train mu {y_mu_train:5.2f} std {y_std_train:5.2f} '
        f'| val mu {y_mu_val:5.2f} std {y_std_val:5.2f} '
        f'| test mu {y_mu_test:5.2f} std {y_std_test:5.2f} '
    )
    print('-'*89)

    # plot training loss
    loss_path = f'ml/charts/{model_type}_training_loss.png'
    line_plot(np.arange(1, epochs+1), [L], labels=['validation'],
              title='training yield loss', should_show=True,
              savepath=loss_path)

    # generate and plot projections
    if False:
        chart_path = f'ml/charts/{model_type}_IG_Tech.png'
        project(best_model, test_loader, loss_fn=train_loss_fn, device=device,
                title='IG Tech', should_show=True, savepath=chart_path,
                logger_name=logger_name)


if __name__ == '__main__':
    main()
