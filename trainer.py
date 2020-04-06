import logging
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from StockDataset import StockDataset, StockBatch
from RNN import RNN
from LSTM import LSTM
from utils import list_files


##############################
# TODO
# build transcoder
# pull data from postgres
# add CLI parser
# visualizations
##############################

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
            mu, std = dataset.get_stats()
        else:
            # set stats for validation and testing data based on training
            dataset.mu, dataset.std = mu, std

        shapes.append(dataset.size())
        loaders.append(DataLoader(dataset, batch_size=tickers_per_batch,
                                  shuffle=True, pin_memory=pmem,
                                  num_workers=num_workers,
                                  collate_fn=StockBatch))
    return shapes, loaders


def test_dataloader(loader, logger_name):
    logger = logging.getLogger(logger_name)
    logger.info('launching dataloader test')
    for i, batch in enumerate(loader):
        logger.info(f'batch idx: {i} X size: {batch.X.size()}'
                    f'y size: {batch.y.size()}')
    logger.info('dataloader test complete')


def train(model, loader, optimizer, loss_fn, device, logger_name,
          grad_norm_max=0.5, log_rate=0.25):
    logger = logging.getLogger(logger_name)
    model.train()
    total_loss = 0.0
    start_time = time.time()
    total_steps = len(loader)
    log_interval = int(np.floor(len(loader)*log_rate))
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        y_pred = model(batch.X.to(device))
        loss = loss_fn(y_pred, batch.y.to(device))
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


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            y_pred = model(batch.X.to(device))
            total_loss += loss_fn(y_pred, batch.y.to(device)).item()
            total_count += len(batch.X)
    return total_loss / total_count


def main():
    # available types: rnn, lstm
    model_type = 'rnn'

    assert model_type in ['rnn', 'lstm'], f'unknown model_type: {model_type}'

    # logger setup
    logger = setup_logger(model_type)

    # device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f'device type: {device.type}')

    # dataloaders setup
    ticker_dir = 'data/equities'
    tickers_per_batch = 4
    mbatch_size = 400  # minibatch size
    num_workers = 4  # number of data loader workers
    pmem = True if device.type == 'cuda' else False

    datashapes, (train_loader, val_loader, test_loader) = setup_dataloaders(
        ticker_dir,
        tickers_per_batch=tickers_per_batch,
        mbatch_size=mbatch_size,
        num_workers=num_workers, pmem=pmem)

    # model setup
    train_xshape, train_yshape = datashapes[0]
    D_in = train_xshape[2]  # number of input features
    H = 10  # number of hidden state features
    D_out = train_yshape[1]  # number of output features

    if model_type == 'rnn':
        model = RNN(D_in, H, D_out, device=device).to(device)
    elif model_type == 'lstm':
        model = LSTM(D_in, H, D_out, device=device).to(device)

    # optimizer setup
    optimize_type = 'sgd'
    lr = 0.001  # learning rate
    if optimize_type == 'adam':
        wd = 0.99  # weight decay
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
    epochs = 3
    best_model = None
    best_val_loss = float('inf')
    train_loss_fn = nn.MSELoss()
    eval_loss_fn = nn.MSELoss(reduction='sum')
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train(model, train_loader, logger_name=model_type, device=device,
              optimizer=optimizer, loss_fn=train_loss_fn)
        val_loss = evaluate(model, val_loader,
                            loss_fn=eval_loss_fn, device=device)
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
    test_loss = evaluate(best_model, test_loader,
                         loss_fn=eval_loss_fn, device=device)
    print('-'*89)
    logger.info(
        f'| End of training | test loss {test_loss:5.2f} | '
    )
    print('-'*89)


if __name__ == '__main__':
    main()
