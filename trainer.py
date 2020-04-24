from os.path import join
from datetime import date, timedelta
import logging
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from db.db_query import get_corptx_ids, get_fwd_credit_tx_ids
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
    'NVDA', 'JNPR', 'ADI', 'ADBE', 'STX', 'AVT', 'ARW', 'KLAC', 'A', 'NTAP',
    'VRSK', 'TECD', 'KEYS', 'CSCO', 'AMD', 'CRM'
]

LEISURE = ['FUN', 'RCL', 'EPR']
RETAIL = [
    'KSS', 'COST', 'MAT', 'ORLY', 'DG', 'HD', 'BBY', 'GPS', 'RL',
    'TIF', 'ROST', 'BBBY', 'HAS', 'DDS', 'WMT',
    'KR', 'AZO', 'WHR', 'AAP'
]
RESTAURANTS = ['SBUX', 'MCD', 'DRI']
CONSUMER = LEISURE + RETAIL + RESTAURANTS

LODGING = ['H']
HOMEBUILDERS = ['LEN', 'TOL', 'KBH', 'PHM', 'BZH', 'MDC']
SHOPPING_CENTER_REITS = ['REG', 'KIM']
DATA_CENTER_REITS = ['DLR', 'AMT']
TRIPLE_NET_REITS = ['O', 'SRC']
REAL_ESTATE = LODGING + HOMEBUILDERS + SHOPPING_CENTER_REITS + \
    DATA_CENTER_REITS + TRIPLE_NET_REITS

TEST_TECH = ['TECD', 'KEYS']
TEST_RE = ['EPR', 'RCL']

OUTPUT_DIR = 'output/models'
OUTPUT_CORP_TX_IDS_DIR = 'output/corp_tx_ids'


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


def setup_dataloaders(tickers, dt_bounds, tick_limit=None, ed='2019-12-31',
                      T=8, release_window=720, mbatch_size=256, num_workers=4,
                      pin_memory=False, logger_name=None,
                      should_load_ids=False, should_save_ids=False,
                      should_load_stats=False, should_save_stats=False,
                      days_lower=30, days_upper=60):
    logger = logging.getLogger(logger_name)

    # fetch corp_tx ids for each dataset
    logger.info(f'fetching corp_tx ids | tickers {len(tickers)}')
    splits = []
    if should_load_ids:
        for p in ['train.csv', 'val.csv', 'test.csv']:
            splits.append(np.loadtxt(
                open(join(OUTPUT_CORP_TX_IDS_DIR, p), 'rb'),
                delimiter=',').astype(int))
    else:
        # get query period splits
        periods = [len(b) for b in dt_bounds]
        logger.info(f'periods {periods} '
                    f'| train {dt_bounds[0]} '
                    f'| val {dt_bounds[1]} '
                    f'| test {dt_bounds[2]} ')

        for bounds in dt_bounds:
            ids = []
            subtotal = 0
            for (sd, ed) in bounds:
                period_ids = get_corptx_ids(tickers,
                                            release_window=release_window,
                                            release_count=T, limit=None,
                                            tick_limit=tick_limit,
                                            sd=sd, ed=ed)
                fwd_period_ids = get_fwd_credit_tx_ids(period_ids.tolist(),
                                                       days_lower, days_upper)
                ids.append(fwd_period_ids)
                subtotal += len(fwd_period_ids)
                logger.info(f'{sd} {ed} ids {len(fwd_period_ids)} '
                            f'| split subtotal {subtotal}')

            ids = np.concatenate(ids, axis=0)[:, 0]
            splits.append(ids)

    logger.info(f'finished getting corp_tx ids '
                f'| train {len(splits[0])} '
                f'| val {len(splits[1])} '
                f'| test {len(splits[2])} '
                f'| total {sum(map(len, splits))}')

    if should_save_ids and not should_load_ids:
        for p, ids in zip(['train.csv', 'val.csv', 'test.csv'], splits):
            np.savetxt(join(OUTPUT_CORP_TX_IDS_DIR, p), ids.astype(int),
                       fmt='%i', delimiter=',')
        logger.info(f'corp_tx ids saved to {OUTPUT_CORP_TX_IDS_DIR}')

    # data loaders
    logger.info('building datasets from corp_tx ids')

    datasets, loaders = [], []
    names = ['training', 'validation', 'testing']
    for i, ticker_ids in enumerate(splits):
        logger.info(f'building {names[i]} dataset | txids {len(ticker_ids)}')
        if i == 0:
            dataset = CreditDataset(tickers,
                                    split_type=names[i],
                                    T=T,
                                    standardize=True,
                                    should_load=should_load_stats,
                                    should_save=should_save_stats,
                                    txids=ticker_ids)
            standard_stats = dataset.standard_stats
        else:
            # set stats for validation and testing data based on training
            dataset = CreditDataset(tickers,
                                    split_type=names[i],
                                    T=T,
                                    standardize=True,
                                    txids=ticker_ids,
                                    should_load=False,
                                    should_save=False,
                                    standard_stats=standard_stats)
        logger.info(f'finished building {names[i]} dataset '
                    f'| txs {len(dataset)}')
        datasets.append(dataset)
        loaders.append(DataLoader(dataset, batch_size=mbatch_size,
                                  shuffle=True, pin_memory=pin_memory,
                                  num_workers=num_workers))

    return datasets, loaders


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
            cur_loss = total_loss / i
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'| step {i:5d}/{total_steps:5d} '
                f'| lr: {lr:0.4f} '
                f'| ms/step: {elapsed*1000/log_interval:5.2f} '
                f'| yield loss: {cur_loss:5.2f} '
            )
            start_time = time.time()
    return total_loss/len(loader)


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
    # make deterministic for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # available types: rnn, lstm, tranformer
    model_type = 'transformer'

    assert model_type in MODEL_TYPES, f'unknown model_type: {model_type}'

    # save both best and trained models
    save_model = True

    # logger setup
    logger_name = model_type
    logger = setup_logger(logger_name)

    # device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device('cpu')
    logger.info(f'device type: {device.type}')

    # dataloaders setup
    tickers = TECH+CONSUMER+REAL_ESTATE
    tick_limit = 1000
    T = 8
    release_window = 720
    mbatch_size = 1024
    num_workers = 12
    pin_memory = True if device.type == 'cuda' else False

    # date boundary setup
    train_periods = 6
    dts = pd.date_range(end='2017-12-31', periods=train_periods+1,
                        freq='Y')
    dts = dts.map(lambda x: x.strftime('%Y-%m-%d')).values
    train_bounds = list(zip(dts[:-1], dts[1:]))
    days_upper = 60
    val_lb = date.fromisoformat(dts[-1])
    val_lb = (val_lb+timedelta(days=days_upper)).strftime('%Y-%m-%d')
    val_bounds = [(val_lb, '2018-12-31')]
    test_bounds = [('2018-12-31', '2019-12-31')]
    dt_bounds = [train_bounds, val_bounds, test_bounds]

    datasets, loaders = setup_dataloaders(tickers, dt_bounds,
                                          T=T,
                                          release_window=release_window,
                                          tick_limit=tick_limit,
                                          mbatch_size=mbatch_size,
                                          num_workers=num_workers,
                                          pin_memory=pin_memory,
                                          logger_name=logger_name,
                                          days_upper=days_upper,
                                          should_load_ids=True,
                                          should_save_ids=False,
                                          should_load_stats=True,
                                          should_save_stats=False)
    train_loader, val_loader, test_loader = loaders
    train_stats = datasets[0].standard_stats

    # check sample shapes of each dataset
    for ds_type, ds in zip(['train', 'val', 'test'], datasets):
        sample = next(iter(ds))
        logger.info(f'{ds_type} size: {sample[0].size()} '
                    f'{sample[1].size()} '
                    f'{sample[2].size()}')

    for lt, loader in zip(['train', 'val', 'test'], loaders):
        logger.info(f'type {lt} | '
                    f'| batches {len(loader)} '
                    f'| minibatch size {mbatch_size} '
                    f'| total records {len(loader)*mbatch_size}')
        if lt == 'train':
            s = train_stats.ctx_stats
            logger.info(f'type {lt} | '
                        f'| mu: {np.exp(s.target[0]):5.2f} '
                        f'| std: {np.exp(s.target[1]):5.2f} ')
    # model setup
    # train_xshape, train_yshape = shapes[0]
    # D_in = train_xshape[2]  # number of input features
    D_in = train_stats.fin_stats.mu.shape[0]  # from model (23 AAPL)
    D_ctx = train_stats.ctx_stats.mu.shape[0]-1  # from model
    D_out = 1  # from model
    # D_out = train_yshape[1]  # number of output features
    if model_type == 'transformer':
        D_embed = 64  # embedding dimension
        # Q = train_xshape[1]  # query matrix dimesion (T)
        # V = train_xshape[1]  # value matrix dimension (T)
        Q = 8  # from model (10 equities)
        V = 8  # from model (10 equities)
        H = 2  # number of heads
        N = 2  # number of encoder and decoder stacks
        attn_size = None  # local attention mask size
        dropout = 0.6  # dropout pct
        P = 8  # periodicity of input data (equities 5)
        model = Transformer(D_in, D_embed, D_ctx, D_out, Q, V, H, N,
                            local_attn_size=attn_size, dropout=dropout,
                            P=P, device=device).to(device)
    elif model_type == 'lstm':
        H = 10  # number of hidden state features
        model = LSTM(D_in, H, D_ctx, D_out, device=device).to(device)
    elif model_type == 'rnn':
        H = 10  # number of hidden state features
        model = RNN(D_in, H, D_ctx, D_out, device=device).to(device)

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
    epochs = 10
    best_model = None
    best_val_loss = float('inf')
    train_stats = datasets[0].standard_stats
    train_loss_fn = MYELoss(standard_stats=train_stats.ctx_stats)
    eval_loss_fn = MYELoss(standard_stats=train_stats.ctx_stats)
    TL = np.zeros(epochs)
    L = np.zeros(epochs)
    logger.info(f'starting training')
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train_loss = train(model, train_loader, logger_name=logger_name,
                           device=device, optimizer=optimizer,
                           loss_fn=train_loss_fn, pin_memory=pin_memory)
        TL[epoch-1] = train_loss
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
                f'train loss {train_loss:5.2f} | '
                f'valid loss {val_loss:5.2f} | '
        )
        print('-'*89)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        if optimize_type == 'sgd':
            scheduler.step()

    logger.info(f'finished training')

    # eval model on test data
    test_loss = evaluate(best_model, test_loader,
                         loss_fn=eval_loss_fn, device=device,
                         logger_name=logger_name,
                         pin_memory=pin_memory)
    test_loss = test_loss.item()

    train_stats = datasets[0].standard_stats
    y_mu_train, y_std_train = np.exp(train_stats.target)

    val_stats = datasets[1].get_stats(should_load=True, should_save=False)
    y_mu_val, y_std_val = np.exp(val_stats.target)

    test_stats = datasets[2].get_stats(should_load=True, should_save=False)
    y_mu_test, y_std_test = np.exp(test_stats.target)

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
    line_plot(np.arange(1, epochs+1), [TL, L],
              labels=['training', 'validation'],
              title=f'{model_type} training progress', should_show=True,
              xlabel='epoch', ylabel='mean yield loss',
              savepath=loss_path)

    if save_model:
        torch.save(model.state_dict(),
                   join(OUTPUT_DIR, f'{model_type}.pt'))
        torch.save(best_model.state_dict(),
                   join(OUTPUT_DIR, f'best_{model_type}.pt'))


if __name__ == '__main__':
    main()
