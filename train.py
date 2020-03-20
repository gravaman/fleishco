import argparse
from os.path import join
from os import environ
import warnings
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from CreditDataset import CreditDataset, Standardize, ToTensor
import models.MultilayerNN

# needs to be before torch tf import
warnings.filterwarnings('ignore', category=FutureWarning)
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from torch.utils.tensorboard import SummaryWriter  # noqa


# hyperparameters
parser = argparse.ArgumentParser(description='Fleishco CLI')

parser.add_argument('--lr', type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--model-type', choices=['multilayernn'],
                    help='which model to train/eval')
parser.add_argument('--hidden-dim', type=int,
                    help='number of hidden features/activations')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='number of batches between logging train status')
parser.add_argument('--log-id', default=1, metavar='N',
                    help='id for log directory')
args = parser.parse_args()

# visualizer
writer = SummaryWriter(f'runs/{args.model_type}_{args.log_id}')


# helpers
def collate(batch):
    x, y = batch[0]
    rows, cols = x.size()
    return (x.view(rows, 1, cols), y.view(rows, 1, 1))


def setup_data(batch_path, root_dir, **kwargs):
    datasets = []
    dataloaders = []
    for split in ['train', 'val', 'test']:
        dataset = CreditDataset(batch_path=batch_path,
                                root_dir=root_dir,
                                split=split,
                                transform=transforms.Compose([
                                    Standardize(params),
                                    ToTensor()
                                ]))
        datasets.append(dataset)
        dataloaders.append(DataLoader(dataset, shuffle=True,
                                      collate_fn=collate,
                                      **kwargs))
    return dataloaders


def train(epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data, targets = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            data, targets = data.cuda(), targets.cuda()

        optimizer.zero_grad()
        y_pred = model(data.float())
        loss = criterion(y_pred, targets.float())
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            val_loss = evaluate('val', n_batches=4)
            train_loss = loss.data
            examples_this_epoch = batch_idx * len(data)
            data_size = len(train_loader.dataset)*len(data)
            epoch_progress = 100. * batch_idx / len(train_loader)
            mu, std = params[0, -1], params[1, -1]
            train_yld_loss = std*(train_loss**0.5+mu)
            val_yld_loss = std*(val_loss**0.5+mu)
            msg = (
                f'Train Epoch: {epoch} '
                f'[{examples_this_epoch}/{data_size} '
                f'({epoch_progress:.0f}%)]\t'
                f'Train Loss: {train_yld_loss:.2f}%\t'
                f'Val Loss: {val_yld_loss:.2f}%'
            )
            print(msg)
            num_iter = epoch*len(train_loader)+batch_idx
            writer.add_scalar('Loss/Train', train_yld_loss,
                              global_step=num_iter)
            writer.add_scalar('Loss/Validation', val_yld_loss,
                              global_step=num_iter)


@torch.no_grad()
def evaluate(split, verbose=False, n_batches=None):
    model.eval()
    loss = 0
    n_examples = 0
    if split == 'val':
        loader = val_loader
    elif split == 'test':
        loader = test_loader
    else:
        raise ValueError(f'eval split must be val or test; got {split}')

    for batch_i, batch in enumerate(loader):
        data, target = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data.float())
        loss += criterion(output, target.float()).data.float()
        n_examples += data.size(0)
        if n_batches and (batch_i >= n_batches):
            break

    if n_batches is None:
        n_batches = len(loader)
    loss /= n_batches
    mu, std = params[0, -1], params[1, -1]
    yld_loss = std*(loss**0.5+mu)
    if verbose:
        msg = (
            f'\n{split} set: Yield mu: {mu:.2f}% std: {std:.2f}% \t '
            f'Average loss: {yld_loss:.2f}%'
        )
        print(msg)

    return loss


# setup hyperparameters
root_dir = 'data/bonds/batches'
batch_path = 'data/bonds/datasets/batch_paths.csv'
params_path = 'data/bonds/datasets/tx_params.csv'

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# get feature dimensions
tmppath = pd.read_csv(batch_path).iloc[0].values[0]
tmp = join(root_dir, tmppath)
feat_cnt = len(pd.read_csv(tmp, nrows=1, index_col=None).columns)-1
sample_size = [1, feat_cnt]

# setup data loaders and model
params = pd.read_csv(params_path).values
train_loader, val_loader, test_loader = setup_data(batch_path,
                                                   root_dir,
                                                   **kwargs)
model = models.MultilayerNN.MultilayerNN(sample_size, args.hidden_dim)
criterion = F.mse_loss
if args.cuda:
    model.cuda()
optimizer = optim.Adam(params=model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)


if __name__ == '__main__':
    # train model and evaluate results
    for epoch in range(1, args.epochs+1):
        train(epoch)

    evaluate('test', verbose=True)
    torch.save(model, args.model_type+'.pt')
