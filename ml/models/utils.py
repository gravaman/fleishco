from os import listdir
from os.path import join, isfile, basename, splitext
import torch
import numpy as np
import matplotlib.pyplot as plt


def path_to_ticker(ticker_path):
    return basename(splitext(ticker_path)[0]).upper()


def list_files(root_dir):
    paths = [join(root_dir, p) for p in listdir(root_dir)
             if isfile(join(root_dir, p))]
    return paths


def pos_encodings(T, P, D_embed):
    """
    Builds positional encoding for time frame T given periodicity P

    params
    T (int): number of time steps
    P (int): number of periods to encode
    D_embed (int): number of dimensions for embedding

    returns
    encodings (T, D_embed): tensor of positional encodings
    """
    positions = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    encodings = torch.sin(positions*2*np.pi/P)
    encodings = encodings.repeat((1, D_embed))
    return encodings


def pos_encodings_attn(T, D_embed):
    """Positional encoding based on attention is all you need paper"""
    positions = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    encodings = torch.zeros((T, D_embed))
    encodings[:, 0::2] = torch.sin(
        positions/torch.pow(1000, torch.arange(0,
                                               D_embed, 2,
                                               dtype=torch.float32)/D_embed))
    encodings[:, 1::2] = torch.cos(
        positions/torch.pow(1000, torch.arange(1,
                                               D_embed, 2,
                                               dtype=torch.float32)/D_embed))
    return encodings


def line_plot(x, Y, labels=None, title=None, should_show=True, savepath=None,
              xlabel=None, ylabel=None):
    """
    Generates line plot for given series.

    params
    x (1D array): x values
    Y (nd array): y series
    labels (str array): series labels
    title (str): title
    should_show (bool): show indicator
    savepath (str): location for storing chart
    """
    # generate plot
    fig, ax = plt.subplots()

    if labels is None:
        labels = [f'series {i}' for i in range(len(Y))]

    for y, label in zip(Y, labels):
        ax.plot(x, y, label=label)

    ax.legend()
    if title:
        ax.set_title(title)

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)

    if should_show:
        plt.show()
