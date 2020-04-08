from os import listdir
from os.path import join, isfile
import torch
import numpy as np


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
