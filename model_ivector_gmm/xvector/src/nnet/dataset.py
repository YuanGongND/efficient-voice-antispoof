import numpy as np
import torch
from torch.utils import data

"""
load data from ivector or xvector
"""


def parse_raw(data_file):
    """
    parse raw (i/x)vector data_file
    """
    f = open(data_file, 'r')
    utt2label_vec = {}
    for line in f:
        items = line.strip().split('  ')
        if items[0][0] != '0' and items[0][0] != '1':
            raise ValueError('Invalid naming: ' + items[0][0])
        utt = items[0][2:]
        label = int(items[0][0])
        vector = np.fromstring(items[1].strip('[]'), sep=' ')
        utt2label_vec[utt] = (label, vector)
    f.close()
    return utt2label_vec


class VectorDataset(data.Dataset):
    """PyTorch dataset that reads ivector or xvector
    """
    def __init__(self, data_file):
        'Initialization'
        """
        data_file: ivector or xvector data file in txt
        sample line: 0_T_1001509  [ -1.6128 -0.1935 0.7...]
        """
        self.data_file = data_file
        self.utt2label_vec = parse_raw(data_file)
        self.key_list = sorted(self.utt2label_vec.keys())

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.key_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        utt_id = self.key_list[index]
        # Load data and get label
        vec = self.utt2label_vec[utt_id][1]
        tensor = torch.tensor(np.expand_dims(vec, axis=0)).float()
        X = tensor
        y = self.utt2label_vec[utt_id][0]

        return utt_id, X, y
