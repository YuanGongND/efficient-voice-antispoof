import os
import numpy as np
import torch
from torch.utils import data
# import src.data_reader.adv_kaldi_io as ako
# import kaldi_io as ko

"""
load data from torchaudio extracted features
"""

M = 1091


def parse_label(utt2label_file):
    """
    parse utt2label file
    whose 1st column is the fileID (without suffix) of the utterance,
    and the 2nd column is the label ('genuine' or 'spoof')
    """
    f = open(utt2label_file, 'r')
    utt2label = {}
    for line in f:
        item = line.strip().split(' ')
        utt2label[item[0]] = int(item[1])
    return utt2label


def load_feat(file_path):
    """
    load feature tensors from files and expand to fixed dims (M, 257)
    """
    tensor = torch.load(file_path)

    n_rep = M // tensor.shape[0]
    tail_dim = M % tensor.shape[0]
    tmp = tensor.expand(n_rep, tensor.shape[0], tensor.shape[1])
    head = tmp.reshape(M-tail_dim, tensor.shape[1])
    tail = tensor[0:tail_dim, :]

    return torch.cat([head, tail], dim=0)


class SpoofDataset(data.Dataset):
    """PyTorch dataset that reads torchaudio extracted feature
    """
    def __init__(self, feat_dir, utt2label_file):
        'Initialization'
        """
        key_list: all wave ids, List
        utt_id: specific utterance id str
        utt2label: namely: Dictionary
        """
        self.feat_dir = feat_dir
        self.utt2label = parse_label(utt2label_file)
        self.key_list = sorted(self.utt2label.keys())

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.key_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        utt_id = self.key_list[index]
        # Load data and get label
        feat_file = os.path.join(self.feat_dir, utt_id+'.pt')
        tensor = load_feat(feat_file)
        # TODO: change load feat to directly compute feat
        # using os.path.join(self.data_dir, utt_id+'.wav')
        tensor = torch.transpose(tensor, 0, 1)
        X = np.expand_dims(tensor, axis=0)
        y = self.utt2label[utt_id]

        return utt_id, X, y
