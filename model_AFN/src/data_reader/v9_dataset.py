import numpy as np
import torch
from torch.utils import data
import adv_kaldi_io as ako
import kaldi_io as ko

"""
For CNN+GRU where it loads one utterance at a time 
"""

class SpoofDataset(data.Dataset):
    """PyTorch dataset that reads kaldi feature
    """
    def __init__(self, scp_file, utt2label_file):
        'Initialization'
        self.scp_file  = scp_file
        self.utt2label = ako.read_key_label(utt2label_file)
        self.key_list  = ako.read_all_key(scp_file)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.key_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        utt_id = self.key_list[index]
        # Load data and get label
        tensor = ako.read_mat_key(self.scp_file, utt_id)
        X = np.expand_dims(tensor, axis=0)
        X = X[:,128:,:]
        y = self.utt2label[utt_id]
        
        return utt_id, X, y
