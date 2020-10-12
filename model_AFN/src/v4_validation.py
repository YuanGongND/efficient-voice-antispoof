import numpy as np
import logging
from timeit import default_timer as timer
from scipy.optimize import fmin_l_bfgs_b, basinhopping
import torch
import torch.nn.functional as F
from .v1_metrics import compute_eer
from .data_reader.vND_dataset import parse_label


"""
utterance-based validation without stochastic search for threshold
important: EER does not need a threshold.
"""

# Get the same logger from main"
logger = logging.getLogger("anti-spoofing")


def validation(args, model, device, val_loader, val_utt2label, rnn=False):
    logger.info("Starting Validation")
    val_loss, val_scores = compute_loss(model, device, val_loader, rnn)
    val_preds, val_labels = utt_scores(val_scores, val_utt2label)
    val_eer = compute_eer(val_labels, val_preds)

    logger.info('===> Validation set: Average loss: {:.4f}\tEER: {:.4f}\n'.format(
                val_loss, val_eer))
    return val_loss, val_eer


def utt_scores(scores, utt2label):
    """return predictions and labels per utterance
    """
    utt2label = parse_label(utt2label)

    preds, labels = [], []
    for key, value in scores.items():
        # print(key.encode("latin-1"), utt2label[key])
        preds.append(value)
        labels.append(utt2label[key])

    return np.array(preds), np.array(labels)


def compute_loss(model, device, data_loader, rnn):
    model.eval()
    loss = 0
    scores = {}

    with torch.no_grad():
        for id_list, data, target in data_loader:
            data, target = data.to(device), target.to(device)
            target = target.view(-1, 1).float()
            if rnn is True:
                model.hidden = model.init_hidden(data.size()[0])  # clear out the hidden state of the LSTM
            output = model(data)[0]
            loss += F.binary_cross_entropy(output, target, size_average=False)
            for i, id in enumerate(id_list):
                scores[id] = output[i].data.cpu().numpy()

    loss /= len(data_loader.dataset)  # average loss

    return loss, scores
