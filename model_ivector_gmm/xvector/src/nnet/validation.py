import numpy as np
import logging
import torch
import torch.nn.functional as F
from nnet.metrics import compute_eer


"""
utterance-based validation without stochastic search for threshold
important: EER does not need a threshold.
"""

# Get the same logger from main"
logger = logging.getLogger("MLP")


def validation(args, model, device, val_loader, val_utt2label):
    logger.info("Starting Validation")
    val_loss, val_scores = compute_loss(model, device, val_loader)
    val_preds, val_labels = utt_scores(val_scores, val_utt2label)
    val_eer = compute_eer(val_labels, val_preds)

    logger.info('===> Validation set: Average loss: {:.4f}\tEER: {:.4f}\n'.format(
                val_loss, val_eer))
    return val_loss, val_eer


def compute_loss(model, device, data_loader):
    model.eval()
    loss = 0
    scores = {}

    with torch.no_grad():
        for id_list, vec, target in data_loader:
            vec, target = vec.to(device), target.to(device)
            target = target.view(-1, 1).float()
            output = model(vec).reshape(-1, 1)
            loss += F.binary_cross_entropy(output, target, size_average=False)
            for i, id in enumerate(id_list):
                scores[id] = output[i].cpu().numpy()

    loss /= len(data_loader.dataset)  # average loss

    return loss, scores


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
