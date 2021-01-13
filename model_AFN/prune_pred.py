# Utilities
import argparse
import random
import time
import os
import sys
from timeit import default_timer as timer

# Libraries
import numpy as np

# Torch
import torch
from torch.utils import data

# Customized Imports
from src.v1_logger import setup_logs
from src.data_reader.v0_dataset import SpoofDataset
from src.v4_prediction import prediction
from src.attention_neuro.simple_attention_network import (  # noqa
    AttenResNet4,
    AttenResNet4Deform,  # debug use only
    AttenResNet4Deform_512,  # debug use only
    AttenResNet4DeformAll
)

# Network compression Imports
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from acceleration_compression.pruning import NetworkPruning  # noqa

##############################################################
# Control Center and Hyperparameter
M = 1091
rnn = False  # rnn
TEST_BATCH_SIZE = 1
atten_channel = 16
atten_activation = 'sigmoid'


def get_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove('temp.p')
    return size


def apply_net_prune(model, logger, percentage, prune_method='L1Unstructured'):
    net_prune = NetworkPruning(model, percentage=percentage, prune_method=prune_method)
    logger.info(
        "#### orignal model size (MB): {}"
        .format(net_prune.print_size_of_model(model))
    )
    logger.info(
        '#### non-zero params before pruning: {}'
        .format(net_prune.get_num_parameters(model, is_nonzero=True))
    )
    model = net_prune.pruning()
    logger.info('===> apply [{}] PRUNING with percentage: {}'.format(prune_method, percentage))  # noqa
    logger.info(
        "#### pruned model size (MB): {}"
        .format(net_prune.print_size_of_model(model))
    )
    logger.info(
        '#### non-zero params after pruning: {}'
        .format(net_prune.get_num_parameters(model, is_nonzero=True))
    )
    return model


def main():
    ##############################################################
    # Settings
    parser = argparse.ArgumentParser(description='Model AFN')
    parser.add_argument('--eval-dir',
                        help='eval feature dir')
    parser.add_argument('--eval-utt2label',
                        help='train utt2label')
    parser.add_argument('--model-path',
                        help='path to the pretrained model')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--logging-dir', required=True,
                        help='model save directory')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--seg', default=None,
                        help='seg method')
    parser.add_argument('--seg-win', type=int,
                        help='segmented window size')
    parser.add_argument('--prune-method', required=True,
                        help='pruning method: L1Unstructured, RandomUnstructured')
    parser.add_argument('--prune-pct', type=float, required=True,
                        help='percentage for pruning')
    args = parser.parse_args()

    torch.cuda.empty_cache()

    # Init model & Setup logs
    if args.seg is None:
        model = AttenResNet4(atten_activation, atten_channel, size1=(257, M))
        run_name = "prune_pred-AFN4-1091-orig" + time.strftime("-%Y_%m_%d")
    else:
        if args.seg_win not in (64, 128, 256, 512):
            raise ValueError("Invalid segment window! Must be 64, 128, 256, or 512")
        model = AttenResNet4DeformAll(atten_activation, atten_channel, size1=(257, args.seg_win))  # noqa
        run_name = "prune_pred-AFN4De-" + str(args.seg_win) + "-" + args.seg + time.strftime("-%Y_%m_%d")  # noqa
    logger = setup_logs(args.logging_dir, run_name)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info("use_cuda is {}".format(use_cuda))

    # Setting random seeds for reproducibility.
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # CUDA determinism

    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)

    ##############################################################
    # Loading the dataset & the best model
    params = {'num_workers': 0,
              'pin_memory': False,
              'worker_init_fn': np.random.seed(args.seed)} if use_cuda else {}

    logger.info('===> loading eval dataset: ' + args.eval_utt2label)
    eval_set = SpoofDataset(args.eval_dir, args.eval_utt2label)
    eval_loader = data.DataLoader(
        eval_set,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        **params
    )  # set shuffle to False
    logger.info('===> loading best model for prediction: ' + args.model_path)
    checkpoint = torch.load(
        os.path.join(args.model_path)
    )
    model.load_state_dict(checkpoint['state_dict'])

    ##############################################################
    # apply network pruning & prediction
    model = apply_net_prune(
        model=model,
        logger=logger,
        percentage=float(args.prune_pct),
        prune_method=args.prune_method
    )

    t_start_eval = timer()
    eval_loss, eval_eer = prediction(args, model, device, eval_loader, args.eval_utt2label, rnn)  # noqa
    t_end_eval = timer()
    logger.info("#### Total prediction time: {}".format(t_end_eval - t_start_eval))
    ###########################################################
    logger.info("################## Success #########################\n\n")


if __name__ == '__main__':
    main()
