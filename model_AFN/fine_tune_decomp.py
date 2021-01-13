# This script onlu runs on the GPU machine, never use it on Pi or Jetson
# Utilities
import argparse
import random
import os
import sys
import inspect
from timeit import default_timer as timer

# Libraries
import numpy as np

# Torch
import torch
from torch.utils import data
import torch.optim as optim

# Custrom Imports
from src.v1_logger import setup_logs
from src.data_reader.v0_dataset import SpoofDataset
from src.v4_validation import validation
from src.v1_training import train, snapshot
from src.attention_neuro.simple_attention_network import (  # noqa
    AttenResNet4,
    AttenResNet4Deform,  # debug use only
    AttenResNet4Deform_512,  # debug use only
    AttenResNet4DeformAll
)
# network decomposition imports
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from acceleration_compression.decomposition import NetworkDecomposition  # noqa

##############################################################
############ Control Center and Hyperparameter ###############
feat_dim = 257
M = 1091
select_best = 'eer'  # eer or val
rnn = False  # rnn
batch_size = test_batch_size = 8
atten_channel = 16
atten_activation = 'sigmoid'


def load_model(model, model_path, freeze=False):
    """load pre-trained model"""
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model params is', model_params)

    return model


def get_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove('temp.p')
    return size


def apply_net_decomp(model, logger, rank, decomp_type='tucker'):
    net_decomp = NetworkDecomposition(model, rank=rank, decomp_type=decomp_type)
    logger.info(
        "#### orignal model size (MB): {}"
        .format(net_decomp.print_size_of_model(model))
    )
    logger.info(
        '#### non-zero params before decomp: {}'
        .format(net_decomp.get_num_parameters(model, is_nonzero=True))
    )
    model = net_decomp.decomposition()
    logger.info('apply [{}] DECOMP with rank: {}'.format(decomp_type, rank))
    logger.info(
        "#### decomp-ed model size (MB): {}"
        .format(net_decomp.print_size_of_model(model))
    )
    logger.info(
        '#### non-zero params after decomp: {}'
        .format(net_decomp.get_num_parameters(model, is_nonzero=True))
    )
    return model
##############################################################


def main():
    ##############################################################
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-dir', required=True,
                        help='train feature dir')
    parser.add_argument('--train-utt2label', required=True,
                        help='train utt2label')
    parser.add_argument('--validation-dir', required=True,
                        help='dev feature dir')
    parser.add_argument('--validation-utt2label', required=True,
                        help='dev utt2label')
    parser.add_argument('--model-path',
                        help='path to the pretrained model')
    parser.add_argument('--logging-dir', required=True,
                        help='model save directory')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--seg', default=None,
                        help='seg method')
    parser.add_argument('--seg-win', type=int, help='segmented window size')
    parser.add_argument('--rank', type=int, required=True,
                        help='rank value for decomposition: 2 or 4')
    parser.add_argument('--decomp-type', default='tucker',
                        help='decomposition type: tucker cp')
    args = parser.parse_args()

    torch.cuda.empty_cache()

    # Init model & Setup logs
    if args.seg is None:
        model = AttenResNet4(atten_activation, atten_channel, size1=(257, M))
        run_name = "fine_decomp-AFN4-" + str(M) + "-orig-rank_" + str(args.rank) + '-' + args.decomp_type  # noqa
    else:
        model = AttenResNet4DeformAll(atten_activation, atten_channel, size1=(257, args.seg_win))  # noqa
        run_name = "fine_decomp-AFN4De-" + str(args.seg_win) + "-" + args.seg + "-rank_" + str(args.rank) + '-' + args.decomp_type  # noqa
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
    model = load_model(model, args.model_path)

    # perform decomposition
    model = apply_net_decomp(
            model=model,
            logger=logger,
            rank=args.rank,
            decomp_type=args.decomp_type
    )
    model.to(device)
    ##############################################################
    # Loading the dataset and fine tune the compressed model
    params = {'num_workers': 0,
              'pin_memory': False,
              'worker_init_fn': np.random.seed(args.seed)} if use_cuda else {}

    logger.info('===> loading train and dev dataset')
    training_set = SpoofDataset(args.train_dir, args.train_utt2label)
    validation_set = SpoofDataset(args.validation_dir, args.validation_utt2label)
    train_loader = data.DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        **params
    )  # set shuffle to True
    validation_loader = data.DataLoader(
        validation_set,
        batch_size=test_batch_size,
        shuffle=False,
        **params
    )  # set shuffle to False

    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)  # noqa

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('#### Decomposed model summary below####\n {}\n'.format(str(model)))
    logger.info('===> Model total parameter: {}\n'.format(model_params))
    ###########################################################
    # Start training
    best_eer, best_loss = np.inf, np.inf
    early_stopping, max_patience = 0, 5  # early stopping and maximum patience

    total_train_time = []
    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()

        # Train and validate
        train(args, model, device, train_loader, optimizer, epoch, rnn)
        val_loss, eer = validation(
            args,
            model,
            device,
            validation_loader,
            args.validation_utt2label,
            rnn
        )
        scheduler.step(val_loss)
        # Save
        if select_best == 'eer':
            is_best = eer < best_eer
            best_eer = min(eer, best_eer)
            best_model = model
        elif select_best == 'val':
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            best_model = model
        snapshot(args.logging_dir, run_name, is_best, {
                'epoch': epoch + 1,
                'best_eer': best_eer,
                'state_dict': model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
        })
        # Early stopping
        if is_best == 1:
            early_stopping = 0
        else:
            early_stopping += 1
        end_epoch_timer = timer()
        logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args.epochs, end_epoch_timer - epoch_timer))  # noqa
        total_train_time.append(end_epoch_timer - epoch_timer)
        if early_stopping == max_patience:
            break
    logger.info("#### Avg. training+validation time per epoch: {}".format(np.average(total_train_time)))  # noqa
    ###########################################################
    logger.info(
        "#### fine-tuned decomp model size (MB): {}"
        .format(get_size_of_model(best_model))
    )
    model_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    logger.info(
        '#### non-zero params after fine-tuning: {}'
        .format(model_params)
    )
    logger.info("################## Done fine-tuning decomp model ######################")


if __name__ == '__main__':
    main()
