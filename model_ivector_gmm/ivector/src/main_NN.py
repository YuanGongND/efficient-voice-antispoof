# Utilities
import argparse
import random
import time
import os
from timeit import default_timer as timer

# Libraries
import numpy as np

# Torch
import torch
from torch.utils import data
import torch.optim as optim

# Customized Imports
from nnet.logger import setup_logs
from nnet.dataset import VectorDataset
from nnet.training import train, snapshot
from nnet.validation import validation
from nnet.prediction import prediction
from nnet.MLP import MultiLayerPerceptron as MLP


##############################################################
# Control Center and Hyperparameter
DIM = 400  # dimension of the (i/x)vector
SELECT_BEST = 'eer'  # eer or val
BATCH_SIZE = TEST_BATCH_SIZE = 8


def main():
    parser = argparse.ArgumentParser(description='Model MLP')
    parser.add_argument('--train-file', required=True,
                        help='train vector file (txt)')
    parser.add_argument('--train-utt2label', required=True,
                        help='train utt2label')
    parser.add_argument('--validation-file', required=True,
                        help='dev vector file (txt)')
    parser.add_argument('--validation-utt2label', required=True,
                        help='dev utt2label')
    parser.add_argument('--eval-file',
                        help='eval vector file (txt)')
    parser.add_argument('--eval-utt2label',
                        help='eval utt2label')
    parser.add_argument('--dim', type=int, required=True,
                        help='input vector dimension')
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
    args = parser.parse_args()

    torch.cuda.empty_cache()

    # Init model & Setup logs
    model = MLP(input_size=args.dim, hidden_size=256, output_size=1)
    run_name = "MLP_" + str(args.dim) + time.strftime("-%Y_%m_%d-%H_%M_%S-")
    logger = setup_logs(args.logging_dir, run_name)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info("use_cuda is {}".format(use_cuda))

    # Global timer
    global_timer = timer()

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
    # Loading the dataset
    params = {'num_workers': 0,
              'pin_memory': False,
              'worker_init_fn': np.random.seed(args.seed)} if use_cuda else {}

    logger.info('===> loading train and dev dataset')

    training_set = VectorDataset(args.train_file)
    validation_set = VectorDataset(args.validation_file)
    train_loader = data.DataLoader(
        training_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        **params
    )  # set shuffle to True
    validation_loader = data.DataLoader(
        validation_set,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        **params
    )  # set shuffle to False

    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)  # noqa

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('#### Model summary below ####\n {}\n'.format(str(model)))
    logger.info('===> Model total # parameter: {}\n'.format(model_params))
    ###########################################################
    # Training
    best_eer = np.inf
    early_stopping, max_patience = 0, 5  # early stopping and maximum patience

    total_train_time = []
    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()

        # Train and validate
        train(args, model, device, train_loader, optimizer, epoch)
        val_loss, eer = validation(
            args,
            model,
            device,
            validation_loader,
            args.validation_utt2label
        )
        scheduler.step(val_loss)
        # Save
        is_best = eer < best_eer
        best_eer = min(eer, best_eer)

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
    # Prediction
    if args.eval_file and args.eval_utt2label:
        logger.info('===> loading eval dataset')
        eval_set = VectorDataset(args.eval_file)
        eval_loader = data.DataLoader(
            eval_set,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False,
            **params
        )  # set shuffle to False

        logger.info('===> loading best model for prediction')
        checkpoint = torch.load(
            os.path.join(args.logging_dir, run_name + '-model_best.pth')
        )
        model.load_state_dict(checkpoint['state_dict'])
        t_start_eval = timer()
        eval_loss, eval_eer = prediction(args, model, device, eval_loader, args.eval_utt2label)  # noqa
        end_global_timer = timer()
        logger.info("#### Total prediction time: {}".format(end_global_timer - t_start_eval))  # noqa
    ###########################################################
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))

    print(0)


if __name__ == '__main__':
    main()
