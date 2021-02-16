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
BATCH_SIZE = TEST_BATCH_SIZE = 1


def main():
    parser = argparse.ArgumentParser(description='Model MLP')
    parser.add_argument('--eval-file',
                        help='eval vector file (txt)')
    parser.add_argument('--eval-utt2label',
                        help='eval utt2label')
    parser.add_argument('--model-path',
                        help='path to the pretrained model')
    parser.add_argument('--dim', type=int, required=True,
                        help='input vector dimension')
    parser.add_argument('--logging-dir', required=True,
                        help='model save directory')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.cuda.empty_cache()

    # Init model & Setup logs
    model = MLP(input_size=args.dim, hidden_size=256, output_size=1)
    run_name = "MLP_pred" + str(args.dim) + time.strftime("-%Y_%m_%d-%H_%M_%S-")
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

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('#### Model summary below ####\n {}\n'.format(str(model)))
    logger.info('===> Model total # parameter: {}\n'.format(model_params))

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
            os.path.join(args.model_path),
            map_location=device
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
