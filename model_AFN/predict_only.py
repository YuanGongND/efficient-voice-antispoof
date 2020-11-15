# Utilities
from __future__ import print_function
import argparse
import time
import sys
import random
from timeit import default_timer as timer

# Libraries
import numpy as np

# Torch
import torch
from torch.utils import data

# Custom Imports
from src.data_reader.vND_dataset import SpoofDataset
from src.v1_logger import setup_logs
from src.v1_metrics import compute_eer
from src.v4_prediction import prediction, scores
from src.attention_neuro.simple_attention_network import AttenResNet, PreAttenResNet, AttenResNet2, AttenResNet3, AttenResNet4, AttenResNet5, AttenResNet4Deform, AttenResNet4Deform_512


# network pruning imports
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from acceleration_compression.pruning import NetworkPruning
from acceleration_compression.quantization import NetworkQuantization
from acceleration_compression.decomposition import NetworkDecomposition

feat_dim = 257
m = 1091
atten_channel = 16
atten_activation = 'sigmoid'
temperature = 10
# model = AttenResNet5(atten_activation, atten_channel, temperature)
MODEL = AttenResNet4(atten_activation, atten_channel)

### trained model weights
model_dir = 'snapshots/attention/'
"""
## AttenResNet4 (channel=16, softmax(dim=3), attention residual)
## snapshots/scoring/attention5
model1 = model_dir + 'attention-2018-07-19_17_31_12-model_best.pth'
model2 = model_dir + 'attention-2018-07-19_18_30_58-model_best.pth'
model3 = model_dir + 'attention-2018-07-20_05_20_07-model_best.pth'
model4 = model_dir + 'attention-2018-07-20_16_40_54-model_best.pth'
model5 = model_dir + 'attention-2018-07-19_18_26_32-model_best.pth'
model6 = model_dir + 'attention-2018-07-19_18_31_09-model_best.pth'
model7 = model_dir + 'attention-2018-07-19_20_48_59-model_best.pth'
models = [model1, model2, model3, model4, model5, model6, model7]
# train: 0.13
# dev: 6.62
# eval: 9.28
"""
# for models fusion
model1 = model_dir + 'attention-2020-10-14_21_53_09-model_best.pth'
models = [model1]
# for single_model
model_path = model_dir + 'attention-2020-10-14_21_53_09-model_best.pth'


def load_model(model, model_path, device, freeze=False):
    """load pre-trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model params is', model_params)

    return model


def apply_net_prune(model, logger, percentage, prune_method='L1Unstructured'):
    net_prune = NetworkPruning(model, percentage=percentage, prune_method=prune_method)
    logger.info(
        "orignal model size (MB): {}"
        .format(net_prune.print_size_of_model(model))
    )
    logger.info(
        '# non-zero params before pruning: {}'
        .format(net_prune.get_num_parameters(model, is_nonzero=True))
    )
    model = net_prune.pruning()
    logger.info('apply [{}] PRUNING with percentage: {}'.format(prune_method, percentage))
    logger.info(
        "pruned model size (MB): {}"
        .format(net_prune.print_size_of_model(model))
    )
    logger.info(
        '# non-zero params after pruning: {}'
        .format(net_prune.get_num_parameters(model, is_nonzero=True))
    )
    return model


def apply_net_quant(model, logger, use_cuda, quant_method='dynamic'):
    # quantization has to be on CPU
    model.to(torch.device('cpu'))
    net_quant = NetworkQuantization(model, quant_method=quant_method)
    logger.info(
        "orignal model size (MB): {}"
        .format(net_quant.print_model_size(model))
    )
    logger.info(
        '# non-zero params before quant: {}'
        .format(net_quant.get_num_parameters(model, is_nonzero=True))
    )
    model = net_quant.quantization()
    logger.info('apply [{}] QUANTIZATION'.format(quant_method))
    logger.info(
        "quanted model size (MB): {}"
        .format(net_quant.print_model_size(model))
    )
    logger.info(
        '# non-zero params after quant: {}'
        .format(net_quant.get_num_parameters(model, is_nonzero=True))
    )
    # if use_cuda:
    #     model.to(torch.device('cuda'))
    #     print('model moved to GPU')
    return model


def apply_net_decomp(model, logger, rank, decomp_type='tucker'):
    net_decomp = NetworkDecomposition(model, rank=rank, decomp_type=decomp_type)
    logger.info(
        "orignal model size (MB): {}"
        .format(net_decomp.print_size_of_model(model))
    )
    logger.info(
        '# non-zero params before decomp: {}'
        .format(net_decomp.get_num_parameters(model, is_nonzero=True))
    )
    model = net_decomp.decomposition()
    logger.info('apply [{}] DECOMP with rank: {}'.format(decomp_type, rank))
    logger.info(
        "decomp-ed model size (MB): {}"
        .format(net_decomp.print_size_of_model(model))
    )
    logger.info(
        '# non-zero params after decomp: {}'
        .format(net_decomp.get_num_parameters(model, is_nonzero=True))
    )
    return model


def main():
    ##############################################################
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--eval-dir',
                        help='directory to eval set')
    parser.add_argument('--eval-utt2label',
                        help='train utt2label')
    parser.add_argument('--model-path',
                        help='trained model')
    parser.add_argument('--logging-dir', required=True,
                        help='model save directory')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--scoring-txt',  # for multi-models
                        help='output scoring text file')
    parser.add_argument('--label-txt',  # for multi-models
                        help='output labels text file')
    # for network compression usage
    parser.add_argument('--compress',  # prune, quant, pq, decomp
                        help='network compress method')
    parser.add_argument('--prune-pct',
                        help='percentage for pruning')
    parser.add_argument('--prune-method', default='L1Unstructured',
                        help='pruning method')
    parser.add_argument('--quant-method', default='dynamic',
                        help='quantization method')
    parser.add_argument('--decomp-rank', type=int,
                        help='rank value for decomposition')
    parser.add_argument('--decomp-type', default='tucker',
                        help='decomposition type')
    parser.add_argument('--input-dim', type=int, default=1091,
                        help='input dimension')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if args.compress in ("quant", 'pq'):
        use_cuda = False
        print("Quantization can only run on CPU")
    print('use_cuda is', use_cuda)
    if use_cuda:
        torch.cuda.empty_cache()

    # Global timer
    global_timer = timer()

    # Setup logs
    if not args.compress:
        args.compress = "no_compress"
    run_name = "pred_only" + time.strftime("-%Y-%m-%d-") + args.compress + '-' + str(args.input_dim)
    logger = setup_logs(args.logging_dir, run_name)
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

    ##############################################################
    # Loading the dataset
    params = {'num_workers': 0,
              'pin_memory': False,
              'worker_init_fn': np.random.seed(args.seed)} if use_cuda else {}

    logger.info('===> loading eval dataset {}'.format(args.eval_dir))
    eval_set = SpoofDataset(args.eval_dir, args.eval_utt2label)
    eval_loader = data.DataLoader(
        eval_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        **params
    )  # set shuffle to False

    ###################### for single model #####################
    t_start_load = timer()
    model = MODEL
    if args.input_dim != 1091:
        if args.input_dim == 512:
            model = AttenResNet4Deform_512(atten_activation, atten_channel, size1=(257, args.input_dim))
        else:
            model = AttenResNet4Deform(atten_activation, atten_channel, size1=(257, args.input_dim))
        model_path = model_dir + 'attention-2020-11-11-feat_{}-model_best.pth'.format(args.input_dim)
    model.to(device)
    logger.info('===> loading {} for prediction'.format(model_path))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    t_end_load = timer()
    logger.info('model loading time: {}'.format(t_end_load - t_start_load))
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('original model params is', model_params)

    #### for network pruning usage ####
    if args.compress == 'prune':
        model = apply_net_prune(
            model=model,
            logger=logger,
            percentage=float(args.prune_pct),
            prune_method=args.prune_method
        )
    elif args.compress == 'quant':
        model = apply_net_quant(
            model=model,
            logger=logger,
            use_cuda=use_cuda,
            quant_method=args.quant_method
        )
    elif args.compress == 'decomp':
        model.to('cpu')  # TODO: fix it! may encounter some errors of tensor on two devices
        model = apply_net_decomp(
            model=model,
            logger=logger,
            rank=args.decomp_rank,
            decomp_type=args.decomp_type
        )
        fine_tuned_path = (
            "./snapshots/attention/decomp-2020-10-14_21_53_09-rank{}-{}-model_best.pth"
            .format(args.decomp_rank, args.decomp_type)
        )
        model = load_model(
            model=model,
            model_path=fine_tuned_path,
            device=device
        )
        model.to(device)  # TODO: along with the previous TODO
    elif args.compress == 'pq':
        model = apply_net_prune(model, logger, float(args.prune_pct), args.prune_method)
        model = apply_net_quant(model, logger, use_cuda, args.prune_method)

    ###################################

    t_start = timer()
    eval_loss, eval_eer = prediction(args, model, device, eval_loader, args.eval_utt2label)  # noqa
    t_end = timer()
    logger.info('===> elapsed time for prediction: {}'.format(t_end - t_start))
    logger.info('===> evalidation set: EER: {:.4f}\n'.format(eval_eer))
    """
    ################### for multiple models #####################
    np.set_printoptions(threshold=sys.maxsize)
    sum_preds = 0
    for model_i in models:
        logger.info('===> loading {} for prediction'.format(model_i))
        checkpoint = torch.load(model_i, map_location=lambda storage, loc: storage)  # load everything onto CPU
        model.load_state_dict(checkpoint['state_dict'])
        # network pruning
        # net_prune = NetworkPruning(model, percentage=0.5)
        # model = net_prune.pruning()
        # network quantinization
        # net_quant = NetworkQuantization(model)
        # model = net_quant.quantization()

        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model params is', model_params)

        t_start = timer()
        eval_preds, eval_labels = scores(args, model, device, eval_loader, args.eval_utt2label)  # noqa
        sum_preds += eval_preds
    sum_preds /= len(models)  # get the average
    eval_eer = compute_eer(eval_labels, sum_preds)
    np.savetxt(args.scoring_txt, sum_preds)
    np.savetxt(args.label_txt, eval_labels)
    logger.info("===> Final predictions done. Here is a snippet")
    logger.info('===> evalidation set: EER: {:.4f}\n'.format(eval_eer))
    ###########################################################
    """
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s\n" % (end_global_timer - global_timer))
    logger.info("==========================DONE==========================\n")


if __name__ == '__main__':
    main()
