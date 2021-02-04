"""
Filename: main.py
Objective: Appy GMM classifier to xvector (512 dim)
Author: Jian Yang
Created: Jan. 31, 2021
Last updated: Feb. 04, 2021
"""

import os
import time
import argparse

import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from timeit import default_timer as timer

from metrics import compute_eer
from logger import setup_logs


def train(train_txt, logger):
    """
    This function temporarily works only on the ASVspoof2017 dataset.
    """
    X_genuine = np.empty((0, 512), float)
    X_spoof = np.empty((0, 512), float)

    label_train = open(train_txt, 'r')
    for line in label_train:
        items = line.strip().split('  ')
        xvector = np.fromstring(items[1].strip('[]'), sep=' ')
        if items[0].startswith('0'):
            # spoof
            X_spoof = np.concatenate((X_spoof, xvector.reshape(1, -1)), axis=0)
        elif items[0].startswith('1'):
            # genuine
            X_genuine = np.concatenate((X_genuine, xvector.reshape(1, -1)), axis=0)
        else:
            raise ValueError('Invalid naming: ' + items[0])
    label_train.close()

    train_start = timer()

    logger.info("===> start training GMM for genuine utt:")
    gmm_genuine = GaussianMixture(
        n_components=512,
        covariance_type='diag',
        reg_covar=10e-6,
        max_iter=100,
        verbose=1
    )
    gmm_genuine.fit(X_genuine)

    logger.info("===> start training GMM for spoof utt:")
    gmm_spoof = GaussianMixture(
        n_components=512,
        covariance_type='diag',
        reg_covar=10e-6,
        max_iter=100,
        verbose=1
    )
    gmm_spoof.fit(X_spoof)

    train_end = timer()

    logger.info("GMM training done!")
    logger.info("#### Total elapsed time for training: {}".format(train_end - train_start))  # noqa

    return gmm_genuine, gmm_spoof


def pred(eval_txt, gmm_genuine, gmm_spoof, logger):
    """
    This function temporarily works only on the ASVspoof2017 dataset.
    """
    y_true = []
    scores = []
    label_eval = open(eval_txt, 'r')

    eval_start = timer()
    for line in label_eval:
        items = line.strip().split('  ')
        if items[0][0] != '1' and items[0][0] != '0':
            raise ValueError("invalid value for label")
        y_true.append(int(items[0][0]))
        xvector = np.fromstring(items[1].strip('[]'), sep=' ')

        llk_genuine = gmm_genuine.score(xvector.reshape(1, -1))
        llk_spoof = gmm_spoof.score(xvector.reshape(1, -1))
        scores.append(llk_genuine - llk_spoof)
    label_eval.close()
    eval_end = timer()

    logger.info("#### Total elapsed time for prediction: %s" % (eval_end - eval_start))

    return y_true, scores


def main():
    parser = argparse.ArgumentParser(description='xvector-GMM for spoof detection')
    parser.add_argument('--train-txt', required=True,
                        help='train xvector in txt')
    parser.add_argument('--eval-txt', required=True,
                        help='eval xvector in txt')
    parser.add_argument('--logging-dir', required=True,
                        help='log save directory')
    parser.add_argument('--suffix', default='',
                        help='suffix for run name, not required')
    parser.add_argument('--model-dir', default=None,
                        help='model save directory')
    args = parser.parse_args()

    # set up loggers
    run_name = "gmm-xvector" + time.strftime("-%Y-%m-%d_%H_%M_%S") + args.suffix
    logger = setup_logs(args.logging_dir, run_name)

    np.random.seed(0)

    global_start = timer()
    # train
    gmm_genuine, gmm_spoof = train(args.train_txt, logger)

    if args.model_dir:
        save_name_g = os.path.join(args.model_dir, run_name + '-genuine.pkl')
        pickle.dump(gmm_genuine, open(save_name_g, 'wb'))
        save_name_s = os.path.join(args.model_dir, run_name + '-spoof.pkl')
        pickle.dump(gmm_spoof, open(save_name_s, 'wb'))

    # eval
    y_true, scores = pred(args.eval_txt, gmm_genuine, gmm_spoof, logger)

    eer = compute_eer(y_true, scores)
    logger.info("====>> Prediction EER: %s" % eer)

    global_end = timer()
    logger.info("#### Total elapsed time: %s" % (global_end - global_start))


if __name__ == "__main__":
    main()
