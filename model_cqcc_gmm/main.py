import os
import time
import argparse

import pickle
import numpy as np
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
from timeit import default_timer as timer

from metrics import compute_eer
from logger import setup_logs


def load_mat_cqcc(file_path: str) -> np.array:
    return loadmat(file_path)['x']


def train(train_dir, train_label_file, logger):
    """
    This function temporarily works only on the ASVspoof2017 dataset.
    """
    X_genuine = np.empty((90, 0), float)
    X_spoof = np.empty((90, 0), float)

    f_train = open(train_label_file)
    for line in f_train:
        item = line.strip().split(' ')
        f_path = os.path.join(train_dir, item[0] + '.wav_cqcc.mat')
        cqcc = load_mat_cqcc(f_path)
        if item[1] == '1':
            X_genuine = np.concatenate((X_genuine, cqcc), axis=1)
        elif item[1] == '0':
            X_spoof = np.concatenate((X_spoof, cqcc), axis=1)
        else:
            raise ValueError("invalid label")
    f_train.close()

    train_start = timer()

    logger.info("===> start training GMM for genuine utt:")
    gmm_genuine = GaussianMixture(
        n_components=512,
        covariance_type='diag',
        reg_covar=10e-6,
        max_iter=100,
        verbose=1
    )
    gmm_genuine.fit(np.transpose(X_genuine))

    logger.info("===> start training GMM for spoof utt:")
    gmm_spoof = GaussianMixture(
        n_components=512,
        covariance_type='diag',
        reg_covar=10e-6,
        max_iter=100,
        verbose=1
    )
    gmm_spoof.fit(np.transpose(X_spoof))

    train_end = timer()

    logger.info("GMM training done!")
    logger.info("#### Total elapsed time for training: {}".format(train_end - train_start))  # noqa

    return gmm_genuine, gmm_spoof


def pred(eval_dir, eval_label_file, gmm_genuine, gmm_spoof, logger):
    """
    This function temporarily works only on the ASVspoof2017 dataset.
    """
    y_true = []
    scores = []
    f_eval = open(eval_label_file)

    eval_start = timer()
    for line in f_eval:
        item = line.strip().split(' ')
        if item[1] != '1' and item[1] != '0':
            raise ValueError("invalid value for lable")
        y_true.append(int(item[1]))
        f_path = os.path.join(eval_dir, item[0] + '.wav_cqcc.mat')
        cqcc = load_mat_cqcc(f_path)
        llk_genuine = gmm_genuine.score(np.transpose(cqcc))
        llk_spoof = gmm_spoof.score(np.transpose(cqcc))
        scores.append(llk_genuine - llk_spoof)
    f_eval.close()
    eval_end = timer()

    logger.info("#### Total elapsed time for prediction: %s" % (eval_end - eval_start))

    return y_true, scores


def main():
    parser = argparse.ArgumentParser(description='CQCC-GMM for spoof detection')
    parser.add_argument('--train-dir', required=True,
                        help='train feature dir')
    parser.add_argument('--train-utt2label', required=True,
                        help='train utt2label')
    parser.add_argument('--eval-dir', required=True,
                        help='eval feature dir')
    parser.add_argument('--eval-utt2label', required=True,
                        help='train utt2label')
    parser.add_argument('--logging-dir', required=True,
                        help='log save directory')
    parser.add_argument('--suffix', default='',
                        help='suffix for run name, not required')
    parser.add_argument('--model-dir', default=None,
                        help='model save directory')
    args = parser.parse_args()

    # set up loggers
    run_name = "gmm-cqcc" + time.strftime("-%Y-%m-%d_%H_%M_%S") + args.suffix
    logger = setup_logs(args.logging_dir, run_name)

    np.random.seed(0)

    global_start = timer()
    # train
    gmm_genuine, gmm_spoof = train(args.train_dir, args.train_utt2label, logger)

    if args.model_dir:
        save_name_g = os.path.join(args.model_dir, run_name + '-genuine.pkl')
        pickle.dump(gmm_genuine, open(save_name_g, 'wb'))
        save_name_s = os.path.join(args.model_dir, run_name + '-spoof.pkl')
        pickle.dump(gmm_spoof, open(save_name_s, 'wb'))

    # eval
    y_true, scores = pred(args.eval_dir, args.eval_utt2label, gmm_genuine, gmm_spoof, logger)  # noqa

    eer = compute_eer(y_true, scores)
    logger.info("====>> Prediction EER: %s" % eer)

    global_end = timer()
    logger.info("#### Total elapsed time: %s" % (global_end - global_start))


if __name__ == "__main__":
    main()
