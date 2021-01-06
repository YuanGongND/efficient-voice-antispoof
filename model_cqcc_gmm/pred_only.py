import os
import time
import argparse

import pickle
import numpy as np
from scipy.io import loadmat
# from sklearn.mixture import GaussianMixture
from timeit import default_timer as timer

from metrics import compute_eer
from logger import setup_logs


def load_mat_cqcc(file_path: str) -> np.array:
    return loadmat(file_path)['x']


def load_model(file_path):
    """
    only works with model saved via pickle
    """
    model = pickle.load(open(file_path, 'rb'))
    return model


def pred(eval_dir, eval_label_file, gmm_genuine, gmm_spoof, logger):
    """
    This function temporarily works only on the ASVspoof2017 dataset.
    """
    y_true = []
    scores = []
    f_eval = open(eval_label_file)

    logger.info("====> Start prediction")

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
    parser.add_argument('--eval-dir', required=True,
                        help='eval feature dir')
    parser.add_argument('--eval-utt2label', required=True,
                        help='train utt2label')
    parser.add_argument('--logging-dir', required=True,
                        help='log save directory')
    parser.add_argument('--suffix', default='',
                        help='suffix for run name, not required')
    parser.add_argument('--model-genuine', required=True,
                        help='pretrained genuine model file (pickle file)')
    parser.add_argument('--model-spoof', required=True,
                        help='pretrained spoof model file (pickle file)')
    args = parser.parse_args()

    # set up loggers
    run_name = "gmm-cqcc" + time.strftime("-%Y-%m-%d_%H_%M_%S") + "-pred_only" + args.suffix  # noqa
    logger = setup_logs(args.logging_dir, run_name)

    np.random.seed(0)

    global_start = timer()
    # load models
    logger.info("====> Loading pretrained models")
    gmm_genuine = load_model(args.model_genuine)
    gmm_spoof = load_model(args.model_spoof)

    # eval
    y_true, scores = pred(args.eval_dir, args.eval_utt2label, gmm_genuine, gmm_spoof, logger)  # noqa

    eer = compute_eer(y_true, scores)
    logger.info("====>> Prediction EER: %s" % eer)

    global_end = timer()
    logger.info("#### Total elapsed time: %s" % (global_end - global_start))


if __name__ == "__main__":
    main()
