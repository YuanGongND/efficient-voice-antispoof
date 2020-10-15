import os
import argparse
import time
import torch
import torchaudio
from timeit import default_timer as timer
from src.v1_logger import setup_logs

SAMPLE_RATE = 16000
SNIP_EDGES = False
ENERGY_FLOOR = 0
M = 1091
DATA_NAME = "asv2017-"


def get_logspec(audio_path, device):
    """
    get the log spectrogram of an audio file and apply cmvn normalization
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device)
    logspec = torchaudio.compliance.kaldi.spectrogram(
        waveform=waveform,
        sample_frequency=SAMPLE_RATE,
        snip_edges=SNIP_EDGES,
        energy_floor=ENERGY_FLOOR
    )
    # apply sliding window cmvn normalization with 3s window
    cmvn_logspec = torchaudio.functional.sliding_window_cmn(logspec, cmn_window=3*sample_rate)  # noqa
    # min_lsp, max_lsp = torch.min(cmvn_logspec), torch.max(cmvn_logspec)
    # norm_logspec = (cmvn_logspec - min_lsp) / (max_lsp - min_lsp)
    # return norm_logspec
    return cmvn_logspec


def expand_logspec(tensor):
    """
    expand logspec to fixed dims (M, 257)
    """

    n_rep = M // tensor.shape[0]
    tail_dim = M % tensor.shape[0]
    tmp = tensor.expand(n_rep, tensor.shape[0], tensor.shape[1])
    head = tmp.reshape(M-tail_dim, tensor.shape[1])
    tail = tensor[0:tail_dim, :]

    return torch.cat([head, tail], dim=0)


def main():
    parser = argparse.ArgumentParser(description='Feature (log-spec) Extraction')
    parser.add_argument('--data-dir', required=True,
                        help='data directory contains wave files')
    parser.add_argument('--label-file', required=True,
                        help='protocol file that contains utt2label mapping')
    parser.add_argument('--feat-dir', required=True,
                        help='feature saving directory')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA for feature extraction')
    parser.add_argument('--logging-dir', required=True,
                        help='log save directory')
    args = parser.parse_args()

    os.makedirs(args.logging_dir, exist_ok=True)
    os.makedirs(args.feat_dir, exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Setup logs
    basename = DATA_NAME + os.path.basename(args.data_dir)
    run_name = "logspec-" + basename + time.strftime("-%Y-%m-%d")
    if os.path.exists(run_name + ".log"):
        os.remove(run_name + ".log")
    logger = setup_logs(args.logging_dir, run_name)

    logger.info("===> Start logspec extraction for dataset: " + args.data_dir)

    # global timer starts
    global_start = timer()

    utt2label_path = os.path.join(
        os.path.dirname(args.label_file),
        os.path.basename(args.label_file)+'.utt2label'
    )

    f_utt2label = open(utt2label_path, 'w')
    f_label = open(args.label_file, 'r')

    for line in f_label:
        item = line.strip().split(' ')
        if item[1] == 'genuine':
            label = 1
        elif item[1] == 'spoof':
            label = 0
        else:
            raise ValueError("Invalid label: " + item[1])
        f_utt2label.write(item[0][:-4] + ' ' + str(label) + '\n')

        audio_path = os.path.join(args.data_dir, item[0])

        t_start = timer()
        logspec = get_logspec(audio_path, device)
        feat = expand_logspec(logspec)
        t_end = timer()
        logger.info(item[0] + "\tfeature extraction time: %s" % (t_end - t_start))

        f_feat_path = os.path.join(args.feat_dir, item[0][:-4]+'.pt')
        torch.save(feat, f_feat_path)

    f_label.close()
    f_utt2label.close()

    global_end = timer()
    logger.info("#### Done logspec extraction for dataset: " + args.data_dir + "####")
    logger.info("Total elapsed time: %s" % (global_end - global_start))


if __name__ == "__main__":
    main()
