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
DEFAULT_SEG = 's'  # taking the first args.seg_win elements
TAIL_SEG = 'e'  # taking the ending args.seg_win elements


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
    min_lsp, max_lsp = torch.min(cmvn_logspec), torch.max(cmvn_logspec)
    norm_logspec = (cmvn_logspec - min_lsp) / (max_lsp - min_lsp)
    return norm_logspec
    # return cmvn_logspec


def expand_logspec(tensor, M=M):
    """
    expand logspec to fixed dims (M, 257)
    """

    n_rep = M // tensor.shape[0]
    tail_dim = M % tensor.shape[0]
    tmp = tensor.expand(n_rep, tensor.shape[0], tensor.shape[1])
    head = tmp.reshape(M-tail_dim, tensor.shape[1])
    tail = tensor[0:tail_dim, :]

    return torch.cat([head, tail], dim=0)


def sliding_sum(tensor, win_size, method):
    """
    method: 'h' or "l"
    """
    l_tensor = tensor.shape[0]
    if win_size > l_tensor:
        raise ValueError("sliding window larger than tensor length!")

    flat_tensor = torch.sum(tensor**2, dim=1)

    left_idx = 0
    tmp_sum = sum(flat_tensor[0: win_size])
    max_sum = sum(flat_tensor[0: win_size])
    min_sum = sum(flat_tensor[0: win_size])

    for i in range(l_tensor - win_size):
        next_sum = tmp_sum + flat_tensor[i + win_size] - flat_tensor[i]
        if method == 'h' and next_sum > max_sum:
            left_idx = i + 1
            max_sum = next_sum
        if method == 'l' and next_sum < min_sum:
            left_idx = i + 1
            min_sum = next_sum
        tmp_sum += flat_tensor[i + win_size] - flat_tensor[i]

    return tensor[left_idx:left_idx+win_size][:]


def segment_logspec(tensor, win_size, method):
    """
    take a fixed segment of logspec by the energy sum
    tensor: torch.Size([x, 257])   x is the original length of the logspec
    win_size: int: 64, 128, 256, 512
    method: str: 'h' - take the highest; 'l': take the lowest
            'hl': highest + lowest; 'lh': lowest + highest
    Output: torch.Size([win_size, 257])
    """
    l_tensor = tensor.shape[0]

    if method == 'h' or method == 'l':
        if win_size >= l_tensor:
            return expand_logspec(tensor, M=win_size)
        else:
            return sliding_sum(tensor, win_size, method)
    elif method == 'hl' or method == 'lh':
        if win_size >= 2 * l_tensor:
            _tensor = expand_logspec(tensor, M=win_size // 2)
            return torch.cat((_tensor, _tensor), 0)
        else:
            h_tensor = sliding_sum(tensor, win_size // 2, 'h')
            l_tensor = sliding_sum(tensor, win_size // 2, 'l')
            if method == 'hl':
                return torch.cat((h_tensor, l_tensor), 0)
            else:
                return torch.cat((l_tensor, h_tensor), 0)
    else:
        raise ValueError("Invalid method type")


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
    parser.add_argument('--segment', action='store_true', default=False,
                        help='whether to segment the logsepc by energy')
    parser.add_argument('--seg-win', type=int,
                        help='the window size to be used for segment: 64, 128..')
    parser.add_argument('--seg-method',
                        help='the method to be used for segment: h, l, hl, lh')
    args = parser.parse_args()

    os.makedirs(args.logging_dir, exist_ok=True)
    os.makedirs(args.feat_dir, exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.segment and (args.seg_method is None or args.seg_win is None):
        raise ValueError("segment method or win_size is missing")

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
        if args.segment:
            if args.seg_method == DEFAULT_SEG:
                feat = expand_logspec(logspec, M=args.seg_win)
            elif args.seg_method == TAIL_SEG:
                feat = torch.flipud(
                    expand_logspec(torch.flipud(logspec), M=args.seg_win)
                )
            else:
                feat = segment_logspec(logspec, args.seg_win, args.seg_method)
        else:
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
