import os
import argparse
import torch
import torchaudio
from timeit import default_timer as timer

SAMPLE_RATE = 16000
SNIP_EDGES = False
ENERGY_FLOOR = 0


# if using GPU, then set device as "cuda"
# device = torch.device('cuda', 0)
device = torch.device('cpu')
def get_logspec(audio_path, device=device):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device)
    logspec = torchaudio.compliance.kaldi.spectrogram(
        waveform=waveform,
        sample_frequency=SAMPLE_RATE,
        snip_edges=SNIP_EDGES,
        energy_floor=ENERGY_FLOOR
    )
    # apply sliding window cmvn normalization with 3s window
    norm_logspec = torchaudio.functional.sliding_window_cmn(logspec, cmn_window=300)
    return norm_logspec


"""
def get_logspec(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    logspec = torchaudio.compliance.kaldi.spectrogram(
        waveform=waveform,
        sample_frequency=SAMPLE_RATE,
        snip_edges=SNIP_EDGES,
        energy_floor=ENERGY_FLOOR
    )
    # apply sliding window cmvn normalization with 3s window
    norm_logspec = torchaudio.functional.sliding_window_cmn(logspec, cmn_window=300)
    return norm_logspec
"""


def main():
    parser = argparse.ArgumentParser(description='Feature (log-spec) Extraction')
    parser.add_argument('--data-dir', required=True,
                        help='data directory contains wave files')
    parser.add_argument('--label-file', required=True,
                        help='protocol file that contains utt2label mapping')
    parser.add_argument('--feat-dir', required=True,
                        help='feature saving directory')
    args = parser.parse_args()

    if not os.path.exists(args.feat_dir):
        os.mkdir(args.feat_dir)

    # global timer starts
    global_timer = timer()

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
        logspec = get_logspec(audio_path)
        f_feat_path = os.path.join(args.feat_dir, item[0][:-4]+'.pt')
        torch.save(logspec, f_feat_path)

    f_label.close()
    f_utt2label.close()

    end_global_timer = timer()
    print("#### Successfully finished logspec extraction! ####")
    print("Total elapsed time: %s" % (end_global_timer - global_timer))


if __name__ == "__main__":
    main()
