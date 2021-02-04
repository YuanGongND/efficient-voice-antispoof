import os
import argparse


def main():
    """
    1. copy label file to the output data directory (if already in utt2spk format)
    2. create tmp directory for later kaldi usage
    3. create wavscp file that point out the path for all wav
    """
    parser = argparse.ArgumentParser(description='ivector runner')
    parser.add_argument('--wav-dir', required=True,
                        help='directory to original audio files')
    parser.add_argument('--label-file', required=True,
                        help='label files')
    parser.add_argument('--out-dir', required=True,
                        help='output directory to data')
    args = parser.parse_args()

    tmp_dir = os.path.join(args.out_dir, "tmp")
    utt2spk_path = os.path.join(args.out_dir, "utt2spk")
    wavscp_path = os.path.join(args.out_dir, "wav.scp")

    if (os.system("mkdir -p %s" % (tmp_dir)) != 0):
        print("Error making directory %s" % (tmp_dir))

    # refer to the label and create utt2spk, wav.scp
    f_utt2spk = open(utt2spk_path, 'w')
    f_wavscp = open(wavscp_path, 'w')
    f_label = open(args.label_file, 'r')
    for line in f_label:
        items = line.strip().split(' ')
        wav_file_path = os.path.join(args.wav_dir, items[0] + '.wav')
        if os.path.isfile(wav_file_path):
            f_utt2spk.write(items[1] + '_' + line)
            f_wavscp.write(items[1] + '_' + items[0] + ' ' + wav_file_path + '\n')
        else:
            raise FileExistsError("wav file does not exist: %s" % wav_file_path)
    f_label.close()
    f_utt2spk.close()
    f_wavscp.close()


if __name__ == '__main__':
    main()
