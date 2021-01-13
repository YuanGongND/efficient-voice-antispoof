# Improving the Computational Efficiency of Voice Anti-spoofing Models

## Introduction
(ToDo: add project description here)

## Citation

## Experiments

### Setup
(TODO: add hardware description and software dependency here, also point out file hierachy)

### Feature Extraction

#### CQCC extraction
To be aligned with the Matlab [baseline experiments](https://www.asvspoof.org/index2017.html) used in ASVspoof2017 challenge, the CQCC features are directly extracted with the Matlab code and saved as .mat files. The .mat files are processed into numpy arrays via scipy.io.loadmat.

#### logspec extraction
The [reference work](https://github.com/jefflai108/Attentive-Filtering-Network) used Kaldi for logspec extraction. To make sure each feature can be fed into models one by one, we used the [equivalent functions provided by torchaudio](https://pytorch.org/audio/stable/transforms.html) to extract the logspec. Parameters are set the same as the reference work.

To run logspec extraction, change directory to "*model_AFN*", then run

```bash
./extract_feat.sh 0
```

Some later experiments will also use segmented logspec, for this purpose, run the following command under the "*model_AFN*" directory:

```bash
./extract_feat.sh 1
```
The above command segments the full length (1091) logspec into reduced length logspec, with length $win_size$ in 64, 128, 256, 512. For each reduced length, there are 6 different ways of segmentation as follows:

- s: taking the starting $win_size$ of the full length logspec
- e: taking the ending $win_size$ of the full length logspec
- h: taking the continuous $win_size$ segment with the highest energy sum
- l: taking the continuous $win_size$ segment with the lowest energy sum
- hl: the first half with the highest energy sum and the second with the lowest
- lh: the first half with the lowest energy sum and the second with the highest

### GMM
To run the full procedure of GMM model (training + prediction), change directory to "*model_cqcc_gmm*" and run the following command:

```bash
./run.sh 0
```

The trained model for genuine and spoof utterances are stored using [pickle](https://docs.python.org/3/library/pickle.html) in the "*model_cqcc_gmm/pretrained*" directory.

To use trained models and run only prediction, run the following:

```bash
./run.sh 1
```
All timing info and logs are saved in the "*model_cqcc_gmm/logs*" directory.


### AFN
#### Baseline AFN
The baseline AFN model (*AttenResNet4*) used here directly adopts the code from the [original project](https://github.com/jefflai108/Attentive-Filtering-Network), with minimal modifications to fit into the torchaudio-extracted features and the update Python version (original work was in Python 2). To run the full process of AFN model (training + validation + prediction) with the full dataset, change directory to "*model_AFN*" and run the following command:

```bash
./run.sh 0
```

The training process will be repeated for 5 times, and we pick the best-performed model (from the log information) that will be used later for prediction and compression. 

#### Deformed AFN
The deformed AFN model (*AttenResNet4DeformAll*) is modified based on the original AFN model (*AttenResNet4*), so that the network can take in input features with reduced dimensions. 

To run the training process for all combinations of the deformed AFN model, change directory to "*model_AFN*" and run the following command:

```bash
./run.sh 1
```

For each combination, the training process will be repeated for 5 times, the best one will be chosen for later prediction and compression usage.

#### Network compression
In this work, we've implemented and tested 3 general compression methods: pruning, quantization, and decomposition. Each method has been applied to the best model for AFN and the deformed AFN, correspondingly. After the model compression, we measure the predicition efficiency on the evaluation set. All the following experiments use the similar commands as below (under "*model_AFN*" directory):

```bash
./run.sh $N$
```
where $N$ correpsonds to different experiments shown below
| $N$ | Device | Model                   | Note                                    |
|-----|--------|-------------------------|-----------------------------------------|
| 10  | CPU    | Original baseline AFN   |                                         |
| 11  | GPU    | Original baseline AFN   |                                         |
| 20  | CPU    | Pruned AFN              | 2 prune methods, prune pct: 0-1         |
| 21  | GPU    | Pruned AFN              | 2 prune methods, prune pct: 0-1         |
| 30  | CPU    | Quantized AFN           | Only support CPU run.                   |
| 40  | GPU    | Original baseline AFN   | Fine-tune the model after decomposition |
| 41  | CPU    | Decomposed AFN          | Load weights from the fine-tuned model  |
| 42  | GPU    | Decomposed AFN          | Load weights from the fine-tuned model  |
| 200 | CPU    | Pruned deformed AFN     | 2 prune methods, prune pct: 0-1         |
| 201 | GPU    | Pruned deformed AFN     | 2 prune methods, prune pct: 0-1         |
| 300 | CPU    | Quantized deformed AFN  | Only support CPU run.                   |
| 400 | GPU    | Deformed AFN            | Fine-tune the model after decomposition |
| 401 | CPU    | Decomposed deformed AFN | Load weights from the fine-tuned model  |
| 402 | GPU    | Decomposed deformed AFN | Load weights from the fine-tuned model  |

### Spearker Recognition


## Authors
