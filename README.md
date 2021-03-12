# Improving the Computational Efficiency of Voice Anti-spoofing Models

## Introduction
With the proliferation of smart voice-controlled systems in recent years, many voice-based IoT applications are becoming increasingly vulnerable to various types of replay attacks. Voice anti-spoofing tasks provide effective countermeasures by detecting spoofing speech utterances in such attacks using machine learning. Compared to conventional machine learning models, deep neural networks (DNNs) show significantly higher effectiveness in anti-spoofing tasks. However, DNN-based models usually require extremely powerful computational resources and are often not suitable for deployment on resource-constrained systems such as consumer products or IoT devices. Therefore, there is a need for new techniques to accelerate and compress such models while maintaining their inference effectiveness. In this work, we propose and explore a series of acceleration and compression methods for voice anti-spoofing models. The proposed methods include general-purpose compression methods and task-specific compression methods. We evaluate the compressed models on both their efficiency and effectiveness. Experiments are also conducted on a variety of platforms including low-resource devices and high-performance computing machines. In our evaluation, the best general-purpose compression method shows 80.55\% inference efficiency improvement with an increase in EER of about 10\%, while the best method of task-specific compression yields a 96.63\% improvement in inference efficiency with the EER increasing by 5.4\%.

## Citation
This work has been submitted to IEEE/ACM Transactions on Audio Speech and Language Processing on March 12th, 2021. Further citation information will be followed.

## Experiments

### Setup

#### Hardware
|        Platform        |                         CPU                         |                GPU               | CUDA version | RAM  (GB) |       Operating System      |
|:----------------------:|:---------------------------------------------------:|:--------------------------------:|:------------:|:---------:|:---------------------------:|
|         Desktop        |   Intel(R) i9-9820X 64-bit CPU @ 3.30GHz, 8 cores   |    2 * Nvidia GeForce RTX 2080   |     10.2     |    128    |         Ubuntu 18.04        |
|   Nvidia Jetson Nano   | Cortex-A57 (ARM v8-A) 64-bit SoC @ 1.5 GHz, 6 cores | 1 *  Built-in Nvidia Maxwell GPU |     10.2     |     4     |         Ubuntu 18.04        |
| Raspberry Pi 4 model B |   Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz, 4 cores  |                N/A               |      N/A     |     4     | Raspberry Pi OS 64-bit beta |

#### Software
- Python: 3.7
- scikit-learn: 0.23.2
- PyTorch: 1.6.0 (PyTorch build for Raspberry Pi: http://mathinf.com/pytorch/arm64/)

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

#### Deformed AFN (with input pruning)
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
| 200* | CPU    | Pruned deformed AFN     | 2 prune methods, prune pct: 0-1         |
| 201* | GPU    | Pruned deformed AFN     | 2 prune methods, prune pct: 0-1         |
| 300* | CPU    | Quantized deformed AFN  | Only support CPU run.                   |
| 400* | GPU    | Deformed AFN            | Fine-tune the model after decomposition |
| 401* | CPU    | Decomposed deformed AFN | Load weights from the fine-tuned model  |
| 402* | GPU    | Decomposed deformed AFN | Load weights from the fine-tuned model  |

*: only executable on desktop machine.

### Multi-task Framework with Spearker Recognition
THe pretrained VoxCeleb i-vector/x-vector extractors are obtained at https://kaldi-asr.org/models/m7
#### ivector_gmm
First, extract ivector. Change directory to model_ivector_gmm/ivector/
```bash
./enroll.sh ../../data/ASVspoof2017
```

Then use the ivector as input for GMM classifier
```bash
./run.sh 0
```

#### xvector_gmm
x-vector realted experiments follow the same directory structure as the i-vector experiments.

## Authors
[Jian Yang](https://github.com/jlinear), [Bryan (Ning) Xia](https://github.com/ningxia), John Bailey, [Yuan Gong](https://github.com/YuanGongND), and Christian Poellabauer.