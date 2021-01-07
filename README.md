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
All timeing info and logs are saved in the "*model_cqcc_gmm/logs*" directory.


### AFN


### Spearker Recognition


## Authors
