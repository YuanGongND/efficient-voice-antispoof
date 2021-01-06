# efficient-voice-antispoof

## Introduction
(ToDo: add project description here)

## Citation

## Experiments

### Setup
(TODO: add hardware description and software dependency here)

### Feature Extraction

#### CQCC extraction
To be aligned with the Matlab [baseline experiments](https://www.asvspoof.org/index2017.html) used in ASVspoof2017 challenge, the CQCC features are directly extracted with the Matlab code and saved as .mat files. The .mat files are processed into numpy arrays via scipy.io.loadmat.

#### logspec extraction
The [reference work](https://github.com/jefflai108/Attentive-Filtering-Network) used Kaldi for logspec extraction. To make sure each feature can be fed into models one by one, we used the [equivalent functions provided by torchaudio](https://pytorch.org/audio/stable/transforms.html) to extract the logspec. Parameters are set the same as the reference work.

To run logspec extraction, change directory to "model_AFN", then run

```bash
./extract_feat.sh
```

### GMM


### AFN


### Spearker Recognition


## Authors
