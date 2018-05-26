# rain-forecast-deconv

Predict weather with Deconvolution Net

## Setup

1. Install python package: `pip3 install -r requires.txt`
2. Config environment: `source config_env`

## Generate Feature

`python3 preprocess.py`

## Train

`python3 keras/deconv.py`

## Evaluation

`python3 keras/predict.py`

## Draw image

`python3 keras/draw.py`

- result will be `output` folder
