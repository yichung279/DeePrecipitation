# rain-forecast-deconv

Predict weather with Deconvolution Net

## Setup

1. Install python package: `pip3 install -r requires.txt`
2. Config environment: `source config_env`

## Workflow

1. Generate Feature: `python3 preprocess.py`
2. Train keras model: `python3 keras/deconv.py`
3. Evaluation: `python3 keras/predict.py`
4. Draw image: `python3 keras/draw.py`
  - Result will be `output` folder
