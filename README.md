# DeePrecipitation

DeePrecipitation is a precipitation nocasting system using ML.With radar echo images in last hour,DeePrecipitation predicts a future radar echo image telling weather about ten minutes later.
## Setup

1. Install python package: `pip3 install -r requires.txt`
2. Config environment: `source config_env`

## Workflow

1. Generate Feature: `python3 preprocess.py`
2. Train keras model: `python3 keras/train.py`
3. Evaluation and Draw images: `python3 keras/predict.py`
  - Result will be `output` folder
