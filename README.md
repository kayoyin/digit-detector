# Retinanet Digit Detector

This repository contains the code to train a Retinanet digit detector from scratch using the SVHN dataset.

## Getting started

First, create a new virtual environment

```
virtualenv venv -p python3
source venv/bin/activate
```

You might need to make sure your python3 link is ready by typing

```bash
which python3
```

Then install the development requirements

```bash
pip install -r requirements.txt
```

## Data pre-processing
Download the training and testing [data](https://drive.google.com/drive/u/0/folders/1Ob5oT9Lcmz7g5mVOcYH3QugA7tV3WsSl), extract and put it in your project as below:
```buildoutcfg
|-- dataset
	|-- train
	|-- test
```

The project uses the Pascal VOC format for the training annotations. To process the annotations, run:
```
sh preprocess.sh
``` 

You should obtain
```buildoutcfg
|-- dataset
	|-- train_anns
        |-- train_imgs
	|-- test
```

## Training
Before training the model, split your training set into a train and a validation set as below.
```buildoutcfg
|-- dataset
	|-- train_anns
        |-- train_imgs
        |-- val_anns
        |-- val_imgs
	|-- test
```

Then, run the following to train the model.
```
python retinanet/main.py
```

## Evaluation
Running the following will generate the predictions of the model for the test set and save them under ``submission.json``
```
python retinanet/predict.py
```