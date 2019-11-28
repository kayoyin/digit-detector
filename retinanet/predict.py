import keras_retinanet
import argparse
import os
import sys
import warnings
import progressbar
import time
import numpy as np
import cv2
import json
import random

import keras
import keras.preprocessing.image
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401

    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import layers  # noqa: F401
from keras_retinanet import losses
from keras_retinanet import models
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.utils.eval import _get_detections


from retina.pascal import PascalVocGenerator

from main import *

def to_dict(labels, boxes, scores):

    output = {
        "bbox": [list(map(int, x[[1,0,3,2]])) for x in boxes],
        "score": scores,
        "label": [10 if x == 0 else x for x in labels]
    }
    return output

def order_submission(submission):
    order = [str(a) for a in range(1,13069)]
    order.sort()
    order = list(map(int, order))
    submission = [s for _, s in sorted(zip(order, submission))]
    return submission

def infer(
    generator,
    model,
    max_detections=100,
):
    all_detections = _get_detections(generator, model, max_detections=max_detections)
    submission = []

    for i in range(generator.size()):
        labels = []
        boxes = []
        scores = []
        for label in range(generator.num_classes()):
            detections = all_detections[i][label]
            for d in detections:
                labels.append(label)
                boxes.append(d[:4])
                scores.append(d[4])

        submission.append(to_dict(labels, boxes, scores))

    submission = order_submission(submission)
    with open("submission.json", 'w') as file:
        json.dump(submission, file)

def run(args=None):
    args = parse_args(args)
    backbone = models.backbone(args.backbone)
    if args.config:
        args.config = read_config_file(args.config)
    _, generator = create_generators(args, backbone.preprocess_image)

    weights = os.path.join(args.snapshot_path, 'checkpoint.h5')
    _, _, model = create_models(
        backbone_retinanet=backbone.retinanet,
        num_classes=10,
        weights=weights,
        multi_gpu=args.multi_gpu,
        freeze_backbone=args.freeze_backbone,
        lr=args.lr,
        config=args.config
    )

    infer(generator, model)

if __name__ == "__main__":
    run(["pascal",
          "dataset/train_imgs",
          "dataset/train_anns",
          "dataset/val_imgs",
          "dataset/train_anns"])