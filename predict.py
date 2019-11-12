# -*- coding: utf-8 -*-

# This driver performs 2-functions for the validation images specified in configuration file:
#     1) evaluate fscore of validation images.
#     2) stores the prediction results of the validation images.

import argparse
import json
import cv2
import numpy as np
import glob
import json
from yolo.frontend import create_yolo
from yolo.backend.utils.box import draw_scaled_boxes
from yolo.backend.utils.annotation import parse_annotation
from yolo.backend.utils.eval.fscore import count_true_positives, calc_score

import os
import yolo


DEFAULT_CONFIG_FILE = "from_scratch.json"
DEFAULT_WEIGHT_FILE = "weights.h5"
DEFAULT_THRESHOLD = 0.3

argparser = argparse.ArgumentParser(
    description='Predict digits driver')

argparser.add_argument(
    '-c',
    '--conf',
    default=DEFAULT_CONFIG_FILE,
    help='path to configuration file')

argparser.add_argument(
    '-t',
    '--threshold',
    default=DEFAULT_THRESHOLD,
    help='detection threshold')

argparser.add_argument(
    '-w',
    '--weights',
    default=DEFAULT_WEIGHT_FILE,
    help='trained weight files')

def get_dict(boxes, probs):
    output = {
        "bbox": [],
        "score": [],
        "label": []
    }

    for box, prob in zip(boxes, probs):
        x1, y1, x2, y2 = list(map(int, box))
        output['bbox'].append((y1, x1, y2, x2))
        output['score'].append(float(np.max(prob)))
        output['label'].append(int(np.argmax(prob)))

    return output


if __name__ == '__main__':
    # 1. extract arguments
    args = argparser.parse_args()
    with open(args.conf) as config_buffer:
        config = json.loads(config_buffer.read())

    # 2. create yolo instance & predict
    yolo = create_yolo(config['model']['architecture'],
                       config['model']['labels'],
                       config['model']['input_size'],
                       config['model']['anchors'])
    yolo.load_weights(args.weights)

    # 3. read image
    write_dname = "predictions"
    if not os.path.exists(write_dname):
        os.makedirs(write_dname)

    submission = []
    for img_path in glob.glob(config['train']['test_image_folder']+'*'):
        print(img_path)
        image = cv2.imread(img_path)
        boxes, probs = yolo.predict(image, float(args.threshold))
        print(boxes, probs)
        submission.append([int(img_path[13:-4]), get_dict(boxes, probs)])
        labels = np.argmax(probs, axis=1) if len(probs) > 0 else [] 
      
        # 4. save detection result
        image = draw_scaled_boxes(image, boxes, probs, config['model']['labels'])
        output_path = os.path.join(write_dname, os.path.basename(img_path))
        
        cv2.imwrite(output_path, image)
        print("{}-boxes are detected. {} saved.".format(len(boxes), output_path))

    submission.sort(key=lambda x: x[0])
    submission = [s[1] for s in submission]

    with open("submission.json", 'w') as file:
        json.dump(submission, file)



