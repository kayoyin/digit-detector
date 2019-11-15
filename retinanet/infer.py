# -*- coding: utf-8 -*-

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import glob
import json

from retina.utils import visualize_boxes

MODEL_PATH = 'resnet50.h5'

def load_inference_model(model_path=os.path.join('snapshots', 'resnet.h5')):
    model = models.load_model(model_path, backbone_name='resnet50')
    model = models.convert_model(model)
    model.summary()
    return model

def post_process(boxes, original_img, preprocessed_img):
    # post-processing
    h, w, _ = preprocessed_img.shape
    h2, w2, _ = original_img.shape
    boxes[:, :, 0] = boxes[:, :, 0] / w * w2
    boxes[:, :, 2] = boxes[:, :, 2] / w * w2
    boxes[:, :, 1] = boxes[:, :, 1] / h * h2
    boxes[:, :, 3] = boxes[:, :, 3] / h * h2
    return boxes

def get_dict(boxes, scores, labels,, score_thresh=0.4):
    output = {
        "bbox": [],
        "score": [],
        "label": []
    }

    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh: continue
        x1, y1, x2, y2 = list(map(int, box))
        output['bbox'].append((y1, x1, y2, x2))
        output['score'].append(float(score))
        if label > 0:
            output['label'].append(int(label))
        else:
            output['label'].append(10)

    return output

if __name__ == '__main__':
    
    model = load_inference_model(MODEL_PATH)
    
    # load images
    submission = []
    order = []
    """
    for img_path in glob.glob('../sample/*.png'):
        image = read_image_bgr(img_path)
        image = preprocess_image(image)
        image, _ = resize_image(image, 416, 448)
        images.append(image)
        img_fname = os.path.basename(img_path)
        order.append(int(img_fname[:-4])-1)
    print(images)
    images = np.array(images)[order]
    print(images.shape)
    boxes, scores, labels = model.predict_on_batch(images)
    print(boxes, scores, labels)
    """
    for img_path in glob.glob('dataset/test/*'):
        print(img_path)
        image = read_image_bgr(img_path)

        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        image = preprocess_image(image)
        image, _ = resize_image(image, 416, 448)

        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        img_fname = os.path.basename(img_path)
        order.append(int(img_fname[:-4])-1)


        boxes = post_process(boxes, draw, image)
        labels = labels[0]
        scores = scores[0]
        boxes = boxes[0]

        submission.append(get_dict(boxes, scores, labels))

        """
        visualize_boxes(draw, boxes, labels, scores, class_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        if not os.path.exists('predicted/'):
            os.makedirs('predicted/')
        cv2.imwrite('predicted/' + img_fname, draw)
        """



    submission = [s for _,s in sorted(zip(order, submission))]
    with open("submission.json", 'w') as file:
        json.dump(submission, file)
    """
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    boxes = post_process(boxes, draw, image)
    labels = labels[0]
    scores = scores[0]
    boxes = boxes[0]
    
    visualize_boxes(draw, boxes, labels, scores, class_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    
    # 5. plot    
    plt.imshow(draw)
    plt.show()
    """


