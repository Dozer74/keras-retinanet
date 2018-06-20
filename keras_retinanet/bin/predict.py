import argparse
import cv2
import os
import sys
from glob import glob
from os.path import join, basename
from matplotlib import pyplot as plt
from tqdm import tqdm

import numpy as np
import keras
import keras.preprocessing.image
import tensorflow as tf

# Allow relative imports when being executed as script.
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin

    __package__ = "keras_retinanet.bin"

from .. import models
from ..models.retinanet import retinanet_bbox


def get_session(memory_fraction):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    return tf.Session(config=config)


def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('images_dir', help='Path to train images folder.')
    parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    parser.add_argument('model', help='Path to the model snapshot.')

    parser.add_argument('--score-threshold', help='Threshold for score value.', default=0.5, type=float)
    parser.add_argument('--backbone', help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int,
                        default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.',
                        type=int, default=1333)
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.5)
    parser.add_argument('--save', dest='save_type', choices=['all', 'image', 'text'], default='image')
    parser.add_argument('--path', dest='save_path', default='results')

    subparsers = parser.add_subparsers(dest='detection_type')
    subparsers.required = True

    subparsers.add_parser('whole')

    parser_b = subparsers.add_parser('crops')
    parser_b.add_argument('--cols', type=int, default=1)
    parser_b.add_argument('--rows', type=int, default=1)
    parser_b.add_argument('--add-whole', action='store_true', default=False)

    return parser.parse_args(args)


def nms_fast(boxes, overlapThresh=0.5):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs,
                         np.concatenate(
                             ([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]


def predict_crop(model, crop_bgr, image_min_side=800, image_max_side=1333, score_threshold=0.33):
    img_ar = preprocess_image(crop_bgr)
    img_ar, scale = resize_image(img_ar, min_side=image_min_side, max_side=image_max_side)
    img_ar = np.expand_dims(img_ar, axis=0)

    boxes, scores, labels = model.predict_on_batch(img_ar)
    boxes /= scale

    results = []
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < score_threshold:
            break

        x1, y1, x2, y2 = box.astype(int)
        results.append([x1, y1, x2, y2, label, score])

    return results


def save_results(results, img_rgb, save_path, save_type):
    if save_type == 'text' or save_type == 'all':
        save_text_path = os.path.splitext(save_path)[0] + '.txt'
        with open(save_text_path, 'w') as f:
            for box in results:
                f.write(','.join(map(str, (*map(int, box[:-1]), box[-1]))) + '\n')

    if save_type == 'image' or save_type == 'all':
        plt.figure(figsize=(20, 12))
        ax = plt.gca()
        for box in results:
            x1, y1, x2, y2, label, score = box
            w, h = x2 - x1, y2 - y1

            color = 'b'
            ax.add_patch(plt.Rectangle((x1, y1), w, h, color=color, fill=False, linewidth=1))

            caption = '{:.2f}'.format(score)
            ax.text(x1, y1 - 10, caption, size=10, color='white',
                    bbox={'edgecolor': color, 'facecolor': color, 'alpha': 0.7, 'pad': 4})

        plt.imshow(img_rgb)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close('all')


def make_crops(img, rows, cols):
    height, width = img.shape[:2]
    x_part = width // rows
    y_part = height // cols

    crops = []
    for y in range(cols):
        for x in range(rows):
            crop = img[y * y_part:(y + 1) * y_part, x * x_part:(x + 1) * x_part]
            crops.append((crop, (x * x_part, y * y_part)))

    return crops


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    keras.backend.tensorflow_backend.set_session(get_session(args.gpu_memory_fraction))

    print('Loading model...', end=' ', flush=True)
    model = models.load_model(args.model, backbone_name=args.backbone)
    prediction_model = retinanet_bbox(model=model, nms=True)
    print('Ok.')

    for path in tqdm(glob(join(args.images_dir, '*.jpg'))):
        img_bgr = read_image_bgr(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        results = []
        if args.detection_type == 'whole':
            results = predict_crop(prediction_model, img_bgr, score_threshold=args.score_threshold)
        else:
            for crop, offsets in make_crops(img_bgr, args.rows, args.cols):
                dx, dy = offsets
                for box in predict_crop(prediction_model, crop, score_threshold=args.score_threshold):
                    x1, y1, x2, y2, label, conf = box
                    results.append([x1 + dx, y1 + dy, x2 + dx, y2 + dy, label, conf])
            if args.add_whole:
                results += predict_crop(prediction_model, img_bgr, score_threshold=args.score_threshold)

        results = np.array(results)
        results = nms_fast(results)

        output_dir = join(os.path.dirname(path), args.save_path)
        os.makedirs(output_dir, exist_ok=True)
        save_path = join(output_dir, basename(path))

        save_results(results, img_rgb, save_path, args.save_type)


if __name__ == '__main__':
    main()