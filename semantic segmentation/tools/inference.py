import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from vedaseg.runners import InferenceRunner
from vedaseg.utils import Config


CLASSES = ('background',
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
    'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
    'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
    'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
    'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
    'floor-other', 'floor-stone', 'floor-tile', 'floor-wood',
    'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass',
    'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat',
    'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
    'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
    'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
    'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
    'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other',
    'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
    'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
    'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
    'window-blind', 'window-other', 'wood')

PALETTE = [[0, 0, 0], [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
            [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
            [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
            [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
            [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
            [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
            [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
            [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
            [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
            [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
            [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
            [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
            [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
            [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
            [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
            [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
            [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
            [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
            [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
            [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
            [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
            [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
            [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
            [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
            [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
            [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
            [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
            [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
            [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
            [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
            [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
            [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
            [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
            [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
            [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
            [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
            [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
            [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
            [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
            [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
            [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
            [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
            [64, 192, 96], [64, 160, 64], [64, 64, 0]]

# CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#             'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
#             'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#             'train', 'tvmonitor')

# PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
#             [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
#             [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
#             [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
#             [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def inverse_pad(output, image_shape):
    h, w = image_shape
    return output[:h, :w]


def plot_result(img, mask, cover):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Vedaseg Demo", y=0.95, fontsize=16)

    ax[0].set_title('image')
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax[1].set_title('mask')
    ax[1].imshow(mask)

    ax[2].set_title('cover')
    ax[2].imshow(cv2.cvtColor(cover, cv2.COLOR_BGR2RGB))
    plt.show()


def result(fname,
           pred_mask,
           classes,
           multi_label=False,
           palette=None,
           show=False,
           out=None):
    if palette is None:
        palette = np.random.randint(0, 255, size=(len(classes), 3))
    else:
        palette = np.array(palette)
    img_ori = cv2.imread(fname)
    mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        if multi_label:
            mask[pred_mask[:, :, label] == 1] = color
        else:
            mask[pred_mask == label, :] = color

    cover = img_ori * 0.5 + mask * 0.5
    cover = cover.astype(np.uint8)

    if out is not None:
        _, fullname = os.path.split(fname)
        fname, _ = os.path.splitext(fullname)
        save_dir = os.path.join(out, fname)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'img.png'), img_ori)
        cv2.imwrite(os.path.join(save_dir, 'mask.png'), mask)
        cv2.imwrite(os.path.join(save_dir, 'cover.png'), cover)
        if multi_label:
            for i in range(pred_mask.shape[-1]):
                cv2.imwrite(os.path.join(save_dir, classes[i] + '.png'),
                            pred_mask[:, :, i] * 255)

    if show:
        plot_result(img_ori, mask, cover)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference a segmentatation model')
    parser.add_argument('config', type=str,
                        help='config file path')
    parser.add_argument('checkpoint', type=str,
                        help='checkpoint file path')
    parser.add_argument('image', type=str,
                        help='input image path')
    parser.add_argument('--show', action='store_true',
                        help='show result images on screen')
    parser.add_argument('--out', default='./result',
                        help='folder to store result images')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    multi_label = cfg.get('multi_label', False)
    inference_cfg = cfg['inference']
    common_cfg = cfg.get('common')

    runner = InferenceRunner(inference_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_shape = image.shape[:2]
    dummy_mask = np.zeros(image_shape)

    output = runner(image, [dummy_mask])
    if multi_label:
        output = output.transpose((1, 2, 0))

    output = inverse_pad(output, image_shape)

    result(args.image, output, multi_label=multi_label,
           classes=CLASSES, palette=PALETTE, show=args.show,
           out=args.out)


if __name__ == '__main__':
    main()
