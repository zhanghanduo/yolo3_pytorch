# coding='utf-8'
import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil
from PIL import Image, ImageDraw

import torch
import torch.nn as nn

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.coco_dataset import COCODataset
from common.bdd_dataset import BDDDataset
from common.utils import non_max_suppression, bbox_iou, load_class_names, plot_boxes, image2torch


def detect(config):
    is_training = False
    # Load and initialize network
    net = ModelMain(config, is_training=is_training)
    net.train(is_training)

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)
    else:
        logging.warning("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    # Load tested img
    imgfile = config["img_path"]
    img = Image.open(imgfile).convert('RGB')
    resized = img.resize((config["img_w"], config["img_h"]))
    input = image2torch(resized)
    input = input.to(torch.device("cuda"))

    start = time.time()
    outputs = net(input)
    output_list = []
    for i in range(3):
        output_list.append(yolo_losses[i](outputs[i]))
    output = torch.cat(output_list, 1)
    output = non_max_suppression(output, config["yolo"]["classes"], conf_thres=0.5, nms_thres=0.4)
    finish = time.time()

    print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    namefile = config["classname_path"]
    class_names = load_class_names(namefile)
    plot_boxes(img, output, 'predictions.jpg', class_names)


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    if len(sys.argv) != 2:
        logging.error("Usage: python detect.py params.py")
        sys.exit()
    params_path = sys.argv[1]
    if not os.path.isfile(params_path):
        logging.error("no params file found! path: {}".format(params_path))
        sys.exit()
    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    detect(config)

    # clean cuda memory
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
