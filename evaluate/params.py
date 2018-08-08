TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 10,
    },
    "batch_size": 16,
    "iou_thres": 0.5,
    "val_path": "/media/hd/Dataset/bdd100k/bdd-data/val.txt",
    "annotation_path": "../data/coco/annotations/instances_val2014.json",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    "pretrain_snapshot": "../darknet_53/size416x416_try0/20180621153005/model_47_4000.pth",  # load checkpoint
    # "pretrain_snapshot": "../weights/yolov3_weights_pytorch.pth",  # load original checkpoint (darknet53 as backbone)
}
