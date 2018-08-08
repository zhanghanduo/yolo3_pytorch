TRAINING_PARAMS = \
    {
        "model_params": {
            "backbone_name": "darknet_53",
            "backbone_pretrained": "../weights/darknet53_weights_pytorch.pth",  # set empty to disable
        },
        "yolo": {
            "anchors": [[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]],
            "classes": 10,
        },
        "lr": {
            "backbone_lr": 0.001,
            "other_lr": 0.01,
            "freeze_backbone": False,  # freeze backbone weights to finetune
            "decay_gamma": 0.1,
            "decay_step": 20,  # decay lr in every ? epochs
        },
        "optimizer": {
            "type": "sgd",
            "weight_decay": 4e-05,
        },
        "batch_size": 16,
        # "train_path": "../data/train_kitti.txt",
        "train_path": "/media/hd/Dataset/bdd100k/bdd-data/train.txt",
        "epochs": 200,
        "img_h": 416,
        "img_w": 416,
        "parallels": [0],  # [0, 1, 2, 3],  # config GPU device
        "working_dir": "/home/hd/python_project/pytorch/YOLOv3_PyTorch",  # replace with your working dir
        # "pretrain_snapshot": "../darknet_53/size416x416_try0/20180621120941/model_0.pth",  # load checkpoint
        "pretrain_snapshot": "../darknet_53/size416x416_try0/20180621153005/model_47_4000.pth",  # load checkpoint
        "evaluate_type": "",
        "try": 0,
        "export_onnx": False,
    }
