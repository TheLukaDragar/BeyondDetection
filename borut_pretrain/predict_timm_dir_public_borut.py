# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import argparse
import json

import numpy as np

# import pandas as pd
import torchvision.models as models
import timm

from transforms import build_transforms
from metrics import get_metrics
from data_utils import images_Dataloader_all

# from network.models import get_swin_transformers

import print_ERR_from_csv as print_ERR
from tqdm import tqdm

import os
import sys
import timm
import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse
import sys

import json
import random
import numpy as np
from tqdm import tqdm
import time
import torch.nn as nn
from logger import create_logger

from transforms import build_transforms
from metrics import get_metrics
from dataset import binary_Rebalanced_Dataloader
from torch.utils.data import DataLoader


import os
from torch.utils.data import WeightedRandomSampler
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import pytorch_lightning as pl
from pytorch_lightning import (
    LightningModule,
    LightningDataModule,
    Trainer,
    seed_everything,
)

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint



class MyModel(LightningModule):
    def __init__(self, model_name, num_class=2, learning_rate=0.00005):
        super(MyModel, self).__init__()
        self.model = timm.create_model(
            model_name, num_classes=num_class, pretrained=True
        )
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        (
            images,
            labels,
        ) = batch  # TODO optional look into the performance on each database
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate
        )
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)
        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=2, verbose=True
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluating network")

    parser.add_argument(
        "--root_path",
        default="/ceph/hpc/data/st2207-pgp-users/DataBase/DeepFake/DFGC-2022/Detection Dataset/10-frames-Public-faces",
        # parser.add_argument('--root_path', default='/ceph/hpc/data/st2207-pgp-users/DataBase/DeepFake/DFGC-2022/Detection Dataset/10-frames-Private-faces',
        type=str,
        help="path to Evaluating dataset",
    )

    parser.add_argument("--checkpoint_path", type=str, default="/ceph/hpc/data/st2207-pgp-users/models_luka/convnext_xlarge_384_in22ft1k/91z6toc7/convnext_xlarge_384_in22ft1k-epoch=02-val_loss=0.07-train_loss=0.08.ckpt")



    parser.add_argument(
        "--output_txt", type=str, default="predictions.txt", help="output txt file with predictions"
    )

    parser.add_argument("--labels", type=str, default="Public Label.json")

    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--val_batch_size", type=int, default=24)



    args = parser.parse_args()
    return args


def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename)["state_dict"])
    return network


def test_model(model, dataloaders):
    prediction = np.array([])
    model.train(False)
    for images in dataloaders:
        input_images = Variable(images.cuda())
        outputs = model(input_images)
        pred_ = torch.nn.functional.softmax(outputs, dim=-1)
        pred_ = pred_.cpu().detach().numpy()[:, 1]

        prediction = np.insert(prediction, 0, pred_)
    return prediction


def main():
    args = parse_args()

    test_videos = os.listdir(args.root_path)



    #extract model name from checkpoint path
    model_name = args.checkpoint_path.split("/")[-1].split("-")[0]
    print("model_name", model_name)


    #create model
    model = MyModel(model_name=model_name)

    # get resolution from model
    resolution = model.model.pretrained_cfg["input_size"][
        1
    ]  # pretrained_cfg since we are using pretrained model

    print("resolution", resolution)

    data_cfg = timm.data.resolve_data_config(model.model.pretrained_cfg)
    print("Timm data_cfg", data_cfg)

    # get std
    norm_std = data_cfg["std"]
    print("using norm_std", norm_std)
    norm_mean = data_cfg["mean"]
    print("using norm_mean", norm_mean)

    # important USE TIMM TRANSFORMS! https://huggingface.co/docs/timm/quickstart
    _, transform_test = build_transforms(
        resolution,
        resolution,
        max_pixel_value=255.0,
        norm_mean=[norm_mean[0], norm_mean[1], norm_mean[2]],
        norm_std=[norm_std[0], norm_std[1], norm_std[2]],
    )

    # load saved model
    checkpoint_path = args.checkpoint_path

    #load from cp file
    model = load_network(model, checkpoint_path).cuda()
    model.train(False)
    model.eval()

    output_txt = args.output_txt

    # swin_large_patch4_window12_384.ms_in22k_ft_in1k-epoch=15-train_loss=0.03.ckpt

    epoch = checkpoint_path.split("/")[-1].split("-")[1].split("=")[-1]
    print("epoch", epoch)

    # epoch = trained.split('_')[-1].split('.')[0]
    if epoch.isnumeric():

        output_txt = args.output_txt[:-4] + "_" + epoch + ".txt"
    result_txt = open(output_txt, "w", encoding="utf-8")

    for idx, video_name in enumerate(tqdm(test_videos)):
        video_path = os.path.join(args.root_path, video_name)
        test_dataset = images_Dataloader_all(video_path, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.val_batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        with torch.no_grad():
            prediction = test_model(model, test_loader)

        if len(prediction) != 0:
            video_prediction = np.mean(prediction, axis=0)
        else:
            video_prediction = 0.5  # default is 0.5
        # print(video_name + '  is  fake' if np.round(video_prediction) == 1
        #    else video_name + '  is  real')
        # print('Probs %f' % video_prediction)

        result_txt.write(video_name + ".mp4, %f" % video_prediction + "\n")
    result_txt.close()

    gt_labels, prediction_scores = print_ERR.init(output_txt, args.labels)
    # gt_labels,prediction_scores = print_ERR.init(args.output_txt,"Private Label.json")
    score = print_ERR.printAUC(gt_labels, prediction_scores)
    print("AUC score for model %s is %f" % (checkpoint_path, score))

if __name__ == "__main__":
    args = parse_args()
    main()
