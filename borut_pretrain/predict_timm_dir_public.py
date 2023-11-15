"""
Author: HanChen
Date: 21.06.2022
"""
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import argparse
import json
import wandb

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
from pretrain_timm import MyModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluating network")

    parser.add_argument(
        "--root_path",
        default="/ceph/hpc/data/st2207-pgp-users/DataBase/DeepFake/DFGC-2022/Detection Dataset/10-frames-Public-faces",
        # parser.add_argument('--root_path', default='/ceph/hpc/data/st2207-pgp-users/DataBase/DeepFake/DFGC-2022/Detection Dataset/10-frames-Private-faces',
        type=str,
        help="path to Evaluating dataset",
    )

    parser.add_argument("--save_path", type=str, default="./save_result/txt")
    # parser.add_argument(
    #     "--model_name",
    #     type=str,
    #     default="swin_large_patch4_window12_384.ms_in22k_ft_in1k",
    # )
    # parser.add_argument('--pre_trained', type=str, default='./save_result/models/efficientnet-b4.pth')
    # parser.add_argument('--pre_trained', type=str, default='./models/swin_large_patch4_window12_384_in22k_40.pth')
    parser.add_argument(
        "--pre_trained_dir",
        type=str,
        default="/ceph/hpc/data/st2207-pgp-users/models_luka/",
    )
    # parser.add_argument('--output_txt', type=str, default='./save_result/pred_efn.txt')
    parser.add_argument(
        "--output_txt", type=str, default="./save_result/pred_swin55_public.txt"
    )
    # parser.add_argument('--output_txt', type=str, default='./save_result/pred_swin6_private.txt')

    parser.add_argument(
        "--latest_only",
        action="store_true",
        default=False,
        help="evaluate the latest model only",
    )

    #run_id
    parser.add_argument("--run_id", type=str, default="ys59z47m")


    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--val_batch_size", type=int, default=24)

    parser.add_argument("--from_epoch", type=int, default=0)


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

    project_name = "borut_pretrain"
    run_id = args.run_id   

    # Authenticate to wandb
    wandb.login()

    # Connect to the specified project
    api = wandb.Api(timeout=30)
    

    #get details of the run
    run = api.run(f"{project_name}/{run_id}")
    model_name = run.config['model_name']
    print("wandb run", run_id)
    print("model_name", model_name)

    # _, transform_test = build_transforms(args.resolution, args.resolution,
    #                     max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225])

    # ptl
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
    trained_models = os.listdir(os.path.join(args.pre_trained_dir, model_name,run_id))
    # print("trained_models", trained_models)
    # trained_models = ['swin_large_patch4_window12_384_in22k_0.pth']
    # trained_models = ["swin_large_patch4_window12_384_in22k_40.pth", "swin_large_patch4_window12_384_in22k.pth"]
    path = os.path.abspath(os.path.join(args.pre_trained_dir, model_name,run_id))
    # keep only .pth files
    files = [os.path.join(path, f) for f in trained_models if f.endswith(".ckpt")]

    files.sort(key=os.path.getctime)
    print("found", files)

    if args.latest_only:
        files = files[-1:]
        print("files", files)


    


    for trained in files:
        # pre_trained = os.path.join(args.pre_trained_dir, trained)
        pre_trained = trained
        model = load_network(model, pre_trained).cuda()
        model.train(False)
        model.eval()

        output_txt = args.output_txt

        # swin_large_patch4_window12_384.ms_in22k_ft_in1k-epoch=15-train_loss=0.03.ckpt

        epoch = trained.split("/")[-1].split("-")[1].split("=")[-1]
        print("epoch", epoch)

        # epoch = trained.split('_')[-1].split('.')[0]
        if epoch.isnumeric():

            if args.from_epoch:
                if int(epoch) < args.from_epoch:
                    print("skipping", epoch)
                    continue


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

        gt_labels, prediction_scores = print_ERR.init(output_txt, "Public Label.json")
        # gt_labels,prediction_scores = print_ERR.init(args.output_txt,"Private Label.json")
        score = print_ERR.printAUC(gt_labels, prediction_scores)
        print("AUC score for model %s is %f" % (pre_trained, score))

        # log to wandb for each epoch

        best_auc = run.summary.get("best_auc", 0)
        if score >= best_auc:
            run.summary["best_auc"] = score
            run.summary["best_auc_epoch"] = epoch
            run.summary.update()
            print("new best_auc", score)
        

        
        


        

    




        



if __name__ == "__main__":
    args = parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    main()
