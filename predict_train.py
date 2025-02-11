print("hello")
import os
import warnings

print("importing modules")
from dataset_tool import (
    RandomSeqFaceFramesDataset,
    FaceFramesSeqPredictionDataset_all_frames,
)
from dataset_tool import build_transforms

print("imported dataset_1")
import torch
import torch.nn as nn

print("imported torch")
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

print("imported pytorch_lightning")
# WandbLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

print("imported pytorch_lightning callbacks")
import argparse

from scipy.stats import pearsonr, spearmanr

from timm import create_model

print("imported timm")
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
import wandb
import random
import numpy as np
from lightning.pytorch import seed_everything

# tqdm
from tqdm import tqdm


from train_timm_loss_NORM2 import TimmModel
import timm



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the model with the given parameters."
    )

    parser.add_argument(
        "--dataset_root",
        default="/ceph/hpc/data/st2207-pgp-users/ldragar/ds3", 
        help="Path to the dataset",
    )
    parser.add_argument(
        "--labels_dir", default="./label/", help="Path to the labels directory."
    )
    # parser.add_argument('--cp_save_dir', default='/d/hpc/projects/FRI/ldragar/checkpoints/', help='Path to save checkpoints.')
    parser.add_argument(
        "--model_dir",
        default="./timm_models_images/",
        help="Path to save the final model.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--seq_len", type=int, default=5, help="Sequence length.")
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed. for reproducibility. -1 for no seed.",
    )
    parser.add_argument(
        "--cp_id",
        default="37orwro0",
        help="id(wandb_id) of the checkpoint to load from the model_dir.",
    )

    #model_name
    # parser.add_argument(
    #     "--model_name",
    #     default="convnext_xlarge_384_in22ft1k",
    #     help="model_name.",
    # )
    


    parser.add_argument(
        "--out_predictions_dir",
        default="./predictions/",
        help="Path to save the predictions.",
    )
    parser.add_argument(
        "--test_labels_dir",
        default="./competition_end_groundtruth/",
        help="Path to the test labels directory.",
    )
    # stage array
    parser.add_argument(
        "--max_frames",
        type=int,
        default=250,
        help="max frames to use for each video",
    )

    args = parser.parse_args()

    print("starting")
    dataset_root = args.dataset_root
    # labels_dir = args.labels_dir
    model_dir = args.model_dir
    batch_size = args.batch_size
    seq_len = args.seq_len
    seed = args.seed
    cp_id = args.cp_id
    out_predictions_dir = args.out_predictions_dir

    # cp_save_dir = args.cp_save_dir
    test_labels_dir = args.test_labels_dir

    if not seed == -1:
        seed_everything(seed, workers=True)


    project_name = "BeyondDetection"
    run_id = args.cp_id   

    # Authenticate to wandb
    wandb.login()

    # Connect to the specified project
    api = wandb.Api(timeout=30)
    

    #get details of the run
    run = api.run(f"{project_name}/{run_id}")
    model_name = run.config['model_name']
    print("wandb run", run_id)
    print("model_name", model_name)


    model = TimmModel(model_name=model_name)

    

    resolution = model.backbone.pretrained_cfg['input_size'][1] #pretrained_cfg since we are using pretrained model


    data_cfg = timm.data.resolve_data_config(model.backbone.pretrained_cfg)
    print("Timm data_cfg", data_cfg)

    #get std 
    norm_std = data_cfg["std"]
    print("using norm_std", norm_std)
    norm_mean = data_cfg["mean"]
    print("using norm_mean", norm_mean)

    #important USE TIMM TRANSFORMS! https://huggingface.co/docs/timm/quickstart
    _, transform_test = build_transforms(
        resolution,
        resolution,
        max_pixel_value=255.0,
        norm_mean=[norm_mean[0], norm_mean[1], norm_mean[2]],
        norm_std=[norm_std[0], norm_std[1], norm_std[2]],
      
    )


    _, transform_test_LR = build_transforms(
        resolution,
        resolution,
        max_pixel_value=255.0,
        norm_mean=[norm_mean[0], norm_mean[1], norm_mean[2]],
        norm_std=[norm_std[0], norm_std[1], norm_std[2]],
        test_lr_flip=True,
    )

    # load checkpoint
    # get files in dir
    files = os.listdir(model_dir)
    # get the one with the same run id
    cp_name = [f for f in files if cp_id in f][0]

    print(f"loading model from checkpoint {cp_name}")
    if not cp_name.endswith(".ckpt"):
        # this is a pt file
        cp_name = os.path.join(model_dir, cp_name, cp_name + ".pt")

    # load checkpoint
    if cp_name.endswith(".ckpt"):
        # load checkpoint
        checkpoint = torch.load(os.path.join(model_dir, cp_name))
        model.load_state_dict(checkpoint["state_dict"])

    else:
        # load pt file
        model.load_state_dict(torch.load(cp_name))

    # PREDICTIONS
    model.eval()  # set the model to evaluation mode

    model = model.to("cuda:0")

    stages = ["1", "2", "3"]

    resultsdir = os.path.join(out_predictions_dir, cp_id, str(seed))
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir, exist_ok=True)

    class Result:
        def __init__(self, test1, test2, test3, name, fn1, fn2, fn3):
            self.test1 = test1
            self.test2 = test2
            self.test3 = test3
            self.name = name
            self.fn1 = fn1
            self.fn2 = fn2
            self.fn3 = fn3
            self.summary = None
            self.weight = None

        def set_summary(self, summary):
            self.summary = summary

        def set_weight(self, weight):
            self.weight = weight

    def score_model(predictions, groundtruth):
        pearson_corr_coef = pearsonr(predictions, groundtruth)[0]
        spearman_corr_coef = spearmanr(predictions, groundtruth)[0]
        mse = np.mean((predictions - groundtruth) ** 2)
        rmse = np.sqrt(mse)
        return {
            "pearson_corr_coef": pearson_corr_coef,
            "spearman_corr_coef": spearman_corr_coef,
            "rmse": rmse,
        }

    def final_score(predictions, groundtruth):
        scores = []
        for i in range(3):
            scores.append(score_model(predictions[i], groundtruth[i]))
        return (
            scores[0]["pearson_corr_coef"]
            + scores[0]["spearman_corr_coef"]
            + scores[1]["pearson_corr_coef"]
            + scores[1]["spearman_corr_coef"]
            + scores[2]["pearson_corr_coef"]
            + scores[2]["spearman_corr_coef"]
        ) / 6

    t1 = []
    t1_lr = []
    t1_non_flip = []
    t2 = []
    t2_lr = []
    t2_non_flip = []
    t3 = []
    t3_lr = []
    t3_non_flip = []

    gt1 = []
    gt2 = []
    gt3 = []

    for stage in stages:
        name = "Test" + stage + "-labels.txt"

        # Initialize lists to store the predictions and the test names
        all_test_labels = []
        all_test_labels_lr = []
        all_test_names = []
        all_test_gt = []
        all_test_std = []
        all_test_std_lr = []
        min_test_frames_scores = []

       
        ds = FaceFramesSeqPredictionDataset_all_frames(
            os.path.join(test_labels_dir, name),
            dataset_root,
            transform=transform_test,
            max_frames=args.max_frames,
        )
        ds_lr = FaceFramesSeqPredictionDataset_all_frames(
            os.path.join(test_labels_dir, name),
            dataset_root,
            transform=transform_test_LR,
            max_frames=args.max_frames,
        )

        print(f"loaded {len(ds)} test examples")

       
        dl = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        dl_lr = DataLoader(
            ds_lr,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        with torch.no_grad():
            # use tqdm

            with open(
                os.path.join(
                    resultsdir, "frame_predictions_Test" + stage + "_preds.txt"
                ),
                "w",
            ) as f:

                with open(
                    os.path.join(
                        resultsdir, "frame_predictions_Test" + stage + "_preds_lr.txt"
                    ),
                    "w",
                ) as f_lr:
                
                    for (sequences, gt, name), (sequences_lr, _, _) in tqdm(
                        zip(dl, dl_lr), desc=f"Predicting {name}", total=len(dl)
                    ):
                        print(f"predicting {name}")
                        # print("sequences", sequences.shape)

                        # sequences_lr, _, _ = next(iter(dl_lr))

                        predictions = []

                        # make predictions for each frame in the video
                        # make a batch of size number of frames in the video

                        sequences = sequences.permute(1, 0, 2, 3, 4)
                        sequences_lr = sequences_lr.permute(1, 0, 2, 3, 4)

                        # sequences_lr = torch.flip(sequences, [3])

                        print("sequences", sequences.shape)

                        sequences = sequences.to(model.device)
                        y = model(sequences)
                        y = y.cpu().numpy()
                        torch.cuda.empty_cache()  # Clear GPU cache

                        sequences_lr = sequences_lr.to(model.device)
                        y_lr = model(sequences_lr)
                        y_lr = y_lr.cpu().numpy()
                        torch.cuda.empty_cache()  # Clear GPU cache

                        # print("y", y)
                        # print("y shape", y.shape)
                        # remove batch dim

                        predictions = y.squeeze(1)
                        predictions_lr = y_lr.squeeze(1)

                    
                        # Compute mean and standard deviation of predictions
                        mean_prediction = np.mean(predictions)
                        mean_prediction_lr = np.mean(predictions_lr)

                    

                        all_test_labels.append(mean_prediction.item())
                        all_test_labels_lr.append(mean_prediction_lr.item())
                        all_test_names.append(name[0])
                        all_test_gt.append(gt.item())
                    

                        # save predictions for each frame 
                        f.write(f"{name[0]},{ ','.join([str(x) for x in predictions]) }\n")
                        #save predictions for each frame lr
                        f_lr.write(f"{name[0]},{ ','.join([str(x) for x in predictions_lr]) }\n")

            print(f"predicted {len(all_test_labels)} labels for {name}")

     
        mean_test_labels = all_test_labels
        mean_test_labels_lr = all_test_labels_lr
       
        print(f"all_test_gt: {all_test_gt}")
        mean_test_gt = all_test_gt


       
        # Save the mean predictions to a file
        with open(
            os.path.join(resultsdir, "Test" + stage + "non_flip_preds.txt"), "w"
        ) as f:
            for i in range(len(all_test_names)):
                f.write(f"{all_test_names[i]},{mean_test_labels[i]}\n")

        with open(
            os.path.join(resultsdir, "Test" + stage + "flip_preds.txt"), "w"
        ) as f:
            for i in range(len(all_test_names)):
                f.write(f"{all_test_names[i]},{mean_test_labels_lr[i]}\n")

        # combine predictions from non flip and lr flip
        mean_test_labels2 = []
        for i in range(len(all_test_names)):
            mean_test_labels2.append((mean_test_labels[i] + mean_test_labels_lr[i]) / 2)

        with open(os.path.join(resultsdir, "Test" + stage + "_preds_nonflip_flip_combined_meanonly.txt"), "w") as f:
            for i in range(len(all_test_names)):
                f.write(f"{all_test_names[i]},{mean_test_labels2[i]}\n")

 
        if stage == "1":
            t1 = np.array(mean_test_labels2) #combined  mean of lr and non flip 
            t1_lr = np.array(mean_test_labels_lr)
            t1_non_flip = np.array(mean_test_labels)
            gt1 = np.array(mean_test_gt)

        if stage == "2":
            t2 = np.array(mean_test_labels2)
            t2_lr = np.array(mean_test_labels_lr)
            t2_non_flip = np.array(mean_test_labels)
            gt2 = np.array(mean_test_gt)

        if stage == "3":
            t3 = np.array(mean_test_labels2)
            t3_lr = np.array(mean_test_labels_lr)
            t3_non_flip = np.array(mean_test_labels)
            gt3 = np.array(mean_test_gt)

        print(
            f"saved {len(mean_test_labels)} predictions to {os.path.join(resultsdir, 'Test'+stage+'_preds.txt')}"
        )

    # score model
    print("scoring model")
    test_set1_score = score_model(t1, gt1)
    test_set2_score = score_model(t2, gt2)
    test_set3_score = score_model(t3, gt3)
    final_score__nonflip_flip_combined_meanonly = final_score([t1, t2, t3], [gt1, gt2, gt3])
    print("test_set1_score_nonflip_flip_combined_meanonly", test_set1_score)
    print("test_set2_score_nonflip_flip_combined_meanonly", test_set2_score)
    print("test_set3_score_nonflip_flip_combined_meanonly", test_set3_score)
    print("final_score_nonflip_flip_combined_meanonly", final_score__nonflip_flip_combined_meanonly)

    # score with non flip
    test_set1_score_non_flip = score_model(t1_non_flip, gt1)
    test_set2_score_non_flip = score_model(t2_non_flip, gt2)
    test_set3_score_non_flip = score_model(t3_non_flip, gt3)
    final_score_non_flip = final_score(
        [t1_non_flip, t2_non_flip, t3_non_flip], [gt1, gt2, gt3]
    )
    print("test_set1_score_non_flip", test_set1_score_non_flip)
    print("test_set2_score_non_flip", test_set2_score_non_flip)
    print("test_set3_score_non_flip", test_set3_score_non_flip)

    print("final_score_non_flip", final_score_non_flip)

    # score with lr flip
    test_set1_score_lr = score_model(t1_lr, gt1)
    test_set2_score_lr = score_model(t2_lr, gt2)
    test_set3_score_lr = score_model(t3_lr, gt3)
    final_score_lr = final_score([t1_lr, t2_lr, t3_lr], [gt1, gt2, gt3])
    print("test_set1_score_lr", test_set1_score_lr)
    print("test_set2_score_lr", test_set2_score_lr)
    print("test_set3_score_lr", test_set3_score_lr)
    print("final_score_lr", final_score_lr)

    # save to resultsdir
    with open(os.path.join(resultsdir, "scores.txt"), "w") as f:
        f.write(f"cp_id,{cp_id}\n")
        f.write(f"seed,{seed}\n")
        f.write(f"args,{args}\n")

        f.write(f"test_set1_score_nonflip_flip_combined_meanonly,{test_set1_score}\n")
        f.write(f"test_set2_score_nonflip_flip_combined_meanonly,{test_set2_score}\n")
        f.write(f"test_set3_score_nonflip_flip_combined_meanonly,{test_set3_score}\n")

        f.write(f"test_set1_score_lr,{test_set1_score_lr}\n")
        f.write(f"test_set2_score_lr,{test_set2_score_lr}\n")
        f.write(f"test_set3_score_lr,{test_set3_score_lr}\n")
    
        f.write(f"test_set1_score_non_flip,{test_set1_score_non_flip}\n")
        f.write(f"test_set2_score_non_flip,{test_set2_score_non_flip}\n")
        f.write(f"test_set3_score_non_flip,{test_set3_score_non_flip}\n")

        f.write(f"final_score_nonflip_flip_combined_meanonly,{final_score__nonflip_flip_combined_meanonly}\n")
        f.write(f"final_score_lr,{final_score_lr}\n")
        f.write(f"final_score_non_flip,{final_score_non_flip}\n")


    