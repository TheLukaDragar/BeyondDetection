print("hello")
import os
import warnings

print("importing modules")
from dataset_tool import (
    RandomSeqFaceFramesDataset,
    FaceFramesSeqPredictionDataset_middle_frames,
)
from dataset_tool import build_transforms
import math

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


from timm import create_model
import timm

print("imported timm")
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
import wandb
import random
import numpy as np
from lightning.pytorch import seed_everything
import random
import numpy as np
from lightning.pytorch import seed_everything


def train_val_split(dataset, train_prop=0.8, val_prop=0.2, seed=None):
    assert (
        0 <= train_prop <= 1 and 0 <= val_prop <= 1
    ), "Proportions must be between 0 and 1"
    assert round(train_prop + val_prop, 10) == 1, "Proportions must sum to 1"

    total_length = len(dataset)
    train_length = int(train_prop * total_length)
    val_length = int(val_prop * total_length)

    if seed is not None:
        return random_split(
            dataset,
            [train_length, val_length],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        return random_split(dataset, [train_length, val_length])


class TimmModel(pl.LightningModule):
    def __init__(
        self,
        pretrained_backbone=None,
        model_name="convnext_tiny",
        dropout=0.1,
        loss="rmse",
        lr=2e-5,
        drop_path_rate=0.0,
        weight_decay=0.01,
    ):
        super(TimmModel, self).__init__()
        self.model_name = model_name

        # load pretrained deepfake detection model
        self.backbone = create_model(
            self.model_name,
            pretrained=True,
            num_classes=2,
            drop_path_rate=drop_path_rate,
        )

        # get first layer size
        classifier_layer = self.backbone.default_cfg[
            "classifier"
        ]  # This gets 'head.fc'

        # Split the string into parts to traverse
        parts = classifier_layer.split(".")  # Splits into ['head', 'fc']

        # Traverse the model attributes dynamically
        layer = self.backbone
        for part in parts:
            layer = getattr(layer, part)

        # Get the number of input features
        in_feats = layer.in_features
        print("in_feats", in_feats)  # Should print 1536
        # customize classifier

        n_features = in_feats
        # self.backbone.head.fc = nn.Linear(n_features, 2) no need to do this as we are using num_classes=2
        # self.backbone = torch.nn.DataParallel(self.backbone)

        # https://lightning.ai/docs/pytorch/stable/deploy/production_intermediate.html !!!

        if not pretrained_backbone == None:
            if pretrained_backbone.endswith(".ckpt"):
                print("loading from ckpt")
                checkpoint = torch.load(pretrained_backbone)
                model_weights = checkpoint["state_dict"]

                for key in list(model_weights):
                    model_weights[key.replace("model.", "")] = model_weights.pop(key)

                self.backbone.load_state_dict(model_weights)
            elif pretrained_backbone.endswith(".pth"):
                print("loading from pth")
                self.backbone.load_state_dict(torch.load(pretrained_backbone))
                self.backbone = self.backbone.module

        else:
            print("no pretrained backbone loaded")

        # self.backbone.load_state_dict(torch.load(og_path))

        # self.backbone.head.fc = nn.Identity()
        # modify backbone classifier to ouput only features

        # Traverse to the parent layer
        parent_layer = self.backbone
        for part in parts[:-1]:
            parent_layer = getattr(parent_layer, part)

        # Replace the last attribute (head.fc) (the classifier) with nn.Identity
        setattr(parent_layer, parts[-1], nn.Identity())

        print("backbone modified head ", getattr(self.backbone, parts[0]))

        self.drop = nn.Dropout(dropout)

        # one frame feature vector
        self.fc = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 1)
        # self.fc3 = nn.Linear(256, 64)
        # self.fc4 = nn.Linear(64, 1)
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

        # Initialize PearsonCorrCoef metric
        self.pearson_corr_coef = torchmetrics.PearsonCorrCoef().to(self.device)
        self.spearman_corr_coef = torchmetrics.SpearmanCorrCoef().to(self.device)
        # rmse
        self.mse_log = torchmetrics.MeanSquaredError().to(self.device)

        self.loss_fn = None
        self.lr = lr
        self.weight_decay = weight_decay

        if loss == "rmse":
            self.loss_fn = self.RMSE
        elif loss == "mae":
            self.loss_fn = self.MAE

        else:
            raise ValueError("Invalid loss function")

        self.save_hyperparameters()

    def RMSE(self, preds, y):
        mse = self.mse(preds.view(-1), y.view(-1))
        return torch.sqrt(mse)

    def forward(self, x):
        # Choose a random index from the sequence dimension
        random_idx = torch.randint(0, x.shape[1], (x.shape[0],))

        # Select a random frame for each item in the batch
        x_random_frame = x[torch.arange(x.shape[0]), random_idx]

        # Process the selected frame with the backbone
        features = self.backbone(
            x_random_frame
        )  # Output shape: (batch_size, n_features)

        # Optionally, you can continue with further processing as needed
        x = self.drop(features)
        x = torch.nn.functional.relu(self.fc(x))
        # x = torch.nn.functional.relu(self.fc2(x))
        # x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc2(x)
        logit = x

        return logit

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        rmse_loss = self.RMSE(preds, y)
        self.log(
            "train_loss", loss.item(), on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "train_rmse_loss",
            rmse_loss.item(),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        rmse_loss = self.RMSE(preds, y)
        loss_value = loss.item()

        self.log("val_loss", loss.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "val_rmse_loss",
            rmse_loss.item(),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def MAE(self, preds, y):
        return self.mae(preds.view(-1), y.view(-1))

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=2, verbose=True
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [lr_scheduler]

    def get_predictions(self):
        return torch.cat(self.test_preds).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the model with the given parameters."
    )

    parser.add_argument(
        "--dataset_root",
        default="../dataset",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--labels_file",
        default="./label/train_set.csv",
        help="Path to the labels train file.",
    )
    # test_labels_dir
    parser.add_argument(
        "--test_labels_dir",
        default="./competition_end_groundtruth/",
        help="Path to the test labels directory.",
    )
    parser.add_argument(
        "--pretrained_backbone",
        default="./DFGC-1st-2022-model/convnext_xlarge_384_in22ft1k_30.pth",
        help="DFGC1st convnext_xlarge_384_in22ft1k_30.pth file path",
    )
    parser.add_argument('--cp_save_dir', default='./timm_models_images_cps/', help='Path to save checkpoints.')
    parser.add_argument(
        "--final_model_save_dir",
        default="./timm_models_images/",
        help="Path to save the final model.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length.")
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed. for reproducibility. Note final model is worse with seed set.",
    )
    parser.add_argument(
        "--wdb_project_name",
        default="BeyondDetection",
        help="Weights and Biases project name.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Accumulate gradients over n batches.",
    )
    # # experiment_name
    # parser.add_argument(
    #     "--experiment_name",
    #     default="convnext_xlarge_384_in22ft1k",
    #     help="Experiment name.",
    # )

    parser.add_argument(
        "--model_name",
        default="swin_large_patch4_window12_384.ms_in22k_ft_in1k",
        help="timm model name.",
    )
    parser.add_argument(
        "--wandb_resume_version", default="None", help="Wandb resume version."
    )
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes.")
    # devices array
    parser.add_argument(
        "--devices", nargs="+", type=int, default=[0, 1], help="Devices to train on."
    )
    # drop out
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    # max_epochs
    parser.add_argument("--max_epochs", type=int, default=33, help="Max epochs.")

    parser.add_argument(
        "--loss",
        default="rmse",
        help="Loss function to use. Supported values: mse, rmse, mae, norm_loss_with_normalization, opdai, hust",
    )
    # lr
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")

    parser.add_argument(
        "--augmentation",
        type=bool,
        default=False,
        help="Whether to use data augmentation.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Proportion of the dataset to use for validation.",
    )

    parser.add_argument(
        "--from_cp_id",
        default="None",
        help="Resume training from checkpoint id.",
    )
    # augment_prob
    parser.add_argument(
        "--augment_prob", type=float, default=0.5, help="Augmentation probability."
    )
    # drop_path_rate
    parser.add_argument(
        "--drop_path_rate", type=float, default=0.0, help="Drop path rate."
    )

    # weight_decay
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay."
    )

    # parser.add_argument('--test_labels_dir', default='/d/hpc/projects/FRI/ldragar/label/', help='Path to the test labels directory.')

    args = parser.parse_args()

    print("starting")
    batch_size = args.batch_size
    seq_len = args.seq_len
    seed = args.seed
    wdb_project_name = args.wdb_project_name

    # cp_save_dir = args.cp_save_dir
    # test_labels_dir = args.test_labels_dir

    if seed != -1:
        seed_everything(seed, workers=True)

    # seed only data

    # convnext_xlarge_384_in22ft1k
    model = TimmModel(
        pretrained_backbone=args.pretrained_backbone,
        model_name=args.model_name,
        dropout=args.dropout,
        loss=args.loss,
        lr=args.lr,
        drop_path_rate=args.drop_path_rate,
        weight_decay=args.weight_decay,
    )

    resolution = model.backbone.pretrained_cfg["input_size"][
        1
    ]  # pretrained_cfg since we are using pretrained model

    data_cfg = timm.data.resolve_data_config(model.backbone.pretrained_cfg)
    print("Timm data_cfg", data_cfg)

    # get std
    norm_std = data_cfg["std"]
    print("using norm_std", norm_std)
    norm_mean = data_cfg["mean"]
    print("using norm_mean", norm_mean)

    # important USE TIMM TRANSFORMS! https://huggingface.co/docs/timm/quickstart
    transform_train, transform_test = build_transforms(
        resolution,
        resolution,
        max_pixel_value=255.0,
        norm_mean=[norm_mean[0], norm_mean[1], norm_mean[2]],
        norm_std=[norm_std[0], norm_std[1], norm_std[2]],
        augment=args.augmentation,
        augment_prob=args.augment_prob,
    )

    print("loading dataset")

    face_frames_dataset = RandomSeqFaceFramesDataset(
        args.dataset_root,
        args.labels_file,
        transform=transform_train,
        seq_len=seq_len,
        seed=seed if seed != -1 else None,
    )

    print("splitting dataset")
    train_ds, val_ds = train_val_split(
        face_frames_dataset,
        train_prop=(1 - args.val_split),
        val_prop=args.val_split,
        seed=seed if seed != -1 else None,
    )

    print("first 5 train labels")
    for i in range(5):
        print(train_ds[i][1])
    print("first 5 val labels")
    for i in range(5):
        print(val_ds[i][1])

    print("loading dataloader")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # if len(test_dl1) % batch_size != 0 or len(test_dl2) % batch_size != 0 or len(test_dl3) % batch_size != 0:
    #     warnings.warn("Uneven inputs detected! With multi-device settings, DistributedSampler may replicate some samples to ensure all devices have the same batch size. TEST WILL NOT BE ACCURATE CRITICAL!")
    #     exit(1)

    print(
        "train dataset will be split into",
        len(train_dl),
        "batches",
        "each batch will have",
        batch_size,
        "samples",
    )
    print(
        "val dataset will be split into",
        len(val_dl),
        "batches",
        "each batch will have",
        batch_size,
        "samples",
    )
    print(
        "each gpu will process batch size of",
        batch_size,
        "samples",
        "global batch size will be",
        batch_size * len(args.devices),
    )

    if len(train_dl) % batch_size != 0 or len(val_dl) % batch_size != 0:
        warnings.warn(
            "Uneven inputs detected! With multi-device settings, DistributedSampler may replicate some samples to ensure all devices have the same batch size."
        )

    print(f"loaded {len(train_dl)} train batches and {len(val_dl)} val batches")

    # print first train example

    for x, y in train_dl:
        print(x.shape)
        print(y.shape)
        print(y)
        break

    if args.wandb_resume_version == "None":
        wandb_logger = WandbLogger(name=args.model_name, project=wdb_project_name)
    else:
        wandb_logger = WandbLogger(
            name=args.model_name, version=args.wandb_resume_version, resume="must"
        )

    if args.from_cp_id != "None":
        checkpoint_path = os.path.join(
            args.final_model_save_dir, args.from_cp_id, f"{args.from_cp_id}.pt"
        )
        print(f"loading model from checkpoint {checkpoint_path}")
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)

    wandb_logger.watch(model, log="all", log_freq=100)

    # log args
    wandb_logger.log_hyperparams(args)

    wandb_run_id = str(wandb_logger.version)
    if wandb_run_id == "None":
        print("no wandb run id this is a copy of model with DDP")

    print("init trainer")

    # save hyperparameters
    wandb_logger.log_hyperparams(model.hparams)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                    dirpath=args.cp_save_dir,
                                        filename=f'{wandb_run_id}-{{epoch:02d}}-{{val_loss:.2f}}', mode='min', save_top_k=1)

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp",
        num_nodes=args.num_nodes,
        devices=args.devices,
        max_epochs=args.max_epochs,
        log_every_n_steps=200,
        callbacks=[
            # EarlyStopping(monitor="val_loss",
            #             mode="min",
            #             patience=4,
            #             ),
            checkpoint_callback
        ],
        logger=wandb_logger,
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=seed != -1,
    )

    print("start training")
    # Train the model
    trainer.fit(model, train_dl, val_dl)
    # Test the model

    # model.current_test_set = "test_set1"
    # trainer.test(model, test_dl1)
    # model.current_test_set = "test_set2"
    # trainer.test(model, test_dl2)
    # model.current_test_set = "test_set3"
    # trainer.test(model, test_dl3)

    if trainer.global_rank == 0:
        # test_set1_score = wandb_logger.experiment.summary["test_set1_score"]
        # test_set2_score = wandb_logger.experiment.summary["test_set2_score"]
        # test_set3_score = wandb_logger.experiment.summary["test_set3_score"]
        # avg_score = (test_set1_score + test_set2_score + test_set3_score) / 3
        # print(f"test_set1_score: {test_set1_score}")
        # print(f"test_set2_score: {test_set2_score}")
        # print(f"test_set3_score: {test_set3_score}")
        # print(f"final_score: {avg_score}")

        # Log the average score to WandB
        # wandb.log({"final_score": avg_score})
        # save model
        model_path = os.path.join(args.final_model_save_dir, wandb_run_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(model.state_dict(), os.path.join(model_path, f"{wandb_run_id}.pt"))
        print(f"finished training, saved last model to {model_path}")

        #get best checkpoint
        best_checkpoint_path = checkpoint_callback.best_model_path
        print(f"best checkpoint is saved in path {best_checkpoint_path}")
        # load best checkpoint

    
        #make a new model that is a copy of the old one but with the best checkpoint weights
        best_model = TimmModel(
            pretrained_backbone=args.pretrained_backbone,
            model_name=args.model_name,
            dropout=args.dropout,
            loss=args.loss,
            lr=args.lr,
            drop_path_rate=args.drop_path_rate,
            weight_decay=args.weight_decay,
        )
        best_model.load_state_dict(torch.load(best_checkpoint_path)["state_dict"])
        print("loaded best checkpoint")

        

        # PREDICTIONS
        model.eval()  # set the model to evaluation mode



        model = model.to("cuda:0")
        best_model = best_model.to("cuda:1")

        stages = ["1", "2", "3"]
        # get wandb run id

        resultsdir = os.path.join("./results/", wandb_run_id)
        if not os.path.exists(resultsdir):
            os.makedirs(resultsdir)

        scores = []
        scores2 = []


        for stage in stages:
            name = "test_set" + stage + ".txt"
            test_labels = []
            test_labels2 = []
            test_names = []
            test_names2 = []
            test_gt= []
            test_gt2 = []

            # use seq len

            ds = FaceFramesSeqPredictionDataset_middle_frames(
                os.path.join(args.test_labels_dir, "Test" + stage + "-labels.txt"),
                args.dataset_root,
                transform=transform_test,
                seq_len=1,
                seed=seed if seed != -1 else None,
            )
            print(f"loaded {len(ds)} test examples")

            with torch.no_grad():
                for x, gt, nameee in ds:
                    x = x.unsqueeze(0).to(model.device)
                    x2 = x.clone().to(best_model.device)
                    # print("x.shape)", x.shape)
                    y = model(x)
                    y2 = best_model(x2)
                    # print("y.shape", y.shape)
                    # print("y", y)
                    y = y.cpu().numpy()
                    y = y[0][0]

                    y2 = y2.cpu().numpy()
                    y2 = y2[0][0]


                    test_gt.append(gt)
                    test_labels.append(y)
                    test_names.append(nameee)

                    test_gt2.append(gt)
                    test_labels2.append(y2)
                    test_names2.append(nameee)


            # compute score for test set
            test_labels = torch.tensor(test_labels).to(model.device)
            test_gt = torch.tensor(test_gt).to(model.device)
            test_names = np.array(test_names)
            test_labels = test_labels.view(-1)

            test_labels2 = torch.tensor(test_labels2).to(best_model.device)
            test_gt2 = torch.tensor(test_gt2).to(best_model.device)
            test_names2 = np.array(test_names2)
            test_labels2 = test_labels2.view(-1)



            plcc = model.pearson_corr_coef(test_labels, test_gt)
            spearman = model.spearman_corr_coef(test_labels, test_gt)
            mse_log = model.mse_log(test_labels, test_gt)
            rmse = torch.sqrt(mse_log)

            plcc2 = best_model.pearson_corr_coef(test_labels2, test_gt2)
            spearman2 = best_model.spearman_corr_coef(test_labels2, test_gt2)
            mse_log2 = best_model.mse_log(test_labels2, test_gt2)
            rmse2 = torch.sqrt(mse_log2)

            

            print(f"test_set{stage}_plcc: {plcc}")
            print(f"test_set{stage}_spearman: {spearman}")
            print(f"test_set{stage}_rmse: {rmse}")

            print(f"test_set{stage}_plcc_best_cp: {plcc2}")
            print(f"test_set{stage}_spearman_best_cp: {spearman2}")
            print(f"test_set{stage}_rmse_best_cp: {rmse2}")



            # save to wandb
            wandb.log({f"final_test_set{stage}_plcc": plcc})
            wandb.log({f"final_test_set{stage}_spearman": spearman})
            wandb.log({f"final_test_set{stage}_rmse": rmse})
            wandb.log({f"final_test_set{stage}_score": (plcc + spearman) / 2})

            wandb.log({f"final_test_set{stage}_plcc_best_cp": plcc2})
            wandb.log({f"final_test_set{stage}_spearman_best_cp": spearman2})
            wandb.log({f"final_test_set{stage}_rmse_best_cp": rmse2})
            wandb.log({f"final_test_set{stage}_score_best_cp": (plcc2 + spearman2) / 2})


            scores.append((plcc + spearman) / 2)
            scores2.append((plcc2 + spearman2) / 2)

            print(f"predicted {len(test_labels)} labels for {name}")
            print(f"len test_names {len(test_names)}")
            print(f"len test_labels {len(test_labels)}")

            # save to file with  Test1_preds.txt, Test2_preds.txt, Test3_preds.txt
            # name, label
            with open(
                os.path.join(resultsdir, "Test" + stage + "_preds.txt"), "w"
            ) as f:
                for i in range(len(test_names)):
                    f.write(f"{test_names[i]},{test_labels[i]}\n")

            print(
                f"saved {len(test_labels)} predictions to {os.path.join(resultsdir, 'Test'+stage+'_preds.txt')}"
            )

            with open(
                os.path.join(resultsdir, "Test" + stage + "_preds_best_cp.txt"), "w"
            ) as f:
                for i in range(len(test_names2)):
                    f.write(f"{test_names2[i]},{test_labels2[i]}\n")

            print(
                f"saved {len(test_labels2)} predictions to {os.path.join(resultsdir, 'Test'+stage+'_preds_best_cp.txt')}"
            )
 

        print(f"final_score: {sum(scores)/3}")
        print(f"final_score_best_cp: {sum(scores2)/3}")
        wandb.log({"final_final_score": sum(scores) / 3})
        wandb.log({"final_final_score_best_cp": sum(scores2) / 3})

        print("done")

    else:
        print("not rank 0 skipping predictions")
