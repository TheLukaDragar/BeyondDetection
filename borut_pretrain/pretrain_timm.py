"""
Author: HanChen
Date: 21.06.2022
"""
# -*- coding: utf-8 -*-
print("start")
import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse


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
from pytorch_lightning import LightningModule, LightningDataModule, Trainer

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
print("imports done")

######################################################################
# Save model
def save_network(network, save_filename):
    torch.save(network.cpu().state_dict(), save_filename)
    if torch.cuda.is_available():
        network.cuda()


def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename))
    return network

def make_weights_for_balanced_classes(train_dataset):
    targets = []

    targets = torch.tensor(train_dataset)

    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in targets])
    return samples_weight


def parse_args():
    parser = argparse.ArgumentParser(description='Training network')
    parser.add_argument('--root_path_dfdc', default='/ceph/hpc/data/st2207-pgp-users/DataBase/DeepFake/DFDC/train_face_crop_png/20-frames-faces/train/',
                        type=str, help='path to DFDC dataset')    
    parser.add_argument('--root_path', default='/ceph/hpc/data/st2207-pgp-users/DataBase/DeepFake/',
                        type=str, help='path to datasets')    

    #parser.add_argument('--save_path', type=str, default='./save_result2_swin')
    parser.add_argument('--save_path', type=str, default='/ceph/hpc/data/st2207-pgp-users/models_luka')
    #parser.add_argument('--model_name', type=str, default='swin_large_patch4_window12_384_in22k')           
    parser.add_argument('--model_name', type=str, default='swin_large_patch4_window12_384.ms_in22k_ft_in1k')           
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--class_name', type=list, 
                        default=['real', 'fake'])
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--adjust_lr_iteration', type=int, default=30000)
    parser.add_argument('--droprate', type=float, default=0.2)
    parser.add_argument('--base_lr', type=float, default=0.00005)
    #parser.add_argument('--base_lr', type=float, default=1.74E-05)
    parser.add_argument('--batch_size', type=int, default=24)
    # parser.add_argument('--resolution', type=int, default=384) #handled by timm
    parser.add_argument('--val_batch_size', type=int, default=128)

    #every_n_epochs
    parser.add_argument('--save_checkpoint_every_n_epochs', type=int, default=1)

    #wandb
    # parser.add_argument('--experiment_name', type=str, default='swin_large_patch4_window12_384_in22k_40')
    parser.add_argument('--project_name', type=str, default='borut_pretrain')

    #pytorch lightning
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument(
        "--devices", nargs="+", type=int, default=[0,1,2,3], help="Devices to train on."
    )




    args = parser.parse_args()
    return args

def load_txt(txt_path='./txt3', logger=None):
    txt_names = os.listdir(txt_path)
    tmp_videos, tmp_labels = [], []
    tmp_real_frames, tmp_fake_frames = 0, 0
    videos, labels = [], []
    for txt_name in txt_names:
        with open(os.path.join(txt_path, txt_name), 'r') as f:
            videos_names = f.readlines()
            for i in videos_names:
                i = os.path.join(args.root_path, i) ##BB add
                if i.find('landmarks') != -1:
                    continue
                frames = len(os.listdir(i.strip().split()[0]))
                if frames == 0:
                    continue
                label = int(i.strip().split()[1])
                if label == 0:
                    tmp_real_frames+=frames
                else:
                    tmp_fake_frames+=frames
                tmp_videos.append(i.strip().split()[0])
                tmp_labels.append(label)
        timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
        print(timeStr, txt_name+': all videos:'+str(len(tmp_labels))+', FAKE:'+str(sum(tmp_labels))+', ratio(FAKE/ALL): '+str(sum(tmp_labels)/len(tmp_labels))+', REAL frames:'+str(tmp_real_frames)+', FAKE frames:'+str(tmp_fake_frames))
        videos.extend(tmp_videos)
        labels.extend(tmp_labels)
        print(timeStr, len(labels), sum(labels), sum(labels)/len(labels))    
        tmp_videos, tmp_labels = [], []
        tmp_real_frames, tmp_fake_frames = 0, 0
        print('\n')
    return videos, labels


class MyModel(LightningModule):
    def __init__(self, model_name, num_class=2, base_lr=0.00005):
        super(MyModel, self).__init__()
        self.model = timm.create_model(model_name, num_classes=num_class, pretrained=True)
        self.criterion = nn.CrossEntropyLoss()
        self.base_lr = base_lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True,sync_dist=True,on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.base_lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)
        return [optimizer], [scheduler]

class MyDataModule(LightningDataModule):
    def __init__(self, train_dataset, weights_train, batch_size):
        super(MyDataModule, self).__init__()
        self.train_dataset = train_dataset
        self.weights_train = weights_train
        self.batch_size = batch_size

    def train_dataloader(self):
        train_sampler = WeightedRandomSampler(self.weights_train, len(self.train_dataset), replacement=True)
        #shuflle is false because of WeightedRandomSampler OK
        return DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.batch_size,shuffle=False, num_workers=6, pin_memory=True)



def main():
    args = parse_args()

    

     # Logger and Trainer
    wandb_logger = WandbLogger(name=args.model_name, project=args.project_name)

    logger = create_logger(output_dir='%s/report' % args.save_path, name=f"{args.model_name}")
    logger.info('Start Training %s' % args.model_name)
    timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    logger.info(timeStr)  

  
    train_videos, train_labels = [], []
    tmp_real_frames, tmp_fake_frames = 0, 0


     #CACHE train_videos, train_labels
    #check if cache exists
    if os.path.exists("train_labels.txt"):

        print("loading train_labels from cache")
       
        train_videos = []
        train_labels = []
        with open("train_labels.txt", "r") as f:
            for line in f:
                video, label = line.strip().split(',')
                train_videos.append(video)
                train_labels.append(int(label))

        # # #cut to only 10 videos
        train_videos = train_videos[:10]
        train_labels = train_labels[:10]

    else:
        print("loading training videos")
        for idx in tqdm(range(0, 50)): #for idx in tqdm(range(0, 50)):
            tmp_real_frames0, tmp_fake_frames0 = 0, 0
            tmp_real, tmp_fake = 0, 0
            sub_name = 'dfdc_train_part_%d' % idx
            video_sub_path = os.path.join(args.root_path_dfdc, sub_name)
            with open(os.path.join(video_sub_path, 'metadata.json'),'r') as metadata_json:
                metadata = json.load(metadata_json)
            #print(f'dfdc_train_part: {idx}',)
            #print(f'Numbers of videos in metadata: {len(metadata.items())}',)
            for key, value in metadata.items(): 
                if value['label'] == 'FAKE': # FAKE or REAL
                    label = 1
                else:
                    label = 0
                inputPath = os.path.join(args.root_path_dfdc, sub_name, key[:-4])
    #BB in case if dir not exists
                if not os.path.exists(inputPath):
                    print("no dir or video: "+inputPath)
                    continue
                
                frames = len(os.listdir(inputPath))
                if frames == 0:
                    continue

                if label == 0:
                    tmp_real_frames+=frames
                    tmp_real_frames0+=frames
                    tmp_real+=1
                else:
                    tmp_fake_frames+=frames
                    tmp_fake_frames0+=frames
                    tmp_fake+=1

                train_videos.append(inputPath)
                train_labels.append(label)
            #print(f'Numbers of videos in dir: {tmp_real+tmp_fake}',)
            #print(f'Real videos in dir: {tmp_real}(frames:{tmp_real_frames0})',)
            #print(f'Fake videos in dir: {tmp_fake}(frames:{tmp_fake_frames0})\n',)
        timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
        if len(train_labels) != 0:
            print(f'DFDC: {timeStr}, all videos:{len(train_labels)}, FAKE:{sum(train_labels)}, ratio(FAKE/ALL): {sum(train_labels)/len(train_labels)}')
        else:
            print(timeStr, len(train_labels), sum(train_labels))
        print('REAL frames:'+str(tmp_real_frames)+', FAKE frames:'+str(tmp_fake_frames))

        #load txt3
        tmp_videos, tmp_labels = load_txt(txt_path='./txt3')
        train_videos += tmp_videos
        train_labels += tmp_labels

        #save train_labels
        with open("train_labels.txt", "w") as f:
            for video, label in zip(train_videos, train_labels):
                f.write(f"{video},{label}\n")



    timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    print(f'{timeStr}, All Train videos: all:{len(train_labels)}, fake:{sum(train_labels)}, ratio(FAKE/ALL):{sum(train_labels)/len(train_labels)}')

    #ptl 
    model = MyModel(model_name=args.model_name, num_class=args.num_class, base_lr=args.base_lr)
    #get resolution from model
    resolution = model.model.pretrained_cfg['input_size'][1] #pretrained_cfg since we are using pretrained model
    
    print("resolution", resolution)

    wandb_logger.log_hyperparams({"resolution": resolution})

    data_cfg = timm.data.resolve_data_config(model.model.pretrained_cfg)
    print("Timm data_cfg", data_cfg)

    #get std 
    norm_std = data_cfg["std"]
    print("using norm_std", norm_std)
    norm_mean = data_cfg["mean"]
    print("using norm_mean", norm_mean)

    wandb_logger.log_hyperparams({"norm_std": norm_std})
    wandb_logger.log_hyperparams({"norm_mean": norm_mean})



    #important USE TIMM TRANSFORMS! https://huggingface.co/docs/timm/quickstart
    transform_train, transform_test = build_transforms(resolution, resolution, 
                    max_pixel_value=255.0, norm_mean=[norm_mean[0], norm_mean[1], norm_mean[2]], norm_std=[norm_std[0], norm_std[1], norm_std[2]])



    

    train_dataset = binary_Rebalanced_Dataloader(video_names=train_videos, video_labels=train_labels, phase='train', 
                                                num_class=args.num_class, transform=transform_train)

    weights_train = make_weights_for_balanced_classes(train_labels)
    # when using WeightedRandomSampler the samples will be drawn from your dataset using the provided weights, 
    # so you won’t get a specific order. In that sense your data will be shuffled.
    # data_sampler = WeightedRandomSampler(weights_train, len(train_labels), replacement=True)


    timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    print(timeStr, 'All Train videos Number: %d' % (len(train_dataset)))
    
    # create model
    # model = timm.create_model(model_name=args.model_name, num_classes=args.num_class, pretrained=True)
    # model = nn.DataParallel(model).cuda()


    # load saved model
    #model = load_network(model, 'save_result_swin/models/swin_large_patch4_window12_384_in22k_30.pth').cuda()
    #model = nn.DataParallel(model).cuda()

    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.base_lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)
    # criterion = torch.nn.CrossEntropyLoss()

    print("batch_size =",args.batch_size)
    # train_loader = torch.utils.data.DataLoader(train_dataset, sampler=data_sampler, batch_size=args.batch_size, drop_last=True,
    #                                            shuffle=False, num_workers=6, pin_memory=True)


  
    data_module = MyDataModule(train_dataset, weights_train, args.batch_size)

   
    wandb_logger.watch(model, log="all", log_freq=100,log_graph=False)
    wandb_logger.log_hyperparams(args)


    wandb_run_id = str(wandb_logger.version)
    if wandb_run_id == "None":
        print("no wandb run id this is a copy of model with DDP")
        #get pid of process
        pid = os.getpid()
        print("pid", pid)

        #get parent pid
        parent_pid = os.getppid()
        print("parent_pid", parent_pid)

        #get args
        print("args", args)



        # #get from os env variable
        # wandb_run_id = os.environ["MY_WANDB_RUN_ID"]
        # print("got wandb run id from env variable", wandb_run_id)

    else:
        #set OS env variable for wandb run id so other subprocesses can access it
        #make a txt file in tmp dir to save wandb run id
        print("setting env variable for wandb run id")
        #add argument to sys.argv
        sys.argv.append("--wandb_run_id")
        sys.argv.append(wandb_run_id)

        

    print("init trainer")

    # save hyperparameters
    wandb_logger.log_hyperparams(model.hparams)
    
   



    # save checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath='%s/%s/%s' % (args.save_path,args.model_name,wandb_run_id),
        filename='%s-{epoch:02d}-{train_loss:.2f}' % args.model_name,
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=args.save_checkpoint_every_n_epochs,
    )
   


    trainer = Trainer(
        max_epochs=args.num_epochs, 
        logger=wandb_logger,
        accelerator="gpu",
        strategy="ddp",
        num_nodes=args.num_nodes,
        devices=args.devices,
        callbacks=[checkpoint_callback],

    )
    if trainer.global_rank == 0:
        #make cp save dir savepath _ model name _ wandb run id
        if not os.path.exists('%s/%s/%s' % (args.save_path,args.model_name,wandb_run_id)):
            os.makedirs('%s/%s/%s' % (args.save_path,args.model_name,wandb_run_id), exist_ok=True)
            print("made dir %s/%s/%s" % (args.save_path,args.model_name,wandb_run_id))




    print(trainer.global_rank, trainer.world_size, os.environ["SLURM_NTASKS"])


    trainer.fit(model, datamodule=data_module)
    print("saving last checkpoint")
    trainer.save_checkpoint('%s/%s/%s/%s_final.ckpt' % (args.save_path,args.model_name,wandb_run_id,args.model_name))

    #just in case
    if trainer.global_rank == 0:
        print("saving model 2")

        # #check if dir exists
        # if not os.path.exists('%s/%s/' % (args.save_path,wandb_run_id)):
        #     os.makedirs('%s/%s/' % (args.save_path,wandb_run_id), exist_ok=True)
        
        # save_network(model, '%s/%s/%s.pth' % (args.save_path,wandb_run_id,args.model_name))


        # trainer.save_checkpoint('%s/%s/%s_ptl.ckpt' % (args.save_path,wandb_run_id,args.model_name))




    # loss_name = ['BCE']
    # iteration = 0
    # running_loss = {loss: 0 for loss in loss_name}
    # for epoch in range(args.num_epochs):
    #     timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    #     logger.info(timeStr+'Epoch {}/{}'.format(epoch, args.num_epochs - 1))
    #     logger.info(timeStr+'-' * 10)

    #     model.train(True)  # Set model to training mode
    #     # Iterate over data (including images and labels).
    #     for index, (images, labels) in enumerate(train_loader):
    #         iteration += 1
    #         # wrap them in Variable
    #         images = Variable(images.cuda().detach())
    #         labels = Variable(labels.cuda().detach())

    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         # Calculate loss
    #         loss = criterion(outputs, labels)

    #         # update the parameters
    #         loss.backward()
    #         optimizer.step()

    #         running_loss['BCE'] += loss.item()
    #         # break
    #         if iteration % 100 == 0:
    #             timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    #             logger.info(timeStr+'Epoch: {:g}, Itera: {:g}, Global Itera: {:g}, Step: {:g}, BCE: {:g} '.
    #                   format(epoch, index, iteration, len(train_loader), *[running_loss[name] / 100 for name in loss_name]))
    #             running_loss = {loss: 0 for loss in loss_name}
            
    #         if iteration % args.adjust_lr_iteration == 0:
    #             scheduler.step()
    #             logger.info('now lr is : {}\n'.format(scheduler.get_last_lr()))
                
    #     if epoch % 10 == 0:
    #         timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    #         logger.info(timeStr + '  Save  Model  ')
    #         save_network(model, '%s/models/%s_%d.pth' % (args.save_path, args.model_name, epoch))
    #         logger.info(timeStr + ' iteration=' + str(iteration) + ', lr=' + str(scheduler.get_last_lr()))
            
    # save_network(model, '%s/models/%s.pth' % (args.save_path, args.model_name))


if __name__ == "__main__":
    args = parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    if not os.path.exists('%s/%s/' % (args.save_path,args.model_name)):
        os.makedirs('%s/%s/' % (args.save_path,args.model_name), exist_ok=True)
    if not os.path.exists('%s/report' % args.save_path):
        os.makedirs('%s/report' % args.save_path, exist_ok=True)

    main()
