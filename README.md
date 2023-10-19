# Beyond Detection Visual Realism of Deepfakes


## Prepare env
```conda activate pytorch_env```

## Training
Allocate a node with 4 gpus for training(optional)

``` bash
salloc --nodes=1 --gres=gpu:4 --ntasks-per-node=4 --mem=0 --time=0-10:00:00 --cpus-per-task=12 --job-name=Interactive_GPU2 --partition=gpu
```

Initialize a sweep with this will return a sweep id
``` bash
wandb sweep train_conv.yaml
```

Run the sweep agent with slurm so that it can run in the background
``` bash
sbatch ./run_agent.sh luka_borut/BeyondDetection/tw32ir3q
```

this results in 
``` bash
wandb agent luka_borut/BeyondDetection/1sag5gvm
wandb: Starting wandb agent 🕵️
2023-10-17 13:43:58,336 - wandb.wandb_agent - INFO - Running runs: []
2023-10-17 13:43:58,695 - wandb.wandb_agent - INFO - Agent received command: run
2023-10-17 13:43:58,695 - wandb.wandb_agent - INFO - Agent starting run with config:
        accumulate_grad_batches: 1
        augment_prob: 0.5
        augmentation: True
        batch_size: 1
        drop_path_rate: 0.1
        dropout: 0.5
        loss: mae
        max_epochs: 32
        og_checkpoint: ./DFGC-1st-2022-model/convnext_xlarge_384_in22ft1k_30.pth
        seed: -1
        seq_len: 8
        val_split: 0.1
        weight_decay: 0.01
2023-10-17 13:43:58,698 - wandb.wandb_agent - INFO - About to run command: /usr/bin/env srun --nodes=1 --cpus-per-task=12 --gpus=4 "-p gpu" --ntasks-per-node=4 --exclusive python3 train_convnext_images.py --accumulate_grad_batches=1 --augment_prob=0.5 --augmentation=True --batch_size=1 --drop_path_rate=0.1 --dropout=0.5 --loss=mae --max_epochs=32 --og_checkpoint=./DFGC-1st-2022-model/convnext_xlarge_384_in22ft1k_30.pth --seed=-1 --seq_len=8 --val_split=0.1 --weight_decay=0.01 --devices 0 1 2 3
srun: error: WARNING: Multiple leaf switches contain nodes: gn[01-60]
2023-10-17 13:44:03,707 - wandb.wandb_agent - INFO - Running runs: ['odogtmgc']

```

This 
