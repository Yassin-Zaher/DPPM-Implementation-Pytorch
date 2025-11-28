#!/bin/bash

sampling_args="--arch UNet --class-cond --sampling-steps 250 --sampling-only --save-dir ./sampled_images/"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8208 main.py \
    --arch UNet --dataset flowers --batch-size 128 --num-sampled-images 50000 $sampling_args \
    --pretrained-ckpt ./trained_models/UNet_flowers-epoch_1000-timesteps_1000-class_condn_True_ema_0.9995.pt


