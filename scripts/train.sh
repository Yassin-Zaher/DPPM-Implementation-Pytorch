#!/bin/bash

# IMPORTANT : RED LBAL, LHADIK 1000 epoch,
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8109 main.py \
    --arch UNet --dataset flowers --class-cond --epochs 100 --batch-size 128 --sampling-steps 50 \
    --data-dir ~/datasets/oxford_102_flowers/
