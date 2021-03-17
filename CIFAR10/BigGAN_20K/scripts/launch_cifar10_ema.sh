#!/bin/bash

ROOT_PATH="./CIFAR10/BigGAN"


EPOCHS=2000
BATCHSIZE=512


### subset of CIFAR10 w/o DiffAugment
CUDA_VISIBLE_DEVICES=1,0 python3 train.py \
--root_path $ROOT_PATH --seed 2020 \
--shuffle --batch_size $BATCHSIZE --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs $EPOCHS \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--data_root data/ --dataset C10_2020_hdf5 --load_in_mem --augment \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 1000 --no_fid --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
2>&1 | tee output_biggan_cifar10_2020.txt
# --resume

# ### subset of CIFAR10 w/ DiffAugment
# CUDA_VISIBLE_DEVICES=1,0 python3 train.py \
# --root_path $ROOT_PATH --seed 2020 \
# --shuffle --batch_size $BATCHSIZE --parallel \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs $EPOCHS \
# --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
# --data_root data/ --dataset C10_2020_hdf5 --load_in_mem --augment \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema --use_ema --ema_start 1000 \
# --test_every 1000 --no_fid --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --DiffAugment_policy 'color,translation,cutout' \
# 2>&1 | tee output_biggan_cifar10_2020.txt
# # --resume


# ### complete CIFAR10 dataset
# CUDA_VISIBLE_DEVICES=1,0 python3 train.py \
# --root_path $ROOT_PATH --seed 2020 \
# --shuffle --batch_size $BATCHSIZE --parallel \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs $EPOCHS \
# --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
# --data_root data/ --dataset C10 --augment \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema --use_ema --ema_start 1000 \
# --test_every 1000 --no_fid --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# 2>&1 | tee output_biggan_cifar10_full_2020.txt
#  #--resume

