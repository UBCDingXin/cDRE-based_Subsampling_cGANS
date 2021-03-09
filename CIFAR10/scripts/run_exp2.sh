############################################
# EXP 2: NTRAIN=20K, BigGAN
ROOT_PATH="./CIFAR10"

NCPU=8
NTRAIN=20000
EXP_NAME="main_ntrain_${NTRAIN}"
GAN_EPOCHS=2000
NFAKE_PER_CLASS=5000
PreCNN="ResNet34"


echo "-------------------------------------------------------------------------------------------------"
echo "No Subsampling"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--exp_name $EXP_NAME --root_path $ROOT_PATH --num_workers $NCPU \
--ntrain $NTRAIN \
--gan_epochs $GAN_EPOCHS \
--dre_precnn_net $PreCNN \
--dre_precnn_epochs 350 --dre_precnn_resume_epoch 0 \
--dre_precnn_lr_base 0.1 --dre_precnn_lr_decay_factor 0.1 --dre_precnn_lr_decay_epochs 150_250 \
--dre_precnn_batch_size_train 128 --dre_precnn_weight_decay 1e-4 --dre_precnn_transform \
--dre_mode conditional --dre_net MLP5 \
--dre_epochs 350 --dre_resume_epoch 0 \
--dre_lr_base 1e-4 --dre_lr_decay_factor 0.1 --dre_lr_decay_epochs 150_250 \
--dre_batch_size 128 --dre_lambda 1e-3 \
--visualize_fake_images \
--eval --eval_FID_batch_size 200 \
--samp_batch_size 1000 --samp_burnin_size 10000 --samp_nfake_per_class $NFAKE_PER_CLASS \




echo "-------------------------------------------------------------------------------------------------"
echo "Subsampling by Unconditional DRE-F-SP+RS"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--exp_name $EXP_NAME --root_path $ROOT_PATH --num_workers $NCPU \
--ntrain $NTRAIN \
--gan_epochs $GAN_EPOCHS \
--dre_precnn_net $PreCNN \
--dre_precnn_epochs 350 --dre_precnn_resume_epoch 0 \
--dre_precnn_lr_base 0.1 --dre_precnn_lr_decay_factor 0.1 --dre_precnn_lr_decay_epochs 150_250 \
--dre_precnn_batch_size_train 128 --dre_precnn_weight_decay 1e-4 --dre_precnn_transform \
--dre_mode unconditional --dre_net MLP5 \
--dre_epochs 350 --dre_resume_epoch 0 \
--dre_lr_base 1e-4 --dre_lr_decay_factor 0.1 --dre_lr_decay_epochs 150_250 \
--dre_batch_size 128 --dre_lambda 1e-3 \
--subsampling \
--visualize_fake_images \
--eval --eval_FID_batch_size 200 \
--samp_batch_size 1000 --samp_burnin_size 10000 --samp_nfake_per_class $NFAKE_PER_CLASS \



echo "-------------------------------------------------------------------------------------------------"
echo "Subsampling by Conditional DRE-F-SP+RS"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--exp_name $EXP_NAME --root_path $ROOT_PATH --num_workers $NCPU \
--ntrain $NTRAIN \
--gan_epochs $GAN_EPOCHS \
--dre_precnn_net $PreCNN \
--dre_precnn_epochs 350 --dre_precnn_resume_epoch 0 \
--dre_precnn_lr_base 0.1 --dre_precnn_lr_decay_factor 0.1 --dre_precnn_lr_decay_epochs 150_250 \
--dre_precnn_batch_size_train 128 --dre_precnn_weight_decay 1e-4 --dre_precnn_transform \
--dre_mode conditional --dre_net MLP5 \
--dre_epochs 350 --dre_resume_epoch 0 \
--dre_lr_base 1e-4 --dre_lr_decay_factor 0.1 --dre_lr_decay_epochs 150_250 \
--dre_batch_size 128 --dre_lambda 1e-3 \
--subsampling \
--visualize_fake_images \
--eval --eval_FID_batch_size 200 \
--samp_batch_size 1000 --samp_burnin_size 10000 --samp_nfake_per_class $NFAKE_PER_CLASS \
