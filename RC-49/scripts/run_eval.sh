

############################################
# EXP 1: NImgPerLabel=25

ROOT_PATH="/home/xin/OneDrive/Working_directory/Subsample_cGANs_via_cDRE/RC-49"
DATA_PATH="/home/xin/OneDrive/Working_directory/Subsample_cGANs_via_cDRE/RC-49/data/RC-49_64x64.h5"

NCPU=8
MAX_N_IMG_PER_LABEL=25
EXP_NAME="main_NImgPerLabel_${MAX_N_IMG_PER_LABEL}"
GAN_NITERS=30000
AE_LAMBDA=1e-3
DRE_LAMBDA=1e-3
DRE_kappa=-1
filter_threshold=0.9


CcGAN_type="hard"

echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN: ${CcGAN_type}; Subsampling: unconditional; Filtering: Yes"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--exp_name $EXP_NAME --root_path $ROOT_PATH --data_path $DATA_PATH --num_workers $NCPU \
--max_num_img_per_label $MAX_N_IMG_PER_LABEL \
--kernel_sigma -1 --threshold_type $CcGAN_type --kappa -2 \
--dre_presae_epochs 200 --dre_presae_resume_epoch 0 \
--dre_presae_lr_base 0.01 --dre_presae_lr_decay_factor 0.1 --dre_presae_lr_decay_freq 50 \
--dre_presae_batch_size_train 128 --dre_presae_weight_decay 1e-4 --dre_presae_lambda_sparsity $AE_LAMBDA \
--dre_mode 'unconditional' \
--dre_epochs 350 --dre_resume_epoch 0 --dre_lr_base 1e-4 --dre_lr_decay_factor 0.1 --dre_lr_decay_epochs 100_200 \
--dre_batch_size 256 --dre_lambda $DRE_LAMBDA \
--subsampling \
--filter_precnn_net VGG11 --filter_mae_percentile_threshold $filter_threshold \
--filter_precnn_epochs 200 --filter_nburnin_per_label 500 \
--visualize_fake_images \
--eval \
--dump_fake_for_NIQE \

