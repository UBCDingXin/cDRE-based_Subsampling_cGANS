import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--num_workers', type=int, default=0)


    ''' Dataset '''
    ## Data split: RC-49 is split into a train set (the last decimal of the degree is odd) and a test set (the last decimal of the degree is even); the unique labels in two sets do not overlap.
    parser.add_argument('--data_split', type=str, default='train',
                        choices=['all', 'train'])
    parser.add_argument('--min_label', type=float, default=0.0)
    parser.add_argument('--max_label', type=float, default=90.0)
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=64, metavar='N', choices=[64])
    parser.add_argument('--max_num_img_per_label', type=int, default=25, metavar='N')
    parser.add_argument('--max_num_img_per_label_after_replica', type=int, default=0, metavar='N')
    parser.add_argument('--show_real_imgs', action='store_true', default=False)
    parser.add_argument('--visualize_fake_images', action='store_true', default=False)


    ''' GAN settings '''
    # label embedding setting
    parser.add_argument('--net_embed', type=str, default='ResNet34_embed') #ResNetXX_emebed
    parser.add_argument('--epoch_cnn_embed', type=int, default=200) #epoch of cnn training for label embedding
    parser.add_argument('--resumeepoch_cnn_embed', type=int, default=0) #epoch of cnn training for label embedding
    parser.add_argument('--epoch_net_y2h', type=int, default=500)
    parser.add_argument('--dim_embed', type=int, default=128) #dimension of the embedding space
    parser.add_argument('--batch_size_embed', type=int, default=256, metavar='N')

    parser.add_argument('--loss_type_gan', type=str, default='vanilla')
    parser.add_argument('--niters_gan', type=int, default=30000, help='number of iterations')
    parser.add_argument('--resume_niters_gan', type=int, default=0)
    parser.add_argument('--save_niters_freq', type=int, default=5000, help='frequency of saving checkpoints')
    parser.add_argument('--lr_g_gan', type=float, default=1e-4, help='learning rate for generator')
    parser.add_argument('--lr_d_gan', type=float, default=1e-4, help='learning rate for discriminator')
    parser.add_argument('--dim_gan', type=int, default=256, help='Latent dimension of GAN')
    parser.add_argument('--batch_size_disc', type=int, default=256)
    parser.add_argument('--batch_size_gene', type=int, default=256)

    parser.add_argument('--kernel_sigma', type=float, default=-1.0,
                        help='If kernel_sigma<0, then use rule-of-thumb formula to compute the sigma.')
    parser.add_argument('--threshold_type', type=str, default='hard', choices=['soft', 'hard'])
    parser.add_argument('--kappa', type=float, default=-2)
    parser.add_argument('--nonzero_soft_weight_threshold', type=float, default=1e-3,
                        help='threshold for determining nonzero weights for SVDL; we neglect images with too small weights')


    ''' DRE Settings '''
    ## Pre-trained AE for feature extraction
    parser.add_argument('--dre_presae_ch', type=int, default=64)
    parser.add_argument('--dre_presae_epochs', type=int, default=200)
    parser.add_argument('--dre_presae_resume_epoch', type=int, default=0, metavar='N')
    parser.add_argument('--dre_presae_lr_base', type=float, default=0.01, help='base learning rate of CNNs')
    parser.add_argument('--dre_presae_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--dre_presae_lr_decay_freq', type=int, default=50)
    parser.add_argument('--dre_presae_batch_size_train', type=int, default=128, metavar='N')
    parser.add_argument('--dre_presae_weight_decay', type=float, default=1e-4)
    parser.add_argument('--dre_presae_lambda_sparsity', type=float, default=1e-3, help='Control the sparsity of the sparse AE.')
    ## DR model in the feature space
    parser.add_argument('--dre_mode', type=str, default='conditional',
                        choices=['conditional','unconditional'])
    parser.add_argument('--dre_net', type=str, default='MLP5',
                        choices=['MLP3', 'MLP5', 'MLP7', 'MLP9'], help='DR Model in the feature space') # DRE in Feature Space
    parser.add_argument('--dre_epochs', type=int, default=350)
    parser.add_argument('--dre_lr_base', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dre_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--dre_lr_decay_epochs', type=str, default='100_200', help='decay lr at which epoch; separate by _')
    parser.add_argument('--dre_batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training DRE')
    parser.add_argument('--dre_lambda', type=float, default=1e-3, help='penalty in DRE')
    parser.add_argument('--dre_resume_epoch', type=int, default=0)

    parser.add_argument('--dre_no_vicinal', action='store_true', default=False)
    parser.add_argument('--dre_threshold_type', type=str, default='hard', choices=['soft', 'hard'])
    parser.add_argument('--dre_kappa', type=float, default=-1)
    parser.add_argument('--dre_nonzero_soft_weight_threshold', type=float, default=1e-3,
                        help='threshold for determining nonzero weights for SVDL; we neglect images with too small weights')


    ''' Filtering '''
    # parser.add_argument('--do_filter', action='store_true', default=False)
    parser.add_argument('--filter_precnn_net', type=str, default='VGG11',
                        help='Use a pre-trained CNN to filter out fake images with large difference between assigned labels and predicted labels.')
    parser.add_argument('--filter_mae_percentile_threshold', type=float, default=1.0)
    parser.add_argument('--filter_precnn_epochs', type=int, default=200)
    parser.add_argument('--filter_precnn_resume_epoch', type=int, default=0, metavar='N')
    parser.add_argument('--filter_precnn_lr_base', type=float, default=0.01, help='base learning rate of CNNs')
    parser.add_argument('--filter_precnn_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--filter_precnn_lr_decay_epochs', type=str, default='50_120')
    parser.add_argument('--filter_precnn_batch_size_train', type=int, default=128, metavar='N')
    parser.add_argument('--filter_precnn_weight_decay', type=float, default=1e-4)
    parser.add_argument('--filter_nburnin_per_label', type=int, default=500)



    ''' Evaluation Settings '''
    '''
    Four evaluation modes:
    Mode 1: eval on unique labels used for GAN training;
    Mode 2. eval on all unique labels in the dataset and when computing FID use all real images in the dataset;
    Mode 3. eval on all unique labels in the dataset and when computing FID only use real images for GAN training in the dataset (to test SFID's effectiveness on unseen labels);
    Mode 4. eval on a interval [min_label, max_label] with num_eval_labels labels.
    '''
    parser.add_argument('--subsampling', action='store_true', default=False)
    parser.add_argument('--subsampling_baseline', action='store_true', default=False,
                        help='The baseline unconditional subsampling method in "Ding, Xin, Z. Jane Wang, and William J. Welch. "Subsampling Generative Adversarial Networks: Density Ratio Estimation in Feature Space With Softplus Loss." IEEE Transactions on Signal Processing 68 (2020): 1910-1922."')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--eval_mode', type=int, default=2)
    parser.add_argument('--num_eval_labels', type=int, default=-1)
    parser.add_argument('--samp_dump_fake_data', action='store_true', default=False) #if we need to dump fake images during generation
    parser.add_argument('--samp_batch_size', type=int, default=1000)
    parser.add_argument('--samp_burnin_size', type=int, default=5000)
    parser.add_argument('--nfake_per_label', type=int, default=200)
    parser.add_argument('--nreal_per_label', type=int, default=-1)
    parser.add_argument('--epoch_FID_CNN', type=int, default=200)
    parser.add_argument('--FID_radius', type=float, default=0)
    parser.add_argument('--FID_num_centers', type=int, default=-1)
    parser.add_argument('--dump_fake_for_NIQE', action='store_true', default=False)

    args = parser.parse_args()

    return args
