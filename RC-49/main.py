print("\n===================================================================================================")

import argparse
import copy
import gc
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
import h5py
import os
import random
from tqdm import tqdm, trange
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import timeit
from PIL import Image
import pickle

### import my stuffs ###
from opts import parse_opts
from utils import *
from models import *
from train_ccgan import train_ccgan, SampCcGAN_given_labels
from train_net_for_label_embed import train_net_embed, train_net_y2h
from train_sparseAE import train_sparseAE
from train_dre import train_dre
from train_cdre import train_cdre
from eval_metrics import cal_FID, cal_labelscore
from train_filter_cnn import train_filter_cnn, test_filter_cnn


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = parse_opts()
print(args)

assert int((args.exp_name).split('_')[2]) == args.max_num_img_per_label

if args.subsampling:
    if args.dre_mode=="conditional":
        subsampling_method = "{}_{}_kappa{}_filter{}".format(args.dre_mode, args.dre_threshold_type, args.dre_kappa, args.filter_mae_percentile_threshold)
    else:
        if args.subsampling_baseline:
            subsampling_method = "baseline_unconditional"
        else:
            subsampling_method = "unconditional_filter{}".format(args.filter_mae_percentile_threshold)
else:
    subsampling_method = "None_filter{}".format(args.filter_mae_percentile_threshold)

#--------------------------------
# system
NGPU = torch.cuda.device_count()

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

#-------------------------------
# Embedding
base_lr_x2y = 0.01
base_lr_y2h = 0.01

#-------------------------------
# eval and sampling parameters
assert args.eval_mode in [1,2,3,4] #evaluation mode must be in 1,2,3,4
if args.data_split == "all":
    args.eval_mode != 1

#-------------------------------
# output folders
eval_models_folder = os.path.join(args.root_path, 'output/eval_models')
assert os.path.exists(eval_models_folder)

output_directory = os.path.join(args.root_path, 'output/{}'.format(args.exp_name))
os.makedirs(output_directory, exist_ok=True)

save_models_folder = os.path.join(output_directory, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)

save_images_folder = os.path.join(output_directory, 'saved_images')
os.makedirs(save_images_folder, exist_ok=True)

save_traincurves_folder = os.path.join(output_directory, 'training_curves')
os.makedirs(save_traincurves_folder, exist_ok=True)

save_evalresults_folder = os.path.join(output_directory, 'eval_results')
os.makedirs(save_evalresults_folder, exist_ok=True)

# if args.samp_dump_fake_data: #if we need to dump fake images during generation
#     dump_fake_images_folder = os.path.join(output_directory, 'cache_dump_fake/CcGAN_niters_{}_subsampling_{}_EvalMode_{}_NFakePerLabel_{}_seed_{}'.format(args.niters_gan, subsampling_method, args.eval_mode, args.nfake_per_label, args.seed))



#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
# data loader
hf = h5py.File(args.data_path, 'r')
labels_all = hf['labels'][:]
labels_all = labels_all.astype(np.float)
images_all = hf['images'][:]
indx_train = hf['indx_train'][:]
hf.close()
print("\n RC-49 dataset shape: {}x{}x{}x{}".format(images_all.shape[0], images_all.shape[1], images_all.shape[2], images_all.shape[3]))

# data split
if args.data_split == "train":
    images_train = images_all[indx_train]
    labels_train_raw = labels_all[indx_train]
    images_test = np.delete(images_all, indx_train, axis=0)
    labels_test_raw = np.delete(labels_all, indx_train, axis=0)
else:
    images_train = copy.deepcopy(images_all)
    labels_train_raw = copy.deepcopy(labels_all)

# only take images with label in (q1, q2)
q1 = args.min_label
q2 = args.max_label
indx = np.where((labels_train_raw>q1)*(labels_train_raw<q2)==True)[0]
labels_train_raw = labels_train_raw[indx]
images_train = images_train[indx]
assert len(labels_train_raw)==len(images_train)

indx = np.where((labels_test_raw>q1)*(labels_test_raw<q2)==True)[0]
labels_test_raw = labels_test_raw[indx]
images_test = images_test[indx]
assert len(labels_test_raw)==len(images_test)

if args.visualize_fake_images or args.eval:
    indx = np.where((labels_all>q1)*(labels_all<q2)==True)[0]
    labels_all = labels_all[indx]
    images_all = images_all[indx]
    assert len(labels_all)==len(images_all)

# for each angle, take no more than args.max_num_img_per_label images
image_num_threshold = args.max_num_img_per_label
print("\n Original set has {} images; For each angle, take no more than {} images>>>".format(len(images_train), image_num_threshold))
unique_labels_tmp = np.sort(np.array(list(set(labels_train_raw))))
for i in tqdm(range(len(unique_labels_tmp))):
    indx_i = np.where(labels_train_raw == unique_labels_tmp[i])[0]
    if len(indx_i)>image_num_threshold:
        np.random.shuffle(indx_i)
        indx_i = indx_i[0:image_num_threshold]
    if i == 0:
        sel_indx = indx_i
    else:
        sel_indx = np.concatenate((sel_indx, indx_i))
images_train = images_train[sel_indx]
labels_train_raw = labels_train_raw[sel_indx]
print("{} images left and there are {} unique labels".format(len(images_train), len(set(labels_train_raw))))

# normalize labels_train_raw
print("\n Range of unnormalized labels: ({},{})".format(np.min(labels_train_raw), np.max(labels_train_raw)))

labels_train = labels_train_raw / args.max_label

print("\n Range of normalized labels: ({},{})".format(np.min(labels_train), np.max(labels_train)))

unique_labels_norm = np.sort(np.array(list(set(labels_train))))

if args.kernel_sigma<0:
    std_label = np.std(labels_train)
    args.kernel_sigma = 1.06*std_label*(len(labels_train))**(-1/5)

    print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
    print("\n The std of {} labels is {} so the kernel sigma is {}".format(len(labels_train), std_label, args.kernel_sigma))

if args.kappa<0:
    n_unique = len(unique_labels_norm)

    diff_list = []
    for i in range(1,n_unique):
        diff_list.append(unique_labels_norm[i] - unique_labels_norm[i-1])
    kappa_base = np.abs(args.kappa)*np.max(np.array(diff_list))

    if args.threshold_type=="hard":
        args.kappa = kappa_base
    else:
        args.kappa = 1/kappa_base**2


#######################################################################################
'''               Pre-trained CNN and GAN for label embedding                       '''
#######################################################################################
net_embed_filename_ckpt = save_models_folder + '/ckpt_{}_epoch_{}_seed_{}.pth'.format(args.net_embed, args.epoch_cnn_embed, args.seed)
net_y2h_filename_ckpt = save_models_folder + '/ckpt_net_y2h_epoch_{}_seed_{}.pth'.format(args.epoch_net_y2h, args.seed)

trainset = IMGs_dataset(images_train, labels_train, normalize=True)
trainloader_embed_net = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_embed, shuffle=True, num_workers=args.num_workers)

if args.net_embed == "ResNet18_embed":
    net_embed = ResNet18_embed(dim_embed=args.dim_embed, ngpu = NGPU)
elif args.net_embed == "ResNet34_embed":
    net_embed = ResNet34_embed(dim_embed=args.dim_embed, ngpu = NGPU)
elif args.net_embed == "ResNet50_embed":
    net_embed = ResNet50_embed(dim_embed=args.dim_embed, ngpu = NGPU)
net_embed = net_embed.cuda()

net_y2h = model_y2h(dim_embed=args.dim_embed)
net_y2h = net_y2h.cuda()

## (1). Train net_embed first: x2h+h2y
if not os.path.isfile(net_embed_filename_ckpt):
    print("\n Start training CNN for label embedding >>>")
    optimizer_net_embed = torch.optim.SGD(net_embed.parameters(), lr = base_lr_x2y, momentum= 0.9, weight_decay=1e-4)
    net_embed = train_net_embed(trainloader_embed_net, None, net_embed, optimizer_net_embed, epochs=args.epoch_cnn_embed, base_lr=base_lr_x2y, save_models_folder = save_models_folder, resumeepoch = args.resumeepoch_cnn_embed)
    # save model
    torch.save({
    'net_state_dict': net_embed.state_dict(),
    }, net_embed_filename_ckpt)
else:
    print("\n net_embed ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(net_embed_filename_ckpt)
    net_embed.load_state_dict(checkpoint['net_state_dict'])
#end not os.path.isfile

## (2). Train y2h
#train a net which maps a label back to the embedding space
if not os.path.isfile(net_y2h_filename_ckpt):
    print("\n Start training net_y2h >>>")
    optimizer_net_y2h = torch.optim.SGD(net_y2h.parameters(), lr = base_lr_y2h, momentum = 0.9, weight_decay=1e-4)
    net_y2h = train_net_y2h(unique_labels_norm, net_y2h, net_embed, optimizer_net_y2h, epochs=args.epoch_net_y2h, base_lr=base_lr_y2h, batch_size=128)
    # save model
    torch.save({
    'net_state_dict': net_y2h.state_dict(),
    }, net_y2h_filename_ckpt)
else:
    print("\n net_y2h ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(net_y2h_filename_ckpt)
    net_y2h.load_state_dict(checkpoint['net_state_dict'])
#end not os.path.isfile

##some simple test
indx_tmp = np.arange(len(unique_labels_norm))
np.random.shuffle(indx_tmp)
indx_tmp = indx_tmp[:10]
labels_tmp = unique_labels_norm[indx_tmp].reshape(-1,1)
labels_tmp = torch.from_numpy(labels_tmp).type(torch.float).cuda()
epsilons_tmp = np.random.normal(0, 0.2, len(labels_tmp))
epsilons_tmp = torch.from_numpy(epsilons_tmp).view(-1,1).type(torch.float).cuda()
labels_tmp = torch.clamp(labels_tmp+epsilons_tmp, 0.0, 1.0)
net_embed.eval()
net_h2y = net_embed.h2y
net_y2h.eval()
with torch.no_grad():
    labels_rec_tmp = net_h2y(net_y2h(labels_tmp)).cpu().numpy().reshape(-1,1)
results = np.concatenate((labels_tmp.cpu().numpy(), labels_rec_tmp), axis=1)
print("\n labels vs reconstructed labels")
print(results)



#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################
print("{}, Sigma is {}, Kappa is {}".format(args.threshold_type, args.kernel_sigma, args.kappa))

start = timeit.default_timer()
print("\n Begin Training CcGAN with {}:".format(subsampling_method))
filename_gan_ckpt = save_models_folder + '/ckpt_CcGAN_niters_{}_seed_{}_{}_{}_{}.pth'.format(args.niters_gan, args.seed, args.threshold_type, args.kernel_sigma, args.kappa)
print(filename_gan_ckpt)


if not os.path.isfile(filename_gan_ckpt):
    save_GANimages_InTrain_folder = save_images_folder + '/CcGAN_{}_{}_{}_InTrain'.format(args.threshold_type, args.kernel_sigma, args.kappa)
    os.makedirs(save_GANimages_InTrain_folder, exist_ok=True)

    netG = CcGAN_Generator(nz=args.dim_gan, dim_embed=args.dim_embed).cuda()
    netD = CcGAN_Discriminator(dim_embed=args.dim_embed).cuda()
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

    # Start training
    netG, netD = train_ccgan(args.kernel_sigma, args.kappa, images_train, labels_train, netG, netD, net_y2h, save_images_folder=save_GANimages_InTrain_folder, save_models_folder=save_models_folder)

    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, filename_gan_ckpt)

else:
    print("Loading pre-trained generator >>>")
    checkpoint = torch.load(filename_gan_ckpt)
    netG = CcGAN_Generator(nz=args.dim_gan, dim_embed=args.dim_embed).cuda()
    netG = nn.DataParallel(netG)
    netG.load_state_dict(checkpoint['netG_state_dict'])

def fn_sampleGAN_given_labels(labels, batch_size, to_numpy=True, verbose=True, PreNetFilter=None, filter_mae_cutoff_point=1e30, num_workers=args.num_workers):
    fake_images, fake_labels = SampCcGAN_given_labels(netG, net_y2h, labels, batch_size = batch_size, to_numpy=to_numpy, verbose=verbose, PreNetFilter=PreNetFilter, filter_mae_cutoff_point=filter_mae_cutoff_point, num_workers=num_workers)
    return fake_images, fake_labels

stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))



#######################################################################################
'''    train PreCNNFilter to filter out synthetic data before subsampling           '''
#######################################################################################
if args.filter_mae_percentile_threshold<1.0:

    prenet_filter_filename_ckpt = save_models_folder + '/ckpt_PreCNNFilter_{}_epoch_{}_seed_{}.pth'.format(args.filter_precnn_net, args.filter_precnn_epochs, args.seed)

    if args.filter_precnn_net[0:3] == "VGG":
        PreNetFilter = VGG(args.filter_precnn_net)
        PreNetFilter = nn.DataParallel(PreNetFilter)
    else:
        raise Exception('Not supproted filtered cnn...')

    # dataloader
    trainset = IMGs_dataset(images_train, labels_train_raw/args.max_label, normalize=True)
    trainloader_filter_cnn = torch.utils.data.DataLoader(trainset, batch_size=args.filter_precnn_batch_size_train, shuffle=True, num_workers=args.num_workers)

    if args.data_split == "train":
        testset = IMGs_dataset(images_test, labels_test_raw/args.max_label, normalize=True)
        testloader_precnn = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=args.num_workers)
    else:
        testloader_precnn = None

    filter_cnn_lr_decay_epochs = (args.filter_precnn_lr_decay_epochs).split("_")
    filter_cnn_lr_decay_epochs = [int(epoch) for epoch in filter_cnn_lr_decay_epochs]

    if not os.path.isfile(prenet_filter_filename_ckpt):
        print("\n Start training PreCNNFilter to filter out synthetic images before subsampling >>>")
        PreNetFilter = train_filter_cnn(net=PreNetFilter, net_name=args.filter_precnn_net, trainloader=trainloader_filter_cnn, testloader=testloader_precnn, epochs=args.filter_precnn_epochs, resume_epoch=args.filter_precnn_resume_epoch, save_freq=filter_cnn_lr_decay_epochs, lr_base=args.filter_precnn_lr_base, lr_decay_factor=args.filter_precnn_lr_decay_factor, lr_decay_epochs=filter_cnn_lr_decay_epochs, weight_decay=args.filter_precnn_weight_decay, seed = args.seed, path_to_ckpt = None)
        PreNetFilter = PreNetFilter.cpu() ##put on cpu
        # save model
        torch.save({
        'net_state_dict': PreNetFilter.state_dict(),
        }, prenet_filter_filename_ckpt)
    else:
        print("\n PreCNNFilter ckpt already exists")
        print("\n Loading...")
        checkpoint = torch.load(prenet_filter_filename_ckpt)
        PreNetFilter.load_state_dict(checkpoint['net_state_dict'])
    #end not os.path.isfile

    unique_labels_norm = np.sort(np.array(list(set(labels_test_raw))))/args.max_label
    filter_fake_labels_assigned = []
    for i in range(len(unique_labels_norm)):
        fake_labels_i = unique_labels_norm[i]*np.ones(args.filter_nburnin_per_label)
        filter_fake_labels_assigned.append(fake_labels_i)
    ##end for i
    filter_fake_labels_assigned = np.concatenate(filter_fake_labels_assigned, axis=0)
    filter_fake_images, _ = fn_sampleGAN_given_labels(labels=filter_fake_labels_assigned, batch_size=args.samp_batch_size, to_numpy=True, verbose=True)
    fake_dataset = IMGs_dataset(filter_fake_images, filter_fake_labels_assigned, normalize=False)
    fake_loader_precnn = torch.utils.data.DataLoader(fake_dataset, batch_size=200, shuffle=False, num_workers=args.num_workers)
    _, filter_fake_mae_loss = test_filter_cnn(net=PreNetFilter, testloader=fake_loader_precnn, verbose=True)
    ### the cut off of the mae, which is used to decide which are samples are left for the subsequent subsampling.
    filter_mae_cutoff_point = np.quantile(filter_fake_mae_loss, q=args.filter_mae_percentile_threshold)

    # if args.threshold_type=="hard":
    #     filter_mae_cutoff_point = args.kappa
    # else:
    #     filter_mae_cutoff_point = np.sqrt(1/args.kappa)

###end args.filter_mae_percentile_threshold



#######################################################################################
'''                                  DRE Training                                   '''
#######################################################################################

if args.subsampling:

    ##############################################
    ''' Pre-trained sparse AE for feature extraction '''
    print("\n -----------------------------------------------------------------------------------------")
    print("\n Pre-trained sparse AE for feature extraction")

    filename_presae_ckpt = save_models_folder + '/ckpt_PreSAEForDRE_epoch_{}_lambda_{}_seed_{}.pth'.format(args.dre_presae_epochs, args.dre_presae_lambda_sparsity, args.seed)
    print('\n' + filename_presae_ckpt)

    # training
    if not os.path.isfile(filename_presae_ckpt):

        save_sae_images_InTrain_folder = save_images_folder + '/SAE_lambda_{}_InTrain_{}'.format(args.dre_presae_lambda_sparsity, args.seed)
        os.makedirs(save_sae_images_InTrain_folder, exist_ok=True)

        # dataloader
        trainset = IMGs_dataset(images_train, labels_train, normalize=True)
        trainloader_sparseAE = torch.utils.data.DataLoader(trainset, batch_size=args.dre_presae_batch_size_train, shuffle=True, num_workers=args.num_workers)

        # initialize net
        dre_presae_encoder_net = encoder_extract(ch=64, dim_bottleneck=args.img_size*args.img_size*args.num_channels).cuda()
        dre_presae_decoder_net = decoder_extract(ch=64, dim_bottleneck=args.img_size*args.img_size*args.num_channels).cuda()
        dre_presae_encoder_net = nn.DataParallel(dre_presae_encoder_net)
        dre_presae_decoder_net = nn.DataParallel(dre_presae_decoder_net)

        print("\n Start training sparseAE model for feature extraction in the DRE >>>")
        dre_presae_encoder_net, dre_presae_decoder_net = train_sparseAE(trainloader=trainloader_sparseAE, net_encoder=dre_presae_encoder_net, net_decoder=dre_presae_decoder_net, save_sae_images_folder=save_sae_images_InTrain_folder, path_to_ckpt=save_models_folder)
        # store model
        torch.save({
            'encoder_net_state_dict': dre_presae_encoder_net.state_dict(),
            # 'decoder_net_state_dict': dre_presae_decoder_net.state_dict(),
        }, filename_presae_ckpt)
        print("\n End training CNN.")
    else:
        print("\n Loading pre-trained sparseAE for feature extraction in DRE.")
        dre_presae_encoder_net = encoder_extract(ch=64, dim_bottleneck=args.img_size*args.img_size*args.num_channels).cuda()
        dre_presae_encoder_net = nn.DataParallel(dre_presae_encoder_net)
        checkpoint = torch.load(filename_presae_ckpt)
        dre_presae_encoder_net.load_state_dict(checkpoint['encoder_net_state_dict'])
    #end if


    ##############################################
    ''' DRE Training '''
    print("\n -----------------------------------------------------------------------------------------")
    print("\n {} DRE training".format(args.dre_mode))

    if args.dre_mode == "conditional": ## Conditional DRE

        if args.dre_kappa<0:
            n_unique = len(unique_labels_norm)

            diff_list = []
            for i in range(1,n_unique):
                diff_list.append(unique_labels_norm[i] - unique_labels_norm[i-1])
            kappa_base = np.abs(args.dre_kappa)*np.max(np.array(diff_list))

            if args.dre_threshold_type=="hard":
                args.dre_kappa = kappa_base
            else:
                args.dre_kappa = 1/kappa_base**2
        #end if

        ## dre filename
        drefile_fullpath = save_models_folder + '/ckpt_{}_DRE-F-SP_{}_epochs_{}_lambda_{}_type_{}_kappa_{}_Filter_{}_CcGAN_{}_{}_{}_{}_seed_{}.pth'.format(args.dre_mode, args.dre_net, args.dre_epochs, args.dre_lambda, args.dre_threshold_type, args.dre_kappa, args.filter_mae_percentile_threshold, args.niters_gan, args.threshold_type, args.kernel_sigma, args.kappa, args.seed)
        print('\n' + drefile_fullpath)

        path_to_ckpt_in_train = save_models_folder + '/ckpt_{}_DRE-F-SP_{}_lambda_{}_type_{}_kappa_{}_Filter_{}_CcGAN_{}_{}_{}_{}_seed_{}'.format(args.dre_mode, args.dre_net, args.dre_lambda, args.dre_threshold_type, args.dre_kappa, args.filter_mae_percentile_threshold, args.niters_gan, args.threshold_type, args.kernel_sigma, args.kappa, args.seed)
        os.makedirs(path_to_ckpt_in_train, exist_ok=True)

        dre_loss_file_fullpath = save_traincurves_folder + '/train_loss_{}_DRE-F-SP_{}_epochs_{}_lambda_{}_type_{}_kappa_{}_Filter_{}_CcGAN_{}_{}_{}_{:.6f}_seed_{}.png'.format(args.dre_mode, args.dre_net, args.dre_epochs, args.dre_lambda, args.dre_threshold_type, args.dre_kappa, args.filter_mae_percentile_threshold, args.niters_gan, args.threshold_type, args.kernel_sigma, args.kappa, args.seed)

        ### init net
        dre_net = cDR_MLP(args.dre_net, p_dropout=0.5, init_in_dim=args.num_channels*args.img_size*args.img_size, dim_embed=args.dim_embed).cuda()
        dre_net = nn.DataParallel(dre_net)

        #if DR model exists, then load the pretrained model; otherwise, start training the model.
        if not os.path.isfile(drefile_fullpath):
            print("\n Begin Training conditional DR in Feature Space: >>>")

            if args.dre_no_vicinal:
                cdre_training_labels = labels_train
            else:
                cdre_training_labels = labels_all/args.max_label

            if args.filter_mae_percentile_threshold<1.0:
                dre_net, avg_train_loss = train_cdre(kernel_sigma=args.kernel_sigma, kappa=args.dre_kappa, train_images=images_train, train_labels=labels_train, test_labels=cdre_training_labels, dre_net=dre_net, dre_precnn_net=dre_presae_encoder_net, netG=netG, net_y2h=net_y2h, PreNetFilter=PreNetFilter, filter_mae_cutoff_point=filter_mae_cutoff_point, path_to_ckpt=path_to_ckpt_in_train)
            else:
                dre_net, avg_train_loss = train_cdre(kernel_sigma=args.kernel_sigma, kappa=args.dre_kappa, train_images=images_train, train_labels=labels_train, test_labels=cdre_training_labels, dre_net=dre_net, dre_precnn_net=dre_presae_encoder_net, netG=netG, net_y2h=net_y2h, path_to_ckpt=path_to_ckpt_in_train)

            # save model
            torch.save({
            'net_state_dict': dre_net.state_dict(),
            }, drefile_fullpath)
            PlotLoss(avg_train_loss, dre_loss_file_fullpath)

        else:
            # if already trained, load pre-trained DR model
            checkpoint_dre_net = torch.load(drefile_fullpath)
            dre_net.load_state_dict(checkpoint_dre_net['net_state_dict'])
        ##end if not

        # Compute density ratio: function for computing a bunch of images in a numpy array
        def comp_cond_density_ratio(imgs, labels, batch_size=args.samp_batch_size):
            #imgs: a torch tensor
            n_imgs = len(imgs)
            if batch_size>n_imgs:
                batch_size = n_imgs

            ##make sure the last iteration has enough samples
            imgs = torch.cat((imgs, imgs[0:batch_size]), dim=0)
            labels = torch.cat((labels, labels[0:batch_size]), dim=0)

            density_ratios = []
            dre_net.eval()
            dre_presae_encoder_net.eval()
            net_y2h.eval()
            # print("\n Begin computing density ratio for images >>")
            with torch.no_grad():
                n_imgs_got = 0
                while n_imgs_got < n_imgs:
                    batch_images = imgs[n_imgs_got:(n_imgs_got+batch_size)]
                    batch_labels = labels[n_imgs_got:(n_imgs_got+batch_size)]
                    batch_images = batch_images.type(torch.float).cuda()
                    batch_labels = batch_labels.type(torch.float).cuda()
                    batch_features = dre_presae_encoder_net(batch_images)
                    batch_ratios = dre_net(batch_features, net_y2h(batch_labels))
                    density_ratios.append(batch_ratios.cpu().detach())
                    n_imgs_got += batch_size
                ### while n_imgs_got
            density_ratios = torch.cat(density_ratios)
            density_ratios = density_ratios[0:n_imgs].numpy()
            return density_ratios

    else:  ## Unconditional DRE

        ## dre filename
        drefile_fullpath = save_models_folder + '/ckpt_{}_DRE-F-SP_{}_epochs_{}_lambda_{}_CcGAN_{}_{}_{}_{}_seed_{}.pth'.format(args.dre_mode, args.dre_net, args.dre_epochs, args.dre_lambda, args.niters_gan, args.threshold_type, args.kernel_sigma, args.kappa, args.seed)
        print('\n' + drefile_fullpath)

        path_to_ckpt_in_train = save_models_folder + '/ckpt_{}_DRE-F-SP_{}_lambda_{}_CcGAN_{}_{}_{}_{}_seed_{}'.format(args.dre_mode, args.dre_net, args.dre_lambda, args.niters_gan, args.threshold_type, args.kernel_sigma, args.kappa, args.seed)
        os.makedirs(path_to_ckpt_in_train, exist_ok=True)

        dre_loss_file_fullpath = save_traincurves_folder + '/train_loss_{}_DRE-F-SP_{}_epochs_{}_lambda_{}_CcGAN_{}_{}_{}_{}_seed_{}.png'.format(args.dre_mode, args.dre_net, args.dre_epochs, args.dre_lambda, args.niters_gan, args.threshold_type, args.kernel_sigma, args.kappa, args.seed)


        trainset = IMGs_dataset(images_train, labels_train, normalize=True)
        trainloader_dre = torch.utils.data.DataLoader(trainset, batch_size=args.dre_batch_size, shuffle=True, num_workers=args.num_workers)

        dre_net = DR_MLP(args.dre_net, p_dropout=0.5, init_in_dim = args.num_channels*args.img_size*args.img_size).cuda()
        dre_net = nn.DataParallel(dre_net)
        #if DR model exists, then load the pretrained model; otherwise, start training the model.
        if not os.path.isfile(drefile_fullpath):
            print("\n Begin Training unconditional DR in Feature Space: >>>")
            dre_net, avg_train_loss = train_dre(trainloader=trainloader_dre, dre_net=dre_net, dre_precnn_net=dre_presae_encoder_net, netG=netG, net_y2h=net_y2h, path_to_ckpt=path_to_ckpt_in_train)
            # save model
            torch.save({
            'net_state_dict': dre_net.state_dict(),
            }, drefile_fullpath)
            PlotLoss(avg_train_loss, dre_loss_file_fullpath)
        else:
            # if already trained, load pre-trained DR model
            checkpoint_dre_net_i = torch.load(drefile_fullpath)
            dre_net.load_state_dict(checkpoint_dre_net_i['net_state_dict'])
        ##end if not

        # Compute density ratio: function for computing a bunch of images in a numpy array
        def comp_cond_density_ratio(imgs, labels, batch_size=args.samp_batch_size):
            #imgs: a torch tensor
            n_imgs = len(imgs)
            if batch_size>n_imgs:
                batch_size = n_imgs

            ##make sure the last iteration has enough samples
            imgs = torch.cat((imgs, imgs[0:batch_size]), dim=0)

            density_ratios = []
            dre_net.eval()
            dre_presae_encoder_net.eval()
            # print("\n Begin computing density ratio for images >>")
            with torch.no_grad():
                n_imgs_got = 0
                while n_imgs_got < n_imgs:
                    batch_images = imgs[n_imgs_got:(n_imgs_got+batch_size)]
                    batch_images = batch_images.type(torch.float).cuda()
                    batch_features = dre_presae_encoder_net(batch_images)
                    batch_ratios = dre_net(batch_features)
                    density_ratios.append(batch_ratios.cpu().detach())
                    n_imgs_got += batch_size
                ### while n_imgs_got
            density_ratios = torch.cat(density_ratios)
            density_ratios = density_ratios[0:n_imgs].numpy()
            return density_ratios

    ##end if

    # Enhanced sampler based on the trained DR model
    # Rejection Sampling:"Discriminator Rejection Sampling"; based on https://github.com/shinseung428/DRS_Tensorflow/blob/master/config.py
    ## conditionally enhanced sampling
    def fn_enhancedSampler_given_label(nfake, given_label, batch_size=args.samp_batch_size, n_burnin=args.samp_burnin_size, PreNetFilter=None, filter_mae_cutoff_point=1e30):
        ## Burn-in Stage
        burnin_labels = given_label * torch.ones(n_burnin)
        burnin_imgs, _ = fn_sampleGAN_given_labels(burnin_labels, batch_size, to_numpy=False, verbose=False)
        burnin_densityratios = comp_cond_density_ratio(burnin_imgs, burnin_labels)

        # print((burnin_densityratios.min(),np.median(burnin_densityratios),burnin_densityratios.max()))
        M_bar = np.max(burnin_densityratios)
        del burnin_imgs, burnin_densityratios; gc.collect()
        ## Rejection sampling
        enhanced_imgs = []
        # pb = SimpleProgressBar()
        # pbar = tqdm(total=nfake)
        num_imgs = 0
        while num_imgs < nfake:
            #comptue density ratios
            batch_labels = given_label * torch.ones(batch_size)
            batch_imgs, _ = fn_sampleGAN_given_labels(batch_labels, batch_size, to_numpy=False, verbose=False)
            batch_ratios = comp_cond_density_ratio(batch_imgs, batch_labels)
            M_bar = np.max([M_bar, np.max(batch_ratios)])
            #DRE threshold
            batch_p = batch_ratios/M_bar
            batch_psi = np.random.uniform(size=batch_size).reshape(-1,1)
            indx_accept = np.where(batch_psi<=batch_p)[0]
            #comptue mae
            if PreNetFilter is not None:
                assert filter_mae_cutoff_point>0
                batch_dataset = IMGs_dataset(batch_imgs, batch_labels, normalize=False)
                batch_loader = torch.utils.data.DataLoader(batch_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
                _, batch_mae_loss = test_filter_cnn(net=PreNetFilter, testloader=batch_loader, verbose=False)
                indx_accept_filter = np.where(batch_mae_loss<=filter_mae_cutoff_point)[0]
                indx_accept = np.intersect1d(np.array(indx_accept).reshape(-1), np.array(indx_accept_filter).reshape(-1))

            ## take samples which satisfy the condition
            if len(indx_accept)>0:
                enhanced_imgs.append((batch_imgs[indx_accept]).numpy())
            num_imgs+=len(indx_accept)
            del batch_imgs, batch_ratios; gc.collect()
            # pb.update(np.min([float(num_imgs)*100/nfake,100]))
            # pbar.update(len(indx_accept))
        # pbar.close()
        enhanced_imgs = np.concatenate(enhanced_imgs, axis=0)
        enhanced_imgs = enhanced_imgs[0:nfake]
        return enhanced_imgs, given_label*np.ones(nfake)

    ## unconditionally enhanced sampling
    def fn_enhancedSampler(nfake, eval_labels_norm, batch_size=args.samp_batch_size, n_burnin=args.samp_burnin_size):

        ## Burn-in Stage
        burnin_labels = np.random.choice(eval_labels_norm, size=n_burnin, replace=True)
        burnin_labels = torch.from_numpy(burnin_labels).type(torch.float)
        burnin_imgs, _ = fn_sampleGAN_given_labels(burnin_labels, batch_size, to_numpy=False, verbose=False)
        burnin_densityratios = comp_cond_density_ratio(burnin_imgs, burnin_labels)

        # print((burnin_densityratios.min(),np.median(burnin_densityratios),burnin_densityratios.max()))
        M_bar = np.max(burnin_densityratios)
        del burnin_imgs, burnin_densityratios; gc.collect()
        ## Rejection sampling
        enhanced_imgs = []
        enhanced_labels = []
        pbar = tqdm(total=nfake)
        num_imgs = 0
        while num_imgs < nfake:
            #comptue density ratios
            batch_labels = np.random.choice(eval_labels_norm, size=batch_size, replace=True)
            batch_labels = torch.from_numpy(batch_labels).type(torch.float)
            batch_imgs, _ = fn_sampleGAN_given_labels(batch_labels, batch_size, to_numpy=False, verbose=False)
            batch_ratios = comp_cond_density_ratio(batch_imgs, batch_labels)
            M_bar = np.max([M_bar, np.max(batch_ratios)])
            #DRE threshold
            batch_p = batch_ratios/M_bar
            batch_psi = np.random.uniform(size=batch_size).reshape(-1,1)
            indx_accept = np.where(batch_psi<=batch_p)[0]
            ## take samples which satisfy the condition
            if len(indx_accept)>0:
                enhanced_imgs.append((batch_imgs[indx_accept]).numpy())
                enhanced_labels.append((batch_labels[indx_accept]).numpy())
            num_imgs+=len(indx_accept)
            del batch_imgs, batch_ratios; gc.collect()
            pbar.update(len(indx_accept))
        # pbar.close()
        enhanced_imgs = np.concatenate(enhanced_imgs, axis=0)
        enhanced_imgs = enhanced_imgs[0:nfake]
        enhanced_labels = np.concatenate(enhanced_labels, axis=0)
        enhanced_labels = enhanced_labels[0:nfake]
        return enhanced_imgs, enhanced_labels



#######################################################################################
'''                      Fake data generation and evaluation                        '''
#######################################################################################
if args.eval:
    print("\n Evaluation in Mode {}...".format(args.eval_mode))

    PreNetFID = encoder_eval(dim_bottleneck=512)
    PreNetFID = nn.DataParallel(PreNetFID)
    Filename_PreCNNForEvalGANs = eval_models_folder + '/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

    # Diversity: entropy of predicted races within each eval center
    PreNetDiversity = ResNet34_class(num_classes=49, ngpu = NGPU) #49 chair types
    Filename_PreCNNForEvalGANs_Diversity = eval_models_folder + '/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_49_chair_types_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_Diversity)
    PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # for LS
    PreNetLS = ResNet34_regre(ngpu = NGPU)
    Filename_PreCNNForEvalGANs_LS = eval_models_folder + '/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_LS)
    PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])


    #####################
    # generate nfake images
    print("\n Start sampling {} fake images per label from GAN >>>".format(args.nfake_per_label))

    if args.eval_mode == 1: #Mode 1: eval on unique labels used for GAN training
        eval_labels = np.sort(np.array(list(set(labels_train_raw)))) #not normalized
    elif args.eval_mode in [2, 3]: #Mode 2 and 3: eval on all unique labels in the dataset
        eval_labels = np.sort(np.array(list(set(labels_all)))) #not normalized
    else: #Mode 4: eval on a interval [min_label, max_label] with num_eval_labels labels
        eval_labels = np.linspace(np.min(labels_all), np.max(labels_all), args.num_eval_labels) #not normalized

    unique_eval_labels = list(set(eval_labels))
    print("\n There are {} unique eval labels.".format(len(unique_eval_labels)))

    eval_labels_norm = eval_labels/args.max_label #normalized


    if args.subsampling:
        if not args.subsampling_baseline:
            print("\n Generating fake images with subsampling ({}).".format(args.dre_mode))
            fake_images = []
            fake_labels_assigned = []
            for i in trange(len(eval_labels)):
                fake_labels_i = eval_labels_norm[i]*np.ones(args.nfake_per_label)
                if args.filter_mae_percentile_threshold<1.0:
                    fake_images_i, fake_labels_i = fn_enhancedSampler_given_label(args.nfake_per_label, eval_labels_norm[i], batch_size=args.samp_batch_size, n_burnin=args.samp_burnin_size, PreNetFilter=PreNetFilter, filter_mae_cutoff_point=filter_mae_cutoff_point)
                else:
                    fake_images_i, fake_labels_i = fn_enhancedSampler_given_label(args.nfake_per_label, eval_labels_norm[i], batch_size=args.samp_batch_size, n_burnin=args.samp_burnin_size, PreNetFilter=None)
                fake_images.append(fake_images_i)
                fake_labels_assigned.append(fake_labels_i)
            ##end for i
            fake_images = np.concatenate(fake_images, axis=0)
            fake_labels_assigned = np.concatenate(fake_labels_assigned, axis=0)
        else:
            print("\n Generating fake images with the baseline unconditional subsampling.")
            ## The baseline unconditional subsampling method.
            fake_images, fake_labels_assigned = fn_enhancedSampler(nfake=int(args.nfake_per_label*len(eval_labels)), eval_labels_norm=eval_labels_norm, batch_size=args.samp_batch_size, n_burnin=args.samp_burnin_size)
            ## histogram of fake labels: to show the imbalance
            fig = plt.figure()
            ax = plt.subplot(111)
            n, bins, patches = plt.hist(fake_labels_assigned*args.max_label, 100, density=False, facecolor='g', alpha=0.75)
            plt.xlabel('Angle')
            plt.ylabel('Frequency')
            # plt.title('Histogram of Angle')
            plt.grid(True)
            plt.savefig(os.path.join(output_directory, 'histogram_of_fake_labels_under_baseline_unconditional_subsampling.png'))
    else:
        print("\n Generating fake images without subsampling.")
        fake_labels_assigned = []
        for i in range(len(eval_labels)):
            fake_labels_i = eval_labels_norm[i]*np.ones(args.nfake_per_label)
            fake_labels_assigned.append(fake_labels_i)
        ##end for i
        fake_labels_assigned = np.concatenate(fake_labels_assigned, axis=0)
        fake_images, _ = fn_sampleGAN_given_labels(labels=fake_labels_assigned, batch_size=args.samp_batch_size, to_numpy=True, verbose=True)
    assert len(fake_images) == args.nfake_per_label*len(eval_labels)
    assert len(fake_labels_assigned) == args.nfake_per_label*len(eval_labels)

    print("End sampling! We got {} fake images.\n".format(len(fake_images)))

    ## dump fake images for evaluation: NIQE
    if args.dump_fake_for_NIQE:
        dump_fake_images_folder = save_evalresults_folder + "/dump_fake_data_for_NIQE/fake_images_CcGAN_{}_Subsampling_{}_nsamp_{}".format(args.threshold_type, subsampling_method, len(fake_images))
        for i in tqdm(range(len(fake_images))):
            label_i = fake_labels_assigned[i]*args.max_label
            filename_i = dump_fake_images_folder + "/{}_{:.1f}.png".format(i, label_i)
            os.makedirs(os.path.dirname(filename_i), exist_ok=True)
            image_i = fake_images[i]
            image_i = ((image_i*0.5+0.5)*255.0).astype(np.uint8)
            image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
            image_i_pil.save(filename_i)
        #end for i

    else: ###compute eval metrics
        #####################
        # normalize real images
        if args.eval_mode in [1, 3]:
            real_images = (images_train/255.0-0.5)/0.5
            real_labels = labels_train_raw #not normalized
        else: #for both mode 2 and 4
            real_images = (images_all/255.0-0.5)/0.5
            real_labels = labels_all #not normalized

        #######################
        # For each label take nreal_per_label images
        unique_labels_real = np.sort(np.array(list(set(real_labels))))
        indx_subset = []
        for i in range(len(unique_labels_real)):
            label_i = unique_labels_real[i]
            indx_i = np.where(real_labels==label_i)[0]
            np.random.shuffle(indx_i)
            if args.nreal_per_label>1:
                indx_i = indx_i[0:args.nreal_per_label]
            indx_subset.append(indx_i)
        indx_subset = np.concatenate(indx_subset)
        real_images = real_images[indx_subset]
        real_labels = real_labels[indx_subset]

        nfake_all = len(fake_images)
        nreal_all = len(real_images)

        #####################
        # Evaluate FID within a sliding window with a radius R on the label's range (not normalized range, i.e., [min_label,max_label]). The center of the sliding window locate on [min_label+R,...,max_label-R].
        if args.eval_mode == 1:
            center_start = np.min(labels_train_raw)+args.FID_radius
            center_stop = np.max(labels_train_raw)-args.FID_radius
        else:
            center_start = np.min(labels_all)+args.FID_radius
            center_stop = np.max(labels_all)-args.FID_radius

        if args.FID_num_centers<=0 and args.FID_radius==0: #completely overlap
            centers_loc = eval_labels #not normalized
        elif args.FID_num_centers>0:
            centers_loc = np.linspace(center_start, center_stop, args.FID_num_centers) #not normalized
        else:
            print("\n Error.")
        FID_over_centers = np.zeros(len(centers_loc))
        entropies_over_centers = np.zeros(len(centers_loc)) # entropy at each center
        labelscores_over_centers = np.zeros(len(centers_loc)) #label score at each center
        num_realimgs_over_centers = np.zeros(len(centers_loc))
        predicted_class_labels_all = []
        for i in range(len(centers_loc)):
            center = centers_loc[i]
            interval_start = (center - args.FID_radius)
            interval_stop = (center + args.FID_radius)
            indx_real = np.where((real_labels>=interval_start)*(real_labels<=interval_stop)==True)[0]
            np.random.shuffle(indx_real)
            real_images_curr = real_images[indx_real]
            num_realimgs_over_centers[i] = len(real_images_curr)
            indx_fake = np.where((fake_labels_assigned>=(interval_start/args.max_label))*(fake_labels_assigned<=(interval_stop/args.max_label))==True)[0]
            np.random.shuffle(indx_fake)
            fake_images_curr = fake_images[indx_fake]
            fake_labels_assigned_curr = fake_labels_assigned[indx_fake]
            # FID
            FID_over_centers[i] = cal_FID(PreNetFID, real_images_curr, fake_images_curr, batch_size = 500, resize = None)
            # Entropy of predicted class labels
            predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_curr, batch_size=500, num_workers=args.num_workers)
            predicted_class_labels_all.append(predicted_class_labels) # to analyze the balance of predicted labels; show why subsampling method leads to high diversity
            entropies_over_centers[i] = compute_entropy(predicted_class_labels)
            # Label score
            labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fake_labels_assigned_curr, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size = 500, resize = None, num_workers=args.num_workers)

            print("\n [{}/{}] Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}; ET:{}.".format(i+1, len(centers_loc), center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i]))
        # end for i
        # average over all centers
        print("\n {} SFID: {}({}); min/max: {}/{}.".format(subsampling_method, np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
        print("\n {} LS over centers: {}({}); min/max: {}/{}.".format(subsampling_method, np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))
        print("\n {} entropy over centers: {}({}); min/max: {}/{}.".format(subsampling_method, np.mean(entropies_over_centers), np.std(entropies_over_centers), np.min(entropies_over_centers), np.max(entropies_over_centers)))

        # dump FID versus number of samples (for each center) to npy
        dump_fid_ls_entropy_over_centers_filename = save_evalresults_folder + "/CcGAN_{}_Subsampling_{}_fid_ls_entropy_over_centers".format(args.threshold_type, subsampling_method)
        np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers, entropies=entropies_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)

        #####################
        # FID: Evaluate FID on all fake images
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = 500, resize = None)
        print("\n {}: FID of {} fake images: {}.".format(subsampling_method, nfake_all, FID))

        #####################
        # Overall LS: abs(y_assigned - y_predicted)
        ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fake_labels_assigned, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size = 200, resize = None)
        print("\n {}: overall LS of {} fake images: {}({}).".format(subsampling_method, nfake_all, ls_mean_overall, ls_std_overall))

        #####################
        # Dump evaluation results
        eval_results_logging_fullpath = save_evalresults_folder + '/eval_results.txt'
        if not os.path.isfile(eval_results_logging_fullpath):
            eval_results_logging_file = open(eval_results_logging_fullpath, "w")
            eval_results_logging_file.close()
        with open(eval_results_logging_fullpath, 'a') as eval_results_logging_file:
            eval_results_logging_file.write("\n===================================================================================================")
            eval_results_logging_file.write("\n CcGAN: {}_{}_{}; Subsampling: {} \n".format(args.threshold_type, args.kernel_sigma, args.kappa, subsampling_method))
            print(args, file=eval_results_logging_file)
            eval_results_logging_file.write("\n SFID: {}({}).".format(np.mean(FID_over_centers), np.std(FID_over_centers)))
            eval_results_logging_file.write("\n LS: {}({}).".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers)))
            eval_results_logging_file.write("\n Diversity: {}({}).".format(np.mean(entropies_over_centers), np.std(entropies_over_centers)))
            eval_results_logging_file.write("\n FID of {} fake images: {}.".format(len(fake_images), FID))

        ##dump predicted class labels
        pickle_filename = save_evalresults_folder + '/predicted_labels_CcGAN_{}_{}_{}_subsampling_{}.pkl'.format(args.threshold_type, args.kernel_sigma, args.kappa, subsampling_method)
        pickle.dump(predicted_class_labels_all, open(pickle_filename, "wb" ) )


#######################################################################################
'''               Visualize fake images of the trained GAN                          '''
#######################################################################################
if args.visualize_fake_images:

    # First, visualize conditional generation # vertical grid
    ## 10 rows; 10 columns (10 samples for each age)
    n_row = 10
    n_col = 10

    displayed_unique_labels = np.sort(np.array(list(set(labels_all))))
    displayed_labels_indx = (np.linspace(0.05, 0.95, n_row)*len(displayed_unique_labels)).astype(np.int)
    displayed_labels = displayed_unique_labels[displayed_labels_indx] #not normalized
    displayed_normalized_labels = displayed_labels/args.max_label

    if args.subsampling:
        fake_images_view = []
        fake_labels_view = []
        for i in range(len(displayed_normalized_labels)):
            fake_labels_i = displayed_normalized_labels[i]*np.ones(n_col)
            fake_images_i, fake_labels_i = fn_enhancedSampler_given_label(n_col, displayed_normalized_labels[i], batch_size=2, n_burnin=args.samp_burnin_size)
            fake_images_view.append(fake_images_i)
            fake_labels_view.append(fake_labels_i)
        ##end for i
        fake_images_view = np.concatenate(fake_images_view, axis=0)
        fake_labels_view = np.concatenate(fake_labels_view, axis=0)
    else:
        fake_labels_view = []
        for i in range(len(displayed_normalized_labels)):
            fake_labels_i = displayed_normalized_labels[i]*np.ones(n_col)
            fake_labels_view.append(fake_labels_i)
        ##end for i
        fake_labels_view = np.concatenate(fake_labels_view, axis=0)
        fake_images_view, _ = fn_sampleGAN_given_labels(labels=fake_labels_view, batch_size=args.samp_batch_size, to_numpy=True, verbose=False)
    assert len(displayed_normalized_labels) == n_row
    assert len(fake_images_view) == n_col*len(displayed_normalized_labels)
    assert len(fake_labels_view) == n_col*len(displayed_normalized_labels)


    ### output fake images from a trained GAN
    filename_fake_images = save_images_folder + '/CcGAN_{}_Subsampling_{}_fake_images_grid_{}x{}.png'.format(args.threshold_type, subsampling_method, n_row, n_col)

    images_show = np.zeros((n_row*n_col, args.num_channels, args.img_size, args.img_size))
    for i_row in range(n_row):
        curr_label = displayed_normalized_labels[i_row]
        indx_i = np.where(fake_labels_view==curr_label)[0]
        for j_col in range(n_col):
            curr_image = fake_images_view[indx_i[j_col]]
            images_show[i_row*n_col+j_col,:,:,:] = curr_image
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, filename_fake_images, nrow=n_col, normalize=True)

    ### output some real images as baseline
    filename_real_images = save_images_folder + '/real_images_grid_{}x{}.png'.format(n_row, n_col)
    if not os.path.isfile(filename_real_images):
        images_show = np.zeros((n_row*n_col, args.num_channels, args.img_size, args.img_size))
        for i_row in range(n_row):
            # generate 3 real images from each interval
            curr_label = displayed_labels[i_row]
            for j_col in range(n_col):
                indx_curr_label = np.where(labels_all==curr_label)[0]
                np.random.shuffle(indx_curr_label)
                indx_curr_label = indx_curr_label[0]
                images_show[i_row*n_col+j_col] = images_all[indx_curr_label]
        #end for i_row
        images_show = (images_show/255.0-0.5)/0.5
        images_show = torch.from_numpy(images_show)
        save_image(images_show.data, filename_real_images, nrow=n_col, normalize=True)





print("\n===================================================================================================")
