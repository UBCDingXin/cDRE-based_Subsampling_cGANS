print("\n ===================================================================================================")

#----------------------------------------
import argparse
import os
import timeit
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from torch import autograd
from torchvision.utils import save_image
from tqdm import tqdm, trange
import gc
from itertools import groupby
import multiprocessing
import h5py
import pickle
import copy
import shutil


#----------------------------------------
from opts import gen_synth_data_opts
from utils import *
from models import *
from train_cnn import train_cnn, test_cnn
from train_dre import train_dre
from train_cdre import train_cdre
from eval_metrics import compute_FID


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = gen_synth_data_opts()
print(args)

assert int((args.exp_name).split('_')[2]) == args.ntrain

if args.subsampling:
    subsampling_method = args.dre_mode
else:
    subsampling_method = "None"

path_torch_home = os.path.join(args.root_path, 'torch_cache')
os.makedirs(path_torch_home, exist_ok=True)
os.environ['TORCH_HOME'] = path_torch_home

#-------------------------------
# GAN and DRE
dre_precnn_lr_decay_epochs  = (args.dre_precnn_lr_decay_epochs).split("_")
dre_precnn_lr_decay_epochs = [int(epoch) for epoch in dre_precnn_lr_decay_epochs]

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

#-------------------------------
# output folders
eval_models_folder = os.path.join(args.root_path, 'output/eval_models')
assert os.path.exists(eval_models_folder)

output_directory = os.path.join(args.root_path, 'output/{}'.format(args.exp_name))
os.makedirs(output_directory, exist_ok=True)

save_models_folder = os.path.join(output_directory, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)

save_traincurves_folder = os.path.join(output_directory, 'training_curves')
os.makedirs(save_traincurves_folder, exist_ok=True)

save_evalresults_folder = os.path.join(output_directory, 'eval_results')
os.makedirs(save_evalresults_folder, exist_ok=True)

if args.samp_dump_fake_data: #if we need to dump fake images during generation
    dump_fake_images_folder = os.path.join(output_directory, 'dump_fake/BigGAN_epochs_{}_subsampling_{}_NfakePerClass_{}_seed_{}'.format(args.gan_epochs, subsampling_method, args.samp_nfake_per_class, args.seed))



#######################################################################################
'''                                  Load Data                                      '''
#######################################################################################
## generate subset
trainset_h5py_file = args.root_path + '/data/CIFAR{}_trainset_{}_seed_{}.h5'.format(args.num_classes, args.ntrain, args.seed)
hf = h5py.File(trainset_h5py_file, 'r')
images_train = hf['images_train'][:]
labels_train = hf['labels_train'][:]
hf.close()

cifar_testset = torchvision.datasets.CIFAR10(root = os.path.join(args.root_path, 'data'), train=False, download=True)

### compute the mean and std for normalization
### Note that: In GAN-based KD, use computed mean and stds to normalize images for precnn training is better than using [0.5,0.5,0.5]
assert images_train.shape[1]==3
train_means = []
train_stds = []
for i in range(3):
    images_i = images_train[:,i,:,:]
    images_i = images_i/255.0
    train_means.append(np.mean(images_i))
    train_stds.append(np.std(images_i))
## for i
# train_means = [0.5,0.5,0.5]
# train_stds = [0.5,0.5,0.5]

images_test = cifar_testset.data
images_test = np.transpose(images_test, (0, 3, 1, 2))
labels_test = np.array(cifar_testset.targets)

print("\n Training set shape: {}x{}x{}x{}; Testing set shape: {}x{}x{}x{}.".format(images_train.shape[0], images_train.shape[1], images_train.shape[2], images_train.shape[3], images_test.shape[0], images_test.shape[1], images_test.shape[2], images_test.shape[3]))

''' transformations '''
if args.dre_precnn_transform:
    transform_precnn_train = transforms.Compose([
                transforms.RandomCrop((args.img_size, args.img_size), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(train_means, train_stds),
                ])
else:
    transform_precnn_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(train_means, train_stds),
                ])

transform_dre = transforms.Compose([
            # transforms.RandomCrop((args.img_size, args.img_size), padding=4), ## note that GAN training does not involve cropping!!!
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]), ##do not use other normalization constants!!!
            ])

# test set for cnn
transform_precnn_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_means, train_stds),
            ])
testset_precnn = IMGs_dataset(images_test, labels_test, transform=transform_precnn_test)
testloader_precnn = torch.utils.data.DataLoader(testset_precnn, batch_size=100, shuffle=False, num_workers=args.num_workers)


#######################################################################################
'''                  Load pre-trained GAN to Memory (not GPU)                       '''
#######################################################################################
ganfile_fullpath = save_models_folder + '/ckpt_BigGAN_ntrain_{}_epochs_{}_seed_{}/G_ema.pth'.format(args.ntrain, args.gan_epochs, args.seed)
assert os.path.exists(ganfile_fullpath)
ckpt_g = torch.load(ganfile_fullpath)
netG = BigGAN_Generator(resolution=args.img_size, G_attn='0', n_classes=args.num_classes, G_shared=False)
netG.load_state_dict(ckpt_g)
netG = nn.DataParallel(netG)


def fn_sampleGAN_given_label(nfake, given_label, batch_size, pretrained_netG=netG, to_numpy=True):
    raw_fake_images = []
    raw_fake_labels = []
    pretrained_netG = pretrained_netG.cuda()
    pretrained_netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < nfake:
            z = torch.randn(batch_size, args.gan_dim_g, dtype=torch.float).cuda()
            labels = (given_label*torch.ones(batch_size)).type(torch.long).cuda()
            batch_fake_images = pretrained_netG(z, labels)
            raw_fake_images.append(batch_fake_images.cpu())
            raw_fake_labels.append(labels.cpu().view(-1))
            tmp += batch_size

    raw_fake_images = torch.cat(raw_fake_images, dim=0)
    raw_fake_labels = torch.cat(raw_fake_labels)

    if to_numpy:
        raw_fake_images = raw_fake_images.numpy()
        raw_fake_labels = raw_fake_labels.numpy()

    return raw_fake_images[0:nfake], raw_fake_labels[0:nfake]


#######################################################################################
'''                                  DRE Training                                   '''
#######################################################################################
if args.subsampling:
    ##############################################
    ''' Pre-trained CNN for feature extraction '''
    print("\n -----------------------------------------------------------------------------------------")
    print("\n Pre-trained CNN for feature extraction")
    # data loader
    trainset_dre_precnn = IMGs_dataset(images_train, labels_train, transform=transform_precnn_train)
    trainloader_dre_precnn = torch.utils.data.DataLoader(trainset_dre_precnn, batch_size=args.dre_precnn_batch_size_train, shuffle=True, num_workers=args.num_workers)
    # Filename
    filename_precnn_ckpt = save_models_folder + '/ckpt_PreCNNForDRE_{}_epoch_{}_transform_{}_ntrain_{}_seed_{}.pth'.format(args.dre_precnn_net, args.dre_precnn_epochs, args.dre_precnn_transform, args.ntrain, args.seed)
    print('\n' + filename_precnn_ckpt)

    path_to_ckpt_in_train = save_models_folder + '/ckpts_in_train_PreCNNForDRE_{}_ntrain_{}_seed_{}'.format(args.dre_precnn_net, args.ntrain, args.seed)
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    # initialize cnn
    dre_precnn_net = cnn_extract_initialization(args.dre_precnn_net, num_classes=args.num_classes, img_size=args.img_size)
    num_parameters = count_parameters(dre_precnn_net)
    # training
    if not os.path.isfile(filename_precnn_ckpt):
        print("\n Start training CNN for feature extraction in the DRE >>>")
        dre_precnn_net = train_cnn(dre_precnn_net, 'PreCNNForDRE_{}'.format(args.dre_precnn_net), trainloader_dre_precnn, testloader_precnn, epochs=args.dre_precnn_epochs, resume_epoch=args.dre_precnn_resume_epoch, lr_base=args.dre_precnn_lr_base, lr_decay_factor=args.dre_precnn_lr_decay_factor, lr_decay_epochs=dre_precnn_lr_decay_epochs, weight_decay=args.dre_precnn_weight_decay, seed = args.seed, extract_feature=True, path_to_ckpt = path_to_ckpt_in_train)

        # store model
        torch.save({
            'net_state_dict': dre_precnn_net.state_dict(),
        }, filename_precnn_ckpt)
        print("\n End training CNN.")
    else:
        print("\n Loading pre-trained CNN for feature extraction in DRE.")
        checkpoint = torch.load(filename_precnn_ckpt)
        dre_precnn_net.load_state_dict(checkpoint['net_state_dict'])
    #end if

    # testing
    _ = test_cnn(dre_precnn_net, testloader_precnn, extract_feature=True, verbose=True)


    ##############################################
    ''' DRE Training '''
    print("\n -----------------------------------------------------------------------------------------")
    print("\n {} DRE training".format(args.dre_mode))
    
    if args.dre_mode == "conditional":

        ### dataloader
        trainset_dre = IMGs_dataset(images_train, labels_train, transform=transform_dre)
        trainloader_dre = torch.utils.data.DataLoader(trainset_dre, batch_size=args.dre_batch_size, shuffle=True, num_workers=args.num_workers)

        ### dre filename
        drefile_fullpath = save_models_folder + '/ckpt_{}_DRE-F-SP_{}_epochs_{}_lambda_{}_ntrain_{}_seed_{}.pth'.format(args.dre_mode, args.dre_net, args.dre_epochs, args.dre_lambda, args.ntrain, args.seed)
        print('\n' + drefile_fullpath)

        path_to_ckpt_in_train = save_models_folder + '/ckpt_{}_DRE-F-SP_{}_lambda_{}_ntrain_{}_seed_{}'.format(args.dre_mode, args.dre_net, args.dre_lambda, args.ntrain, args.seed)
        os.makedirs(path_to_ckpt_in_train, exist_ok=True)

        dre_loss_file_fullpath = save_traincurves_folder + '/train_loss_{}_DRE-F-SP_{}_epochs_{}_lambda_{}_ntrain_{}_seed_{}.png'.format(args.dre_mode, args.dre_net, args.dre_epochs, args.dre_lambda, args.ntrain, args.seed)

        ### dre training
        dre_net = cDR_MLP(args.dre_net, p_dropout=0.5, init_in_dim = args.num_channels*args.img_size*args.img_size, num_classes = args.num_classes).cuda()
        dre_net = nn.DataParallel(dre_net)
        #if DR model exists, then load the pretrained model; otherwise, start training the model.
        if not os.path.isfile(drefile_fullpath):
            print("\n Begin Training conditional DR in Feature Space: >>>")
            dre_net, avg_train_loss = train_cdre(trainloader_dre, dre_net, dre_precnn_net, netG, path_to_ckpt=path_to_ckpt_in_train)
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
            dre_precnn_net.eval()
            # print("\n Begin computing density ratio for images >>")
            with torch.no_grad():
                n_imgs_got = 0
                while n_imgs_got < n_imgs:
                    batch_images = imgs[n_imgs_got:(n_imgs_got+batch_size)]
                    batch_labels = labels[n_imgs_got:(n_imgs_got+batch_size)]
                    batch_images = batch_images.type(torch.float).cuda()
                    batch_labels = batch_labels.type(torch.long).cuda()
                    _, batch_features = dre_precnn_net(batch_images)
                    batch_ratios = dre_net(batch_features, batch_labels)
                    density_ratios.append(batch_ratios.cpu().detach())
                    n_imgs_got += batch_size
                ### while n_imgs_got
            density_ratios = torch.cat(density_ratios)
            density_ratios = density_ratios[0:n_imgs].numpy()
            return density_ratios

    else: #unconditional
        dre_net_list = []
        for class_i in range(args.num_classes):
            print("\n Fit DRE-F-SP for Class {}...".format(class_i))

            ### data loader
            indx_class_i = np.where(labels_train==class_i)[0]
            images_train_i = images_train[indx_class_i]
            labels_train_i = labels_train[indx_class_i]
            trainset_dre_i = IMGs_dataset(images_train_i, labels_train_i, transform=transform_dre)
            trainloader_dre_i = torch.utils.data.DataLoader(trainset_dre_i, batch_size=args.dre_batch_size, shuffle=True, num_workers=args.num_workers)
            del images_train_i, labels_train_i; gc.collect()

            ### dre filenames
            drefile_fullpath = save_models_folder + '/ckpt_{}_DRE-F-SP_{}_epochs_{}_lambda_{}_ntrain_{}_seed_{}_class_{}.pth'.format(args.dre_mode, args.dre_net, args.dre_epochs, args.dre_lambda, args.ntrain, args.seed, class_i)
            print('\n' + drefile_fullpath)

            path_to_ckpt_in_train = save_models_folder + '/ckpt_{}_DRE-F-SP_{}_lambda_{}_ntrain_{}_seed_{}_class_{}'.format(args.dre_mode, args.dre_net, args.dre_lambda, args.ntrain, args.seed, class_i)
            os.makedirs(path_to_ckpt_in_train, exist_ok=True)

            dre_loss_file_fullpath = save_traincurves_folder + '/train_loss_{}_DRE-F-SP_{}_epochs_{}_lambda_{}_ntrain_{}_seed_{}_class_{}.png'.format(args.dre_mode, args.dre_net, args.dre_epochs, args.dre_lambda, args.ntrain, args.seed, class_i)

            ### dre training
            dre_net_i = DR_MLP(args.dre_net, p_dropout=0.5, init_in_dim = args.num_channels*args.img_size*args.img_size).cuda()
            dre_net_i = nn.DataParallel(dre_net_i)
            #if DR model exists, then load the pretrained model; otherwise, start training the model.
            if not os.path.isfile(drefile_fullpath):
                print("\n Begin Training unconditional DR in Feature Space: >>>")
                dre_net_i, avg_train_loss = train_dre(trainloader=trainloader_dre_i, dre_net=dre_net_i, dre_precnn_net=dre_precnn_net, netG=netG, path_to_ckpt=path_to_ckpt_in_train)
                # save model
                torch.save({
                'net_state_dict': dre_net_i.state_dict(),
                }, drefile_fullpath)
                PlotLoss(avg_train_loss, dre_loss_file_fullpath)
            else:
                # if already trained, load pre-trained DR model
                checkpoint_dre_net_i = torch.load(drefile_fullpath)
                dre_net_i.load_state_dict(checkpoint_dre_net_i['net_state_dict'])
            ##end if not
            dre_net_i.cpu()
            dre_net_list.append(dre_net_i)
        ### end for class_i

        # Compute density ratio: function for computing a bunch of images in a numpy array
        def comp_cond_density_ratio(imgs, labels, batch_size=args.samp_batch_size):
            #imgs: a torch tensor
            n_imgs = len(imgs)
            if batch_size>n_imgs:
                batch_size = n_imgs

            assert torch.sum(labels).item()==len(labels)*labels[0].item() ## make sure all element are the same
            class_i = labels[0].item()

            ##make sure the last iteration has enough samples
            imgs = torch.cat((imgs, imgs[0:batch_size]), dim=0)

            density_ratios = []
            dre_net_i = dre_net_list[class_i] #take the density ratio model for class i
            dre_net_i.cuda()
            dre_net_i.eval()
            dre_precnn_net.eval()
            # print("\n Begin computing density ratio for images >>")
            with torch.no_grad():
                n_imgs_got = 0
                while n_imgs_got < n_imgs:
                    batch_images = imgs[n_imgs_got:(n_imgs_got+batch_size)]
                    batch_images = batch_images.type(torch.float).cuda()
                    _, batch_features = dre_precnn_net(batch_images)
                    batch_ratios = dre_net_i(batch_features)
                    density_ratios.append(batch_ratios.cpu().detach())
                    n_imgs_got += batch_size
                ### while n_imgs_got
            density_ratios = torch.cat(density_ratios)
            density_ratios = density_ratios[0:n_imgs].numpy()
            return density_ratios
    ###end if args.dre_mode

    # Enhanced sampler based on the trained DR model
    # Rejection Sampling:"Discriminator Rejection Sampling"; based on https://github.com/shinseung428/DRS_Tensorflow/blob/master/config.py
    def fn_enhancedSampler_given_label(nfake, given_label, batch_size=args.samp_batch_size, verbose=True):
        ## Burn-in Stage
        n_burnin = args.samp_burnin_size
        burnin_imgs, burnin_labels = fn_sampleGAN_given_label(n_burnin, given_label, batch_size, to_numpy=False)
        burnin_densityratios = comp_cond_density_ratio(burnin_imgs, burnin_labels)
        # print((burnin_densityratios.min(),np.median(burnin_densityratios),burnin_densityratios.max()))
        M_bar = np.max(burnin_densityratios)
        del burnin_imgs, burnin_densityratios; gc.collect()
        ## Rejection sampling
        enhanced_imgs = []
        if verbose:
            pb = SimpleProgressBar()
            # pbar = tqdm(total=nfake)
        num_imgs = 0
        while num_imgs < nfake:
            batch_imgs, batch_labels = fn_sampleGAN_given_label(batch_size, given_label, batch_size, to_numpy=False)
            batch_ratios = comp_cond_density_ratio(batch_imgs, batch_labels)
            M_bar = np.max([M_bar, np.max(batch_ratios)])
            #threshold
            batch_p = batch_ratios/M_bar
            batch_psi = np.random.uniform(size=batch_size).reshape(-1,1)
            indx_accept = np.where(batch_psi<=batch_p)[0]
            if len(indx_accept)>0:
                enhanced_imgs.append(batch_imgs[indx_accept])
            num_imgs+=len(indx_accept)
            del batch_imgs, batch_ratios; gc.collect()
            if verbose:
                pb.update(np.min([float(num_imgs)*100/nfake,100]))
                # pbar.update(len(indx_accept))
        # pbar.close()
        enhanced_imgs = np.concatenate(enhanced_imgs, axis=0)
        enhanced_imgs = enhanced_imgs[0:nfake]
        return enhanced_imgs, given_label*np.ones(nfake)


#######################################################################################
'''                      Fake data generation and evaluation                        '''
#######################################################################################

if args.eval:
    ##############################################
    ''' Generate fake data '''
    print('\n Start generating fake data...')
    fake_images = []
    fake_labels = []
    for i in range(args.num_classes):
        print("\n Generate {} fake images for class {}/{}.".format(args.samp_nfake_per_class, i+1, args.num_classes))
        if args.subsampling:
            fake_images_i, fake_labels_i = fn_enhancedSampler_given_label(nfake=args.samp_nfake_per_class, given_label=i, batch_size=args.samp_batch_size)
        else:
            fake_images_i, fake_labels_i = fn_sampleGAN_given_label(nfake=args.samp_nfake_per_class, given_label=i, batch_size=args.samp_batch_size, pretrained_netG=netG, to_numpy=True)
        assert fake_images_i.max()<=1 and fake_images_i.min()>=-1
        ## denormalize images to save memory
        fake_images_i = (fake_images_i*0.5+0.5)*255.0
        fake_images_i = fake_images_i.astype(np.uint8)
        assert fake_images_i.max()>1 and fake_images_i.max()<=255.0

        
        if args.samp_dump_fake_data:## if we need to dump fake images to local since generating images for a lot of classes may be time consuming.
            ## dump to local
            dump_fake_images_folder_i = os.path.join(dump_fake_images_folder, 'class_{}'.format(i+1))
            if os.path.exists(dump_fake_images_folder_i):
                shutil.rmtree(dump_fake_images_folder_i)
            os.makedirs(dump_fake_images_folder_i)
            print('\n Dumping class {} fake data to local...'.format(i+1))
            for j in trange(len(fake_images_i)):
                path_img_j = os.path.join(dump_fake_images_folder_i, '{}.png'.format(j))
                img_j = np.transpose(fake_images_i[j], (1,2,0)) #CxHxW to HxWxC
                img_j = Image.fromarray(np.uint8(img_j), mode = 'RGB')
                img_j.save(path_img_j)
            assert len(os.listdir(dump_fake_images_folder_i)) == len(fake_images_i)
        else:
            ## save in a list
            fake_images.append(fake_images_i)
            fake_labels.append(fake_labels_i.reshape(-1))
    del fake_images_i, fake_labels_i; gc.collect()
    print('\n End generating fake data!')
    
    if args.samp_dump_fake_data:
        fake_images = []
        fake_labels = []
        for i in range(args.num_classes):
            print('\n Loading class {} fake data from local...'.format(i+1))
            dump_fake_images_folder_i = os.path.join(dump_fake_images_folder, 'class_{}'.format(i+1))
            assert os.path.exists(dump_fake_images_folder_i)
            filenames_i = os.listdir(dump_fake_images_folder_i)
            assert len(filenames_i) == args.samp_nfake_per_class
            for j in trange(len(filenames_i)):
                filename_j_in_i = os.path.join(dump_fake_images_folder_i, filenames_i[j])
                img_j_in_i = Image.open(filename_j_in_i)
                img_j_in_i = np.array(img_j_in_i)
                img_j_in_i = np.transpose(img_j_in_i, (2,0,1))
                assert img_j_in_i.shape[0]==3
                img_j_in_i = img_j_in_i[np.newaxis,:,:,:]
                fake_images.append(img_j_in_i)
                fake_labels.append(i)
            ##end for j
        ##end for i
    fake_images = np.concatenate(fake_images, axis=0)
    if not args.samp_dump_fake_data:
        fake_labels = np.concatenate(fake_labels)
    else:
        fake_labels = np.array(fake_labels)
    assert len(fake_images) == len(fake_labels)

    ##############################################
    ''' Load real data '''
    print('\n Loading real data for evaluation...')
    del images_train, labels_train; gc.collect()
    realdata_h5py_file = args.root_path + '/data/CIFAR{}_trainset_50000_seed_{}.h5'.format(args.num_classes, args.seed)
    hf = h5py.File(realdata_h5py_file, 'r')
    real_images = hf['images_train'][:]
    real_labels = hf['labels_train'][:]
    hf.close()

    assert real_images.max()>1 and fake_images.max()>1
    real_images = (real_images/255.0-0.5)/0.5
    fake_images = (fake_images/255.0-0.5)/0.5



    ##############################################
    ''' Compute Intra-FID '''
    PreNetFID = Inception3(num_classes=args.num_classes, aux_logits=True, transform_input=False)
    Filename_PreCNNForEvalGANs = eval_models_folder + '/ckpt_PreCNNForEvalGANs_InceptionV3_epoch_200_SEED_2019_Transformation_True'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetFID = nn.DataParallel(PreNetFID).cuda()
    PreNetFID.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # PreNetFID = inception_v3(pretrained=True, transform_input=True)
    # PreNetFID = nn.DataParallel(PreNetFID).cuda()

    print('\n Start computing Intra-FID...')
    start = timeit.default_timer()
    FID_all = np.zeros(args.num_classes)
    for i in range(args.num_classes):
        indx_real_i = np.where(real_labels==i)[0]
        real_images_i = real_images[indx_real_i]
        indx_fake_i = np.where(fake_labels==i)[0]
        fake_images_i = fake_images[indx_fake_i]
        ## compute FID within each class
        FID_all[i] = compute_FID(PreNetFID, real_images_i, fake_images_i, batch_size = args.eval_FID_batch_size, resize = (299, 299))
        print("\r Class:{}; Real:{}; Fake:{}; FID:{}; Time elapses:{}s.".format(i+1, len(real_images_i), len(fake_images_i), FID_all[i], timeit.default_timer()-start))
    ##end for i
    print('\n End computing Intra-FID')
    # average over all classes
    print("\n {} Intra-FID: {}({}); min/max: {}/{}.".format(subsampling_method, np.mean(FID_all), np.std(FID_all), np.min(FID_all), np.max(FID_all)))

    # dump FID versus class to npy
    dump_fids_filename = save_evalresults_folder + "/BigGAN_epoch_{}_subsampling_{}_fids".format(args.gan_epochs, subsampling_method)
    np.savez(dump_fids_filename, fids=FID_all)

    #####################
    # FID: Evaluate FID on all fake images
    indx_shuffle_real = np.arange(len(real_images)); np.random.shuffle(indx_shuffle_real)
    indx_shuffle_fake = np.arange(len(fake_images)); np.random.shuffle(indx_shuffle_fake)
    FID = compute_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = 500, resize = (299, 299))
    print("\n BigGAN, {}: FID of {} fake images: {}.".format(subsampling_method, len(fake_images), FID))

    #####################
    # Dump evaluation results
    eval_results_fullpath = os.path.join(save_evalresults_folder, 'BigGAN_epoch_{}_subsampling_{}.txt'.format(args.gan_epochs, subsampling_method))
    if not os.path.isfile(eval_results_fullpath):
        eval_results_logging_file = open(eval_results_fullpath, "w")
        eval_results_logging_file.close()
    with open(eval_results_fullpath, 'a') as eval_results_logging_file:
        eval_results_logging_file.write("\n===================================================================================================")
        eval_results_logging_file.write("\n Subsampling {} \n".format(subsampling_method))
        print(args, file=eval_results_logging_file)
        eval_results_logging_file.write("\n Intra-FID: {}({}); min/max: {}/{}.".format(np.mean(FID_all), np.std(FID_all), np.min(FID_all), np.max(FID_all)))
        eval_results_logging_file.write("\n FID of {} fake images: {}.".format(len(fake_images), FID))
##end if args.eval


if args.eval_ref:
    ##############################################
    ''' Load real data '''
    print('\n Loading real data for evaluation...')
    del images_train, labels_train; gc.collect()
    realdata_h5py_file = args.root_path + '/data/CIFAR{}_trainset_50000_seed_{}.h5'.format(args.num_classes, args.seed)
    hf = h5py.File(realdata_h5py_file, 'r')
    train_images = hf['images_train'][:]
    train_labels = hf['labels_train'][:]
    hf.close()

    cifar_testset = torchvision.datasets.CIFAR10(root = os.path.join(args.root_path, 'data'), train=False, download=True)
    test_images = cifar_testset.data
    test_images = np.transpose(test_images, (0, 3, 1, 2))
    test_labels = np.array(cifar_testset.targets)

    train_images = (train_images/255.0-0.5)/0.5
    test_images = (test_images/255.0-0.5)/0.5

    ##############################################
    ''' Compute Intra-FID '''
    PreNetFID = Inception3(num_classes=args.num_classes, aux_logits=True, transform_input=False)
    Filename_PreCNNForEvalGANs = eval_models_folder + '/ckpt_PreCNNForEvalGANs_InceptionV3_epoch_200_SEED_2019_Transformation_True'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetFID = nn.DataParallel(PreNetFID).cuda()
    PreNetFID.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # PreNetFID = inception_v3(pretrained=True, transform_input=True)
    # PreNetFID = nn.DataParallel(PreNetFID).cuda()

    print('\n Start computing Intra-FID...')
    start = timeit.default_timer()
    FID_all = np.zeros(args.num_classes)
    for i in range(args.num_classes):
        indx_train_i = np.where(train_labels==i)[0]
        train_images_i = train_images[indx_train_i]
        indx_test_i = np.where(test_labels==i)[0]
        test_images_i = test_images[indx_test_i]
        ## compute FID within each class
        FID_all[i] = compute_FID(PreNetFID, train_images_i, test_images_i, batch_size = args.eval_FID_batch_size, resize = (299, 299))
        print("\r Ref. Class:{}; Train:{}; Test:{}; FID:{}; Time elapses:{}s.".format(i+1, len(train_images_i), len(test_images_i), FID_all[i], timeit.default_timer()-start))
    ##end for i
    print('\n End computing Intra-FID')
    # average over all classes
    print("\n Ref. Intra-FID: {}({}); min/max: {}/{}.".format(np.mean(FID_all), np.std(FID_all), np.min(FID_all), np.max(FID_all)))

    #####################
    # FID: Evaluate FID on all fake images
    indx_shuffle_train = np.arange(len(train_images)); np.random.shuffle(indx_shuffle_train)
    indx_shuffle_test = np.arange(len(test_images)); np.random.shuffle(indx_shuffle_test)
    FID = compute_FID(PreNetFID, train_images[indx_shuffle_train], test_images[indx_shuffle_test], batch_size = 500, resize = (299, 299))
    print("\n Ref. FID of {} test images: {}.".format(len(test_images), FID))
### end if args.eval_ref



#######################################################################################
'''               Visualize fake images of the trained GAN                          '''
#######################################################################################
if args.visualize_fake_images:

    # First, visualize conditional generation # vertical grid
    ## 10 rows; 10 columns (10 samples for each class)
    n_row = args.num_classes
    n_col = 10

    fake_images_view = []
    fake_labels_view = []
    for i in range(args.num_classes):
        fake_labels_i = i*np.ones(n_col)
        if args.subsampling:
            fake_images_i, _ = fn_enhancedSampler_given_label(nfake=n_col, given_label=i, batch_size=100, verbose=False)
        else:
            fake_images_i, _ = fn_sampleGAN_given_label(nfake=n_col, given_label=i, batch_size=100, pretrained_netG=netG, to_numpy=True)
        fake_images_view.append(fake_images_i)
        fake_labels_view.append(fake_labels_i)
    ##end for i
    fake_images_view = np.concatenate(fake_images_view, axis=0)
    fake_labels_view = np.concatenate(fake_labels_view, axis=0)

    ### output fake images from a trained GAN
    filename_fake_images = save_evalresults_folder + '/BigGAN_epoch_{}_subsampling_{}_fake_image_grid_{}x{}.png'.format(args.gan_epochs, subsampling_method, n_row, n_col)
    
    images_show = np.zeros((n_row*n_col, args.num_channels, args.img_size, args.img_size))
    for i_row in range(n_row):
        indx_i = np.where(fake_labels_view==i_row)[0]
        for j_col in range(n_col):
            curr_image = fake_images_view[indx_i[j_col]]
            images_show[i_row*n_col+j_col,:,:,:] = curr_image
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, filename_fake_images, nrow=n_col, normalize=True)

### end if args.visualize_fake_images




print("\n ===================================================================================================")