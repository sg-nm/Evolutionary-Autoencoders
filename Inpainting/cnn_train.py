#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import random
from skimage.measure import compare_psnr

from cnn_model import CGP2CNN_autoencoder


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.apply(weights_init_normal_)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_normal_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


# __init__: load dataset
# __call__: training the CNN defined by CGP list
class CNN_train():
    def __init__(self, dataset_name, validation=True, verbose=True, imgSize=64, batchsize=16):
        # dataset_name: name of data set ('celebA' or 'cars' or 'svhn')
        # validation  : [True]  model train/validation mode
        #               [False] model test mode for final evaluation of the evolved model
        # verbose     : flag of display
        self.verbose = verbose
        self.imgSize = imgSize
        self.validation = validation
        self.batchsize = batchsize
        self.channel = 3
        num_work = 2

        # load dataset
        if dataset_name == 'svhn' or dataset_name == 'celebA' or dataset_name == 'cars':
            if dataset_name == 'svhn':
                if self.validation:
                    dataset = dset.SVHN(root='./svhn', split='train', download=True,
                            transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.Scale(self.imgSize),transforms.ToTensor(),]))
                    test_dataset = dset.SVHN(root='./svhn', split='extra', download=True, 
                            transform=transforms.Compose([transforms.Scale(self.imgSize), transforms.ToTensor(),]))
                else:
                    dataset = dset.SVHN(root='./svhn', split='train', download=True,
                            transform=transforms.Compose([transforms.Scale(self.imgSize),transforms.ToTensor(),]))
                    test_dataset = dset.SVHN(root='./svhn', split='test', download=True, 
                            transform=transforms.Compose([transforms.Scale(self.imgSize),transforms.ToTensor(),]))
                self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batchsize, shuffle=True, num_workers=int(num_work), drop_last=True)
                self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=int(num_work))
            elif dataset_name == 'celebA':
                if self.validation:
                    data_transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor()])
                    test_data_transform = transforms.Compose([transforms.ToTensor()])
                    dataset = dset.ImageFolder(root='/dataset/celebA/train', transform=data_transform)
                    test_dataset = dset.ImageFolder(root='/dataset/celebA/val', transform=test_data_transform)
                else:
                    data_transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor()])
                    test_data_transform = transforms.Compose([transforms.ToTensor()])
                    dataset = dset.ImageFolder(root='/dataset/celebA/train', transform=data_transform)
                    test_dataset = dset.ImageFolder(root='/dataset/celebA/test', transform=test_data_transform)
                self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batchsize, shuffle=True, num_workers=int(num_work), drop_last=True)
                self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=int(num_work))
            elif dataset_name == 'cars':
                if self.validation:
                    data_transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor()])
                    test_data_transform = transforms.Compose([transforms.ToTensor()])
                    dataset = dset.ImageFolder(root='/dataset/cars/train', transform=data_transform)
                    test_dataset = dset.ImageFolder(root='/dataset/cars/val', transform=test_data_transform)
                else:
                    data_transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor()])
                    test_data_transform = transforms.Compose([transforms.ToTensor()])
                    dataset = dset.ImageFolder(root='/dataset/cars/retrain', transform=data_transform)
                    test_dataset = dset.ImageFolder(root='/dataset/cars/test', transform=test_data_transform)
                self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batchsize, shuffle=True, num_workers=int(num_work), drop_last=True)
                self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=int(num_work))
            print('train num', len(self.dataloader.dataset))
            print('test num ', len(self.test_dataloader.dataset))
        else:
            print('\tInvalid input dataset name at CNN_train()')
            exit(1)

    def __call__(self, cgp, gpuID, epoch_num=200, out_model='mymodel.model', mask_type='center'):
        if self.verbose:
            print('GPUID    :', gpuID)
            print('epoch_num:', epoch_num)
            print('mast type:', mask_type)

        
        torch.backends.cudnn.benchmark = True
        model = CGP2CNN_autoencoder(cgp, self.channel, self.imgSize)
        model.cuda(gpuID)
        criterion = nn.MSELoss()
        criterion.cuda(gpuID)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

        input = torch.FloatTensor(self.batchsize, self.channel, self.imgSize, self.imgSize)
        input = input.cuda(gpuID)
        # center mask
        mask = torch.FloatTensor(1, self.channel, self.imgSize, self.imgSize).fill_(1.0)
        scale = 0.25
        l = int(self.imgSize*scale)
        u = int(self.imgSize*(1.0-scale))
        mask[:,0, l:u, l:u] = 0.0
        mask[:,1, l:u, l:u] = 0.0
        mask[:,2, l:u, l:u] = 0.0
        mask = mask.cuda(gpuID)
        mask = Variable(mask)
        # for outputs
        if not os.path.exists('./outputs'):
            os.mkdir('./outputs')

        for epoch in range(1, epoch_num+1):
            start_time = time.time()
            if self.verbose:
                print('epoch', epoch)
            train_loss = 0
            ite = 0
            for module in model.children():
                module.train(True)
            for _, (data, target) in enumerate(self.dataloader):
                # data = data[:,0:1,:,:] # in the case of using gray-scale images
                data, target = data.cuda(gpuID), target.cuda(gpuID)
                input.resize_as_(data).copy_(data)
                input_ = Variable(input)
                if mask_type == 'center':
                    data_noise = torch.mul(input_, mask)
                elif mask_type == 'pixel':
                    data_noise = self.random_pixel_mask(input_, [self.imgSize, self.imgSize], gpuID)
                else:
                    data_noise = self.half_mask(input_, gpuID)
                optimizer.zero_grad()
                try:
                    output = model(data_noise)
                except:
                    import traceback
                    traceback.print_exc()
                    return 0.
                loss = criterion(output, input_)
                train_loss += loss.data[0]
                loss.backward()
                optimizer.step()
                if ite == 100:
                    vutils.save_image(data_noise.data, './noise_samples%d.png' % gpuID, normalize=False)
                    vutils.save_image(input_.data, './org_samples%d.png' % gpuID, normalize=False)
                    vutils.save_image(output.data, './output%d.png' % gpuID, normalize=False)
                ite += 1
                if ite == 1000:
                    break
            print('Train set : Average loss: {:.4f}'.format(train_loss))
            print('time ', time.time()-start_time)
            if self.validation: # for evolution
                if epoch == epoch_num:
                    for module in model.children():
                        module.train(False)
                    t_loss = self.__test_per_std(model, criterion, gpuID, input, mask_type, mask)
            else: # for retrain the best model
                if epoch % 10 == 0:
                    for module in model.children():
                        module.train(False)
                    t_loss = self.__test_per_std(model, criterion, gpuID, input, mask_type, mask)
                if epoch == 200:
                    for param_group in optimizer.param_groups:
                        tmp = param_group['lr']
                    tmp *= 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = tmp
                if epoch == 400:
                    for param_group in optimizer.param_groups:
                        tmp = param_group['lr']
                    tmp *= 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = tmp
        
        model = model.cpu()
        torch.save(model.state_dict(), './model_%d.pth' % int(gpuID))
        return t_loss


    def random_pixel_mask(self, inp, image_shape, gpuID, fraction_masked=0.8):
        mask = torch.rand(image_shape)
        mask[mask<fraction_masked] = 0.0
        mask = mask.cuda(gpuID)
        mask = Variable(mask)
        out = torch.mul(inp, mask)
        return out

    def half_mask(self, inp, gpuID):
        mask = torch.FloatTensor(1, inp.size(1), inp.size(2), inp.size(3)).fill_(1.0)
        w = int(inp.size(2)/2)
        r = np.random.rand()
        if r < 0.25: # left
            mask[:,:,:, 0:w] = 0.0
        elif r < 0.5: # up
            mask[:,:,0:w,:] = 0.0
        elif r < 0.75: # right
            mask[:,:,:,w:inp.size(3)] = 0.0
        else: # bottom
            mask[:,:,w:inp.size(2),:] = 0.0

        mask = mask.cuda(gpuID)
        mask = Variable(mask)
        out = torch.mul(inp, mask)
        return out

    def calcPSNR(self, image1, image2):
        image1 *= 255
        image2 *= 255
        image1[image1>255] = 255
        image1[image1<0] = 0
        image2[image2>255] = 255
        image2[image2<0] = 0
        return compare_psnr(image1, image2, data_range=255)


    def __test_per_std(self, model, criterion, gpuID, input, mask_type, mask):
        test_loss = 0
        psnr = 0
        psnr2 = 0
        ite = 0
        for _, (data, target) in enumerate(self.test_dataloader):
            # data = data[:,0:1,:,:] # in the case of gray-scale
            data, target = data.cuda(gpuID), target.cuda(gpuID)
            input.resize_as_(data).copy_(data)
            input_ = Variable(input, volatile=True)
            if mask_type == 'center':
                data_noise = torch.mul(input_, mask)
            elif mask_type == 'pixel':
                data_noise = self.random_pixel_mask(input_, [self.imgSize, self.imgSize], gpuID)
            else:
                data_noise = self.half_mask(input_, gpuID)
            try:
                output = model(data_noise)
            except:
                import traceback
                traceback.print_exc()
                return 0.
            loss = criterion(output, input_)
            psnr += -10 * math.log10(loss.data[0])
            test_loss += loss.data[0]

            if ite < 2000:
                vutils.save_image(output.data, './outputs/test_output_%02d.png' % int(ite), normalize=False)
                vutils.save_image(data_noise.data, './outputs/test_output_%02d_.png' % int(ite), normalize=False)
                vutils.save_image(input_.data, './outputs/test_output_%02d__.png' % int(ite), normalize=False)
            ite += 1

            # # PSNR
            # img1 = (output.data).cpu().numpy()
            # img2 = (input_.data).cpu().numpy()
            # imdf = img2*255.0 - img1*255.0
            # imdf = imdf ** 2
            # rmse = np.sqrt(np.mean(imdf))
            # psnr2 += 20 * math.log10(255.0/rmse)
            
        psnr /= (ite)
        psnr2 /= (ite)
        test_loss /= (ite)
        print('Test PSNR: {:.4f}'.format(psnr))
        # print('Test PSNR2: {:.4f}'.format(psnr2))
        print('Test loss:  {:.4f}'.format(test_loss))

        return psnr
