import os
import time
import copy
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from utils import *

from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np





def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--modelevl', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode')  # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=300, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real',
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='results', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--naive', action='store_true')
    parser.add_argument('--doorping', action='store_true')
    parser.add_argument('--test_model', action='store_true')
    parser.add_argument('--layer', type=int, default=-2)
    parser.add_argument('--portion', type=float, default=0)
    parser.add_argument('--backdoor_size', type=int, default=2)
    parser.add_argument('--support_dataset', default=None, type=str)
    parser.add_argument('--trigger_label', type=int, default=0)
    parser.add_argument('--model_init', type=str, default="imagenet-pretrained")
    parser.add_argument('--invisible', action='store_false')
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=10)

    args = parser.parse_args()
    args.clean = True
    args.outer_loop, args.inner_loop ,args.syn= get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False
    args.doorping_trigger = False
    args.invisible_trigger = False

    args.rat =1
    args.lr_img=0.1
    PP = 1
    args.ratio=0.007


    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)



    eval_it_pool = np.arange(0, args.Iteration + 1,
                             50).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [
        args.Iteration]  # The list of iterations when we evaluate models and record results.


    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset,
                                                                                                         args.data_path,
                                                                                                         args)
    _, _, _, _, _, _, _, _, testloader_trigger = get_dataset(args.dataset, args.data_path, args)
    if args.doorping:
        doorping_perm = np.random.permutation(len(dst_train))[0: int(len(dst_train) * args.portion)]
        input_size = (im_size[0], im_size[1], channel)
        trigger_loc = (im_size[0] - 1 - args.backdoor_size, im_size[0] - 1)
        args.init_trigger = np.zeros(input_size)
        init_backdoor = np.random.randint(1, 256, (args.backdoor_size, args.backdoor_size, channel))
        args.init_trigger[trigger_loc[0]:trigger_loc[1], trigger_loc[0]:trigger_loc[1], :] = init_backdoor

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        args.mask = torch.FloatTensor(np.float32(args.init_trigger > 0).transpose((2, 0, 1))).to(args.device)
        if channel == 1:
            args.init_trigger = np.squeeze(args.init_trigger)
        args.init_trigger = Image.fromarray(args.init_trigger.astype(np.uint8))
        args.init_trigger = transform(args.init_trigger)
        args.init_trigger = args.init_trigger.unsqueeze(0).to(args.device, non_blocking=True)
        args.init_trigger = args.init_trigger.requires_grad_()  # size 1*3x32x32

    if args.invisible:
        doorping_perm = np.random.permutation(len(dst_train))[0: int(len(dst_train) * args.portion)]

        for img, label in dst_test:
            if label == args.trigger_label:
                args.init_trigger = img
                break


        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        args.init_trigger = args.init_trigger.unsqueeze(0).to(args.device, non_blocking=True)
        args.init_trigger = args.init_trigger.requires_grad_()

        input_size = (im_size[0], im_size[1], channel)
        args.black = np.zeros(input_size)
        args.black = transform(args.black)
        args.black = args.black.unsqueeze(0).to(args.device, non_blocking=True)

    def get_real_gw(num_classes, args, net_parameters, criterion, net):
        gw_real_list = []
        for c in range(num_classes):
            img_real = get_images(c, args.batch_real)
            lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
            output_real = net(img_real)
            loss_real = criterion(output_real, lab_real)
            gw_real = torch.autograd.grad(loss_real, net_parameters, create_graph=False)
            gw_real_list.append(gw_real)  # 假设梯度是单元素元组，取第0项

        # 解包列表为独立变量返回
        return tuple(gw_real_list)


    def get_trigger_gw(num_classes,image_tri_augmented1):
        loss_triggers = []
        gw_triggers = []

        for c in range(num_classes):
            # 准备触发标签
            trigger_label = torch.tensor(c).unsqueeze(0).to(args.device).repeat(1)

            # 计算输出和损失
            output_trigger = net(image_tri_augmented1)
            loss_trigger = criterion(output_trigger, trigger_label)
            loss_triggers.append(loss_trigger)

            # 计算梯度
            gw_trigger = torch.autograd.grad(loss_trigger, net_parameters, create_graph=True)
            gw_triggers.append(gw_trigger)

        # 解包返回（假设 num_classes=10）
        return tuple(gw_triggers)  # 返回 gw_trigger0, gw_trigger1, ..., gw_trigger9
    def cut_trigger(init_trigger_set):
        init_trigger = init_trigger_set.to(args.device)
        half_length = init_trigger_set.shape[2] // 2
        # import pdb; pdb.set_trace()
        a, b, e, d = init_trigger[:, :, :half_length, :half_length], init_trigger[:, :,
                                                                     half_length:,
                                                                     :half_length], init_trigger[
                                                                                    :,
                                                                                    :,
                                                                                    :half_length,
                                                                                    half_length:], init_trigger[
                                                                                                   :,
                                                                                                   :,
                                                                                                   half_length:,
                                                                                                   half_length:]
        a, b, e, d = nn.functional.interpolate(a, scale_factor=2, mode='bilinear',
                                               align_corners=True), nn.functional.interpolate(b,
                                                                                              scale_factor=2,
                                                                                              mode='bilinear',
                                                                                              align_corners=True), \
            nn.functional.interpolate(e, scale_factor=2, mode='bilinear',
                                      align_corners=True), nn.functional.interpolate(d,
                                                                                     scale_factor=2,
                                                                                     mode='bilinear',
                                                                                     align_corners=True)
        # a, b, c, d = image_syn.clone(), image_syn.clone(), image_syn.clone(), image_syn.clone()
        image_tri_augmented = torch.cat([a, b, e, d], dim=0)
        image_tri_augmented.requires_grad_()
        image_tri_augmented2 = image_tri_augmented.clone().detach()
        return image_tri_augmented2,half_length


    cta = []
    asr = []
    for exp in range(1):
        args.Iteration = 300
        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]  # size 1x3x32x32
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]

        # for i, lab in enumerate(labels_all):
        #     indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)



        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)

        def get_images(c, n):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]
        def get_images_fixed(c, n=10):  # Get first n images from class c (fixed selection)
            idx_fixed = indices_class[c][10:n+10]  # Select first n indices without shuffling
            return images_all[idx_fixed]

        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float,
                                requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc) * i for i in range(num_classes)], dtype=torch.long,
                                 requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images_fixed(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')
        image_rrrreal= copy.deepcopy(image_syn.detach())


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)  # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        optimizer_tri=torch.optim.Adam([args.init_trigger],lr=0.08,betas=[0.9,0.99])
        optimizer_tri.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        accs__test_trigger = list()
        open=0
        syn=0
        it=0
        while it < args.Iteration + 1:


            ''' Evaluate synthetic data '''
            if it in eval_it_pool :

                if args.dsa:
                    args.epoch_eval_train = 1000
                    args.dc_aug_param = None
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model,
                                                    args.ipc)  # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                if args.dsa or args.dc_aug_param['strategy'] != 'none':
                    args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                else:
                    args.epoch_eval_train = 300

                accs = []
                accs_trigger = []


                for it_eval in range(1):

                    net_eval = get_network(args.modelevl, channel, num_classes, im_size).to(
                        args.device)  # get a random model
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                        label_syn.detach())  # avoid any unaware modification
                    _, acc_train, acc_test, acc_test_trigger = evaluate_synset(it_eval, net_eval, image_syn_eval,
                                                                               label_syn_eval, testloader,
                                                                               testloader_trigger, channel, num_classes,
                                                                               im_size, args)


                    accs.append(acc_test)
                    accs_trigger.append(acc_test_trigger)


                print(
                    'Evaluate %d random %s, clean mean = %.4f clean std = %.4f, trigger mean = %.4f trigger std = %.4f\n-------------------------' % (
                        len(accs), args.model, np.mean(accs), np.std(accs), np.mean(accs_trigger),
                        np.std(accs_trigger)))






            ''' Train synthetic data '''

            net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model

            net.train()
            # embed=net.embed_channel_avg
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0

            for ol in range(args.outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():  # BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train()  # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real)  # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  # BatchNorm
                            module.eval()  # fix mu and sigma of every BatchNorm layer

                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(args.device)
                if PP!=0:
                    for c in range(num_classes):
                        img_real = get_images(c, args.batch_real)
                        lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                        img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape(
                            (args.ipc, channel, im_size[0], im_size[1]))
                        lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                        if args.dsa:
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                            img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                        output_real = net(img_real)
                        loss_real = criterion(output_real, lab_real)
                        gw_real_i = torch.autograd.grad(loss_real, net_parameters)

                        output_syn = net(img_syn)
                        loss_syn = criterion(output_syn, lab_syn)
                        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                        gw_real = gw_real_i

                        loss += match_loss(gw_syn, gw_real, args)

                    optimizer_img.zero_grad()
                    loss.backward()
                    optimizer_img.step()
                    loss_avg += loss.item()

                loss = torch.tensor(0.0).to(args.device)
                if syn == 0 or PP==0:

                    for c in range(num_classes):
                        if c == 0:
                            for c in range(num_classes):
                                img_real = get_images(0, args.batch_real)
                                lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                                img_syn = image_syn[0 * args.ipc:(0 + 1) * args.ipc].reshape(
                                    (args.ipc, channel, im_size[0], im_size[1]))
                                lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c
                                init_trigger = args.init_trigger.to(args.device)
                                if args.dsa:
                                    seed = int(time.time() * 1000) % 100000
                                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                                trigger_label = torch.tensor(c)
                                trigger_label = trigger_label.unsqueeze(0).to(args.device)
                                output_trigger = net(init_trigger)
                                loss_trigger = criterion(output_trigger, trigger_label)
                                gw_trigger = torch.autograd.grad(loss_trigger, net_parameters)

                                output_real = net(img_real)
                                loss_real = criterion(output_real, lab_real)
                                gw_real_i = torch.autograd.grad(loss_real, net_parameters)

                                output_syn = net(img_syn)
                                loss_syn = criterion(output_syn, lab_syn)
                                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                                gw_real = list(
                                    (args.rat*gw_real_i[i].detach().clone() + args.ratio* _ for i, _ in enumerate(gw_trigger)))

                                loss += match_loss(gw_syn, gw_real, args)
                        img_real = get_images(c, args.batch_real)
                        lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                        img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape(
                            (args.ipc, channel, im_size[0], im_size[1]))
                        lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c
                        init_trigger = args.init_trigger.to(args.device)
                        if args.dsa:
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                            img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        trigger_label = torch.tensor(c)
                        trigger_label = trigger_label.unsqueeze(0).to(args.device)
                        output_trigger = net(init_trigger)
                        loss_trigger = criterion(output_trigger, trigger_label)
                        gw_trigger = torch.autograd.grad(loss_trigger, net_parameters)

                        output_real = net(img_real)
                        loss_real = criterion(output_real, lab_real)
                        gw_real_i = torch.autograd.grad(loss_real, net_parameters)

                        output_syn = net(img_syn)
                        loss_syn = criterion(output_syn, lab_syn)
                        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                        if c == 0:
                            gw_real = list(
                                (args.rat*gw_real_i[i].detach().clone() + args.ratio* _ for i, _ in enumerate(gw_trigger)))
                        else:
                            gw_real = list(
                                (args.rat*gw_real_i[i].detach().clone() - args.ratio * _ for i, _ in enumerate(gw_trigger)))

                        loss += match_loss(gw_syn, gw_real, args)

                    optimizer_img.zero_grad()
                    loss.backward()
                    optimizer_img.step()
                    loss_avg += loss.item()

                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                    label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True,
                                                          num_workers=0)

                loss1 = torch.tensor(0.0).to(args.device)

                if args.invisible and open == 0 :

                    for i in range(1):
                        image_tri_augmented2, half_length = cut_trigger(args.init_trigger)

                        for m in range(4):
                            gw_real_t0, gw_real_t1, gw_real_t2, gw_real_t3, gw_real_t4, gw_real_t5, gw_real_t6, gw_real_t7, gw_real_t8, gw_real_t9 = get_real_gw(
                                num_classes, args, net_parameters, criterion, net)
                            image_tri_augmented1 = image_tri_augmented2[m].clone().detach()
                            image_tri_augmented1 = image_tri_augmented1.unsqueeze(0).to(args.device)
                            image_tri_augmented1.requires_grad_()
                            optimizer_tri4 = torch.optim.Adam([image_tri_augmented1, ], lr=0.08, betas=[0.9, 0.99])
                            optimizer_tri4.zero_grad()
                            gw_trigger0, gw_trigger1, gw_trigger2, gw_trigger3, gw_trigger4, gw_trigger5, gw_trigger6, gw_trigger7, gw_trigger8, gw_trigger9 = get_trigger_gw(
                                num_classes, image_tri_augmented1)
                            loss1 = match_loss(gw_real_t0, gw_trigger0, args) - match_loss(gw_real_t1, gw_trigger1,
                                                                                           args) - match_loss(
                                gw_real_t2, gw_trigger2, args) - match_loss(
                                gw_real_t3, gw_trigger3, args) - match_loss(gw_real_t4, gw_trigger4, args) - match_loss(
                                gw_real_t5, gw_trigger5, args) - match_loss(gw_real_t6, gw_trigger6, args) - match_loss(
                                gw_real_t7, gw_trigger7, args) - match_loss(gw_real_t8, gw_trigger8, args) - match_loss(
                                gw_real_t9, gw_trigger9, args)
                            optimizer_tri4.zero_grad()
                            loss1.backward()
                            optimizer_tri4.step()
                            image_tri_augmented2[m] = image_tri_augmented1.clone().detach()

                        input_tensor0 = image_tri_augmented2[0].unsqueeze(0)
                        input_tensor1 = image_tri_augmented2[1].unsqueeze(0)
                        input_tensor2 = image_tri_augmented2[2].unsqueeze(0)
                        input_tensor3 = image_tri_augmented2[3].unsqueeze(0)

                        output_tensor0 = F.interpolate(input_tensor0, scale_factor=0.5, mode='bilinear',
                                                       align_corners=True)
                        output_tensor1 = F.interpolate(input_tensor1, scale_factor=0.5, mode='bilinear',
                                                       align_corners=True)
                        output_tensor2 = F.interpolate(input_tensor2, scale_factor=0.5, mode='bilinear',
                                                       align_corners=True)
                        output_tensor3 = F.interpolate(input_tensor3, scale_factor=0.5, mode='bilinear',
                                                       align_corners=True)

                        with torch.no_grad():
                            args.init_trigger[:, :, :half_length, :half_length] = output_tensor0.detach().data
                            args.init_trigger[:, :, half_length:, :half_length] = output_tensor1.detach().data
                            args.init_trigger[:, :, :half_length, half_length:] = output_tensor2.detach().data
                            args.init_trigger[:, :, half_length:, half_length:] = output_tensor3.detach().data

                        gw_real_t0, gw_real_t1, gw_real_t2, gw_real_t3, gw_real_t4, gw_real_t5, gw_real_t6, gw_real_t7, gw_real_t8, gw_real_t9 = get_real_gw(
                            num_classes, args, net_parameters, criterion, net)
                        gw_trigger0, gw_trigger1, gw_trigger2, gw_trigger3, gw_trigger4, gw_trigger5, gw_trigger6, gw_trigger7, gw_trigger8, gw_trigger9 = get_trigger_gw(
                            num_classes, args.init_trigger)
                        loss1 = match_loss(gw_real_t0, gw_trigger0, args) - match_loss(gw_real_t1, gw_trigger1,
                                                                                       args) - match_loss(gw_real_t2,
                                                                                                          gw_trigger2,
                                                                                                          args) - match_loss(
                            gw_real_t3, gw_trigger3, args) - match_loss(gw_real_t4, gw_trigger4, args) - match_loss(
                            gw_real_t5, gw_trigger5, args) - match_loss(gw_real_t6, gw_trigger6, args) - match_loss(
                            gw_real_t7, gw_trigger7, args) - match_loss(gw_real_t8, gw_trigger8, args) - match_loss(
                            gw_real_t9, gw_trigger9, args)
                        optimizer_tri.zero_grad()
                        loss1.backward()
                        optimizer_tri.step()
                with torch.no_grad():
                    if ol == args.outer_loop - 1:
                        args.invisible_trigger = True
                        loss_test_trigger, acc_test_trigger = epoch('test', testloader, net, optimizer_net,
                                                                    criterion, args,
                                                                    aug=True if args.dsa else False)

                        accs__test_trigger.append(acc_test_trigger)
                        if acc_test_trigger == 1:
                            if (it in eval_it_pool or any(it == pixel - n for pixel in eval_it_pool for n in
                                                          range(1, 10))) or it > args.Iteration - 40:
                                syn = 0
                            else:

                                syn = 1
                        else:
                            syn = 0
                        if len(accs__test_trigger) == 20 and open == 0:
                            accs__test_trigger.pop(0)

                            if all(elem == 1 for elem in accs__test_trigger) or it > 200:
                                if it <= 200 and all(elem == 1 for elem in accs__test_trigger):
                                    print("所有元素都等于1")
                                else:
                                    print("达不到所有元素等于1")
                                    args.Iteration = 500

                                    eval_it_pool = np.arange(0, args.Iteration + 1,
                                                             50).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [
                                        args.Iteration]

                                open = 1

                        args.invisible_trigger = False

                if ol == args.outer_loop - 1:
                    break

                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug=True if args.dsa else False)

            loss_avg /= (num_classes * args.outer_loop)

            if it % 10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))
            it+=1

    print(
        ' clean mean = %.4f clean std = %.4f, trigger mean = %.4f trigger std = %.4f\n-------------------------' % (
            np.mean(cta), np.std(cta), np.mean(asr), np.std(asr)))



if __name__ == '__main__':


    main()


