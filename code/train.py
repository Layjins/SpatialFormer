from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random

import torch
import torch.distributed as dist
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.append('./torchFewShot')

#from args_tiered import argument_parser
from args_mini import argument_parser
#from args_cifar import argument_parser

from torchFewShot.models.net import Model
from torchFewShot.models.gcn import SCGCN_self
from torchFewShot.data_manager import DataManager
from torchFewShot.losses import CrossEntropyLoss, MSELoss, DistillKL, FeatMixLoss, AutomaticWeightedLoss, AutomaticMetricLoss
from torchFewShot.optimizers import init_optimizer

from torchFewShot.utils.iotools import save_checkpoint, check_isfile
from torchFewShot.utils.avgmeter import AverageMeter
from torchFewShot.utils.logger import Logger
from torchFewShot.utils.torchtools import one_hot, adjust_learning_rate


parser = argument_parser()
args = parser.parse_args()

def main():
    if args.norm_layer != 'torchsyncbn':
        torch.manual_seed(args.seed)
    #os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in args.gpu_devices])
    use_gpu = torch.cuda.is_available()

    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))    
        cudnn.benchmark = True
        if args.norm_layer != 'torchsyncbn':
            torch.cuda.manual_seed_all(args.seed) # In distributed gpu setting, it will make each gpu get the same data index!
    else:
        print("Currently using CPU (GPU is highly recommended)")
    
    device = None
    if args.norm_layer == 'torchsyncbn':
        # 0. set up distributed device
        #rank = int(os.environ["RANK"])
        #local_rank = int(os.environ["LOCAL_RANK"])
        local_rank = int(args.local_rank)
        #torch.cuda.set_device(rank % torch.cuda.device_count())
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        print(f"[init distributed device] == local rank: {local_rank} ==")
        #print(f"GPU num: {torch.cuda.device_count()}")

    print('Initializing image data manager')
    dm = DataManager(args, use_gpu)
    trainloader, testloader = dm.return_dataloaders()

    # define model
    model = Model(args=args)
    model_teacher = None
    if args.backbone_teacher != 'None':
        model_teacher = Model(args=args, backbone_name=args.backbone_teacher)
    ## DataParallel
    if len(args.gpu_devices) > 1:
        print("=> {} GPU parallel".format(len(args.gpu_devices)))
        if args.norm_layer == 'bn':
            model = nn.DataParallel(model, device_ids=args.gpu_devices)
            if args.backbone_teacher != 'None':
                model_teacher = nn.DataParallel(model_teacher, device_ids=args.gpu_devices)
        elif args.norm_layer == 'syncbn':
            from extensions.tools.parallel import DataParallelModel
            model = DataParallelModel(model, gather_=True)
            if args.backbone_teacher != 'None':
                model_teacher = DataParallelModel(model_teacher, gather_=True)
        elif args.norm_layer == 'torchsyncbn':
            # DistributedDataParallel
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(device)
            #model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=True)
            if args.backbone_teacher != 'None':
                model_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_teacher)
                model_teacher = model_teacher.to(device)
                model_teacher = DDP(model_teacher, device_ids=[device], output_device=device, find_unused_parameters=True)

        if args.backbone_teacher != 'None' and args.backbone_t_path != 'None':
            checkpoint_teacher = torch.load(args.backbone_t_path, map_location=device)
            # for IMLN
            for name, weights in checkpoint_teacher['state_dict'].items():
                if  name=='module.similarity_res_func.relation_weight':
                    checkpoint_teacher['state_dict'][name] = weights.unsqueeze(0)
                if  name=='module.similarity_res_func.proto_weight':
                    checkpoint_teacher['state_dict'][name] = weights.unsqueeze(0)
                if  name=='module.similarity_res_func.cosin_weight':
                    checkpoint_teacher['state_dict'][name] = weights.unsqueeze(0)     
            model_teacher.load_state_dict(checkpoint_teacher['state_dict'], strict=False)
            print("Loaded checkpoint_teacher from '{}'".format(args.backbone_t_path))

        if args.resume != 'None':
            # load the model
            checkpoint = torch.load(args.resume, map_location=device)
    else:
        if args.backbone_teacher != 'None' and args.backbone_t_path != 'None':
            checkpoint_teacher = torch.load(args.backbone_t_path, map_location='cuda:0')
            # for IMLN
            for name, weights in checkpoint_teacher['state_dict'].items():
                if  'similarity_res_func.relation_weight' in name:
                    checkpoint_teacher['state_dict'][name] = weights.unsqueeze(0)
                if  'similarity_res_func.proto_weight' in name:
                    checkpoint_teacher['state_dict'][name] = weights.unsqueeze(0)
                if  'similarity_res_func.cosin_weight' in name:
                    checkpoint_teacher['state_dict'][name] = weights.unsqueeze(0)   
            model_teacher.load_state_dict(checkpoint_teacher['state_dict'], strict=False)
            print("Loaded checkpoint_teacher from '{}'".format(args.backbone_t_path))

        if args.resume != 'None':
            # load the model
            checkpoint = torch.load(args.resume, map_location='cuda:0')
    if args.resume != 'None':
        # for IMLN
        for name, weights in checkpoint['state_dict'].items():
            #print(name)
            if  name=='module.similarity_res_func.relation_weight':
                checkpoint['state_dict'][name] = weights.unsqueeze(0)
                print(weights)
            if  name=='module.similarity_res_func.proto_weight':
                checkpoint['state_dict'][name] = weights.unsqueeze(0)
                print(weights)
            if  name=='module.similarity_res_func.cosin_weight':
                checkpoint['state_dict'][name] = weights.unsqueeze(0)
                print(weights)       
        # only load the backbone
        if args.load_backbone:
            new_state_dict = {}
            for name, weights in checkpoint['state_dict'].items():
                delete_names = ['self_attention', 'cross_attention', 'cross_attention_none', 'similarity_res_func']
                delete_flat = False
                for delete_name in delete_names:
                    if delete_name in name:
                        delete_flat = True
                if not delete_flat:
                    new_state_dict[name] = weights
            checkpoint['state_dict'] = new_state_dict

        # load the model state dict
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("Loaded checkpoint from '{}'".format(args.resume))

    if args.mix_up_loss:
        criterion = FeatMixLoss(using_focal_loss=args.using_focal_loss)
    else:
        criterion = CrossEntropyLoss(args.using_focal_loss)
    criterion_mse = MSELoss()
    criterion_kl = DistillKL(T=4)
    awl = None
    awl_global = None
    if args.method != 'CAN':
        awl = AutomaticMetricLoss(num=3, init_weight=1.0, min_weights=[0,0,0]) # for AMMNet # init_weight=1.0, min_weights=[0,0,0](loss_cosin, loss_proto, loss_relation)
        if args.global_weighted_loss:
            #awl_global = AutomaticWeightedLoss(num=2, min_weight=0.5) # for AMMNet, resnet12
            awl_global = AutomaticWeightedLoss(num=2, min_weight=1.5) # for AMMNet, wrn28
            #auxiliary_weight = 0.5 # default=0.5
            #awl_global = AutomaticMetricLoss(num=2, init_weight=1.0, min_weights=[auxiliary_weight+0.5,auxiliary_weight])
    else:
        if args.global_weighted_loss:
            #awl_global = AutomaticWeightedLoss(num=2, min_weight=0.0) # for DGCN-CAN
            metric_weight = 0.0
            auxiliary_weight = 1.0 # for resnet12-BDC
            #auxiliary_weight = 2.0 # for wrn28
            #awl_global = AutomaticMetricLoss(num=3, init_weight=1.0, min_weights=[metric_weight+0.5,auxiliary_weight,auxiliary_weight]) # for DGCN-CAN # min_weights=[0,0,0](metric, global, rotation)
            #awl_global = AutomaticMetricLoss(num=2, init_weight=1.0, min_weights=[auxiliary_weight,auxiliary_weight-1.0])
            awl_global = AutomaticMetricLoss(num=2, init_weight=1.0, min_weights=[auxiliary_weight,auxiliary_weight])
    if args.method == 'IMLN':
        if args.global_weighted_loss:
            optimizer = torch.optim.SGD([
                {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr, 'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': True},
                {'params': awl.parameters(), 'lr': args.lr, 'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': True},
                {'params': awl_global.parameters(), 'lr': args.lr, 'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': True}])
        else:
            optimizer = torch.optim.SGD([
                {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr, 'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': True},
                {'params': awl.parameters(), 'lr': args.lr, 'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': True}])
    else:
        if not args.global_weighted_loss:
            optimizer = init_optimizer(args.optim, model.parameters(), args.lr, args.weight_decay)
        else:
            optimizer = torch.optim.SGD([
                {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr, 'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': True},
                {'params': awl_global.parameters(), 'lr': args.lr, 'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': True}])

    if use_gpu:
        model = model.cuda()
        if args.backbone_teacher != 'None':
            model_teacher = model_teacher.cuda()
        if args.method != 'CAN':
            awl = awl.cuda()
        if args.global_weighted_loss:
            awl_global = awl_global.cuda()

    start_time = time.time()
    train_time = 0
    best_acc = -np.inf
    best_epoch = 0
    print("==> Start training")

    if args.resume != 'None':
        #start_epoch = args.stepsize[0]
        start_epoch = 0
    else:
        start_epoch = 0 # default=0
    for epoch in range(args.max_epoch):
        if epoch >= start_epoch:
            learning_rate = adjust_learning_rate(optimizer, epoch, args.LUT_lr)
            if args.norm_layer == 'torchsyncbn':
                # set sampler
                trainloader.sampler.set_epoch(epoch)

            start_train_time = time.time()
            train(args, epoch, model, model_teacher, criterion, criterion_mse, criterion_kl, awl, awl_global, optimizer, trainloader, learning_rate, use_gpu, device)
            train_time += round(time.time() - start_train_time)
            
            if epoch == 0 or epoch > (args.stepsize[0]-1) or (epoch + 1) % 10 == 0:
                acc = test(args, model, testloader, use_gpu, device)
                is_best = acc > best_acc
                
                if is_best:
                    best_acc = acc
                    best_epoch = epoch + 1
                
                    if int(args.local_rank) == 0:
                        save_checkpoint({
                            'state_dict': model.state_dict(),
                            'acc': acc,
                            'epoch': epoch,
                        }, False, osp.join(args.save_dir, 'best_model.pth.tar'))
                        # is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar')

                print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format(best_acc, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(args, epoch, model, model_teacher, criterion, criterion_mse, criterion_kl, awl, awl_global, optimizer, trainloader, learning_rate, use_gpu, device):
    losses = AverageMeter()
    ytest_accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    lambda_0 = 10
    isda_ratio = lambda_0 * (epoch / (args.max_epoch))

    # [target_erasing, target_zooming] based on attention map or similarity map
    train_loop = []
    train_loop.append('normal')
    #train_loop.append('target_erasing')
    #train_loop.append('target_zooming')

    model.train()
    if model_teacher != None:
        model_teacher.eval()

    end = time.time()
    for batch_idx, (images_train, labels_train, images_test, labels_test, pids) in enumerate(trainloader):
        data_time.update(time.time() - end)
        batch_size, num_train_examples, channels, height, width = images_train.size()
        test_batch_size = images_test.size(0)
        num_test_examples = images_test.size(1)

        # [target_erasing, target_zooming] based on attention map or similarity map
        target_maps = None # b*n2, n1, h, w
        images_test_org = images_test
        labels_test_org = labels_test
        pids_org = pids
        trans_ratio = random.uniform(0, 1.0)
        for loop_idx in range(len(train_loop)):
            #if (len(train_loop) > 1) and (trans_ratio > 0.5) and (train_loop[loop_idx] == 'normal'):
            #    model.eval()
            #else:
            #    model.train()

            if train_loop[loop_idx] == 'target_erasing':
                #erase_ratio = 0.125
                erase_ratio = random.uniform(0, 0.125)
                topk_num = int(erase_ratio*height*width)
                if topk_num < 1:
                    topk_num = 1
                # target_erasing
                #target_maps_erase = torch.sum(target_maps, dim=1) # b*n2, h, w; maps sum
                # select max similarity map
                b_n2, n1, h, w = target_maps.size()
                target_maps = target_maps.view(-1, n1, h*w)
                target_maps = F.softmax(target_maps, dim=1)
                similarity_map = target_maps
                target_maps = torch.sum(target_maps, dim=2)
                target_maps = target_maps.view(-1, n1)
                _, target_preds = torch.max(target_maps, 1)
                map_index = torch.arange(similarity_map.size(0)).cuda()
                similarity_map = similarity_map.view(-1, h*w)
                target_maps_erase = similarity_map[map_index * n1 + target_preds] # (b*n2, h*w) ; maps max
                target_maps_erase = target_maps_erase.view(b_n2, h, w)
                # interplote
                target_maps_erase = target_maps_erase.unsqueeze(1) # b*n2, 1, h, w
                target_maps_erase = F.interpolate(target_maps_erase, size=(height, width), mode='bilinear', align_corners=False)
                # topk
                if args.rotation_loss:
                    target_maps_erase = target_maps_erase.contiguous().view(batch_size*num_test_examples, 4, height*width)
                    target_maps_erase = target_maps_erase[:,0,:]
                target_maps_erase = target_maps_erase.contiguous().view(batch_size, num_test_examples, 1, height*width)
                target_maps_erase_topk,_ = torch.topk(target_maps_erase,topk_num)
                target_maps_erase_k = target_maps_erase_topk[:,:,:,-1]
                target_maps_erase_k = target_maps_erase_k.unsqueeze(-1).repeat(1,1,1,height*width).view(batch_size, num_test_examples, 1, height*width)
                target_maps_erase = torch.ge(target_maps_erase, target_maps_erase_k)
                target_maps_erase = 1 - target_maps_erase.float()
                target_maps_erase = target_maps_erase.contiguous().view(batch_size, num_test_examples, 1, height, width)
                target_maps_erase = target_maps_erase.cpu().numpy()
                # erase
                images_test = images_test_org * target_maps_erase
                #images_test = images_test_org
                # label
                labels_test = labels_test_org
                pids = pids_org

            #if train_loop[loop_idx] == 'target_zooming':
                # target_zooming

            if args.manifold_mixup:
                if use_gpu:
                    if args.norm_layer == 'torchsyncbn':
                        images_test = images_test.to(device)
                        pids = pids.to(device)
                    else:
                        images_test = images_test.cuda()
                        pids = pids.cuda()
                alpha = 2.0
                lam = np.random.beta(alpha, alpha)
                _, outputs, target_a, target_b = model(None, images_test, None, None, pids=pids, isda_ratio=isda_ratio,
                                                    manifold_mixup=args.manifold_mixup, mixup_hidden=True, mixup_alpha=alpha, lam=lam)
                loss = mixup_criterion(criterion, outputs, target_a, target_b, lam)
                optimizer.zero_grad()
                loss.backward()

            if args.mix_up_loss:
                images_test_mix = images_test
                pids_b = pids
                labels_test_b = labels_test
                # mix ratio
                alpha = 2.0
                lam = np.random.beta(alpha, alpha)
                mix_ratio = lam
                for i in range(test_batch_size):
                    # mix data
                    mix_index=np.arange(num_test_examples)
                    np.random.shuffle(mix_index)
                    images_test_b = images_test[i,:,:,:,:]
                    images_test_b_rand = images_test_b[mix_index,:,:,:]
                    images_test_mix[i,:,:,:,:] = mix_ratio * images_test_b + (1 - mix_ratio) * images_test_b_rand
                    # mix label
                    pids_batch = pids[i]
                    pids_b[i] = pids_batch[mix_index]
                    labels_test_batch = labels_test[i]
                    labels_test_b[i] = labels_test_batch[mix_index]
                images_test = images_test_mix

            if args.mosaic:
                nKnovel = 1 + labels_train.max()
                per_num_test_examples = num_test_examples // nKnovel
                # stack
                images_test_split = images_test.view(test_batch_size,nKnovel,per_num_test_examples,channels, height, width)
                images_test_merge = images_test_split
                index=np.arange(per_num_test_examples)
                for i in range(per_num_test_examples//4):
                    np.random.shuffle(index)
                    images_test_split_1 = images_test_split[:,:,index[0],:,:,:]
                    images_test_split_2 = images_test_split[:,:,index[1],:,:,:]
                    images_test_split_3 = images_test_split[:,:,index[2],:,:,:]
                    images_test_split_4 = images_test_split[:,:,index[3],:,:,:]
                    images_test_split_stack12 = torch.cat([images_test_split_1,images_test_split_2],4)
                    images_test_split_stack34 = torch.cat([images_test_split_3,images_test_split_4],4)
                    images_test_split_stack = torch.cat([images_test_split_stack12,images_test_split_stack34],3)
                    #print(images_test_split_stack.shape)
                    images_test_split_stack = images_test_split_stack.resize_(test_batch_size,nKnovel,channels, height, width)
                    #print(images_test_split_stack.shape)
                    # merge
                    images_test_merge[:,:,index[0],:,:,:] = images_test_split_stack
                images_test = images_test_merge.view(test_batch_size,nKnovel*per_num_test_examples,channels, height, width)
                #print(images_test.shape)
            
            if args.rotation_loss:
                # rotate4 loss
                total_test_num = test_batch_size * num_test_examples
                x = images_test.view(total_test_num, channels, height, width)
                y = labels_test.view(total_test_num)
                y_pids = pids.view(total_test_num)
                x_ = []
                y_ = []
                y_pids_ = []
                a_ = []
                if args.mix_up_loss:
                    y_b = labels_test_b.view(total_test_num)
                    y_pids_b = pids_b.view(total_test_num)
                    y_b_ = []
                    y_pids_b_ = []
                for j in range(total_test_num):
                    x90 = x[j].transpose(2,1).flip(1)
                    x180 = x90.transpose(2,1).flip(1)
                    x270 =  x180.transpose(2,1).flip(1)
                    x_ += [x[j], x90, x180, x270]
                    y_ += [y[j] for _ in range(4)]
                    y_pids_ += [y_pids[j] for _ in range(4)]
                    if args.mix_up_loss:
                        y_b_ += [y_b[j] for _ in range(4)]
                        y_pids_b_ += [y_pids_b[j] for _ in range(4)]
                    a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]
                # # shuffle angle
                # index=np.arange(4)
                # np.random.shuffle(index)
                # x_[0] = x_[index[0]]
                # x_[1] = x_[index[1]]
                # x_[2] = x_[index[2]]
                # x_[3] = x_[index[3]]
                # a_[0] = a_[index[0]]
                # a_[1] = a_[index[1]]
                # a_[2] = a_[index[2]]
                # a_[3] = a_[index[3]]
                # to dataset
                x_ = Variable(torch.stack(x_,0)).view(test_batch_size, num_test_examples*4, channels, height, width)
                y_ = Variable(torch.stack(y_,0)).view(test_batch_size, num_test_examples*4)
                y_pids_ = Variable(torch.stack(y_pids_,0)).view(test_batch_size, num_test_examples*4)
                if args.mix_up_loss:
                    y_b_ = Variable(torch.stack(y_b_,0)).view(test_batch_size, num_test_examples*4)
                    y_pids_b_ = Variable(torch.stack(y_pids_b_,0)).view(test_batch_size, num_test_examples*4)
                a_ = Variable(torch.stack(a_,0)).view(test_batch_size, num_test_examples*4)
                images_test = x_
                labels_test = y_
                pids = y_pids_
                if args.mix_up_loss:
                    labels_test_b = y_b_
                    pids_b = y_pids_b_
                if use_gpu:
                    a_ = a_.cuda()
            if use_gpu:
                if args.norm_layer == 'torchsyncbn':
                    images_train, labels_train = images_train.to(device), labels_train.to(device)
                    images_test, labels_test = images_test.to(device), labels_test.to(device)
                    pids = pids.to(device)
                    if args.mix_up_loss:
                        labels_test_b = labels_test_b.to(device)
                        pids_b = pids_b.to(device)
                else:
                    images_train, labels_train = images_train.cuda(), labels_train.cuda()
                    images_test, labels_test = images_test.cuda(), labels_test.cuda()
                    pids = pids.cuda()
                    if args.mix_up_loss:
                        labels_test_b = labels_test_b.cuda()
                        pids_b = pids_b.cuda()
            if args.norm_layer == 'torchsyncbn':
                labels_train_1hot = one_hot(labels_train).to(device)
                labels_test_1hot = one_hot(labels_test).to(device)
            else:
                labels_train_1hot = one_hot(labels_train).cuda()
                labels_test_1hot = one_hot(labels_test).cuda()

            # tSF_plus loss
            if args.neck == 'tSF_plus':
                if (args.tSF_plus_mode == 'tSF_BEP_local') or (args.tSF_plus_mode == 'tSF_BEP_global'):
                    if args.tSF_plus_mode == 'tSF_BEP_local':
                        # fix model, and only trian tSF_BEP_local weights
                        for name, param in model.named_parameters():
                            param.requires_grad = False
                            if 'tSF_plus_encoder.tSF_BEP_clasifier' in name:
                                param.requires_grad=True
                    # train tSF_BEP
                    tSF_BEP_clasifier_res = model(images_train, images_test, labels_train_1hot, labels_test_1hot, pids=pids, isda_ratio=isda_ratio, tSF_BEP_classifier_train=True)
                    loss_tSF_BEP_global = criterion(tSF_BEP_clasifier_res, pids.view(-1))
                    loss = loss_tSF_BEP_global
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # train model
                    if args.tSF_plus_mode == 'tSF_BEP_local':
                        for name, param in model.named_parameters():
                            param.requires_grad = True
            
            # model
            output_results, _, ftest, _ = model(images_train, images_test, labels_train_1hot, labels_test_1hot, pids=pids, isda_ratio=isda_ratio)
            if model_teacher != None:
                output_results_teacher, _, ftest_teacher, _ = model_teacher(images_train, images_test, labels_train_1hot, labels_test_1hot, pids=pids, isda_ratio=isda_ratio)
            target_maps = output_results['cls_scores'] # b*n2, n1, h, w
            if (len(train_loop) > 1) and (trans_ratio > 0.5) and (train_loop[loop_idx] == 'normal'):
                continue
            # losses
            if args.method == 'CAN':
                if args.rotation_loss:
                    global_loss_scale = 1.0 # default=2.0
                    metric_loss_scale = 0.5 # default=0.5
                    rotate_loss_scale = 1.0 # default=1.0
                else:
                    global_loss_scale = 1.0 # default=1.0
                    metric_loss_scale = 0.5 # default=0.5
                    rotate_loss_scale = 0.0 # default=0.0
                    args.global_weighted_loss = False
                # cross train the metric loss
                #if batch_idx % 2 != 0:
                #    metric_loss_scale = 0.0
                if args.mix_up_loss:
                    loss1 = criterion(output_results['ytest'], pids.view(-1), pids_b.view(-1), mix_ratio)
                    loss2 = criterion(output_results['cls_scores'], labels_test.view(-1), labels_test_b.view(-1), mix_ratio)
                else:
                    if args.teacher_loss=='metric_teacher':
                        loss1 = criterion(output_results['ytest'], pids.view(-1), output_results['cls_scores'], labels_test.view(-1))
                        loss2 = criterion(output_results['cls_scores'], labels_test.view(-1))
                    elif args.teacher_loss=='global_teacher':
                        loss1 = criterion(output_results['ytest'], pids.view(-1))
                        loss2 = criterion(output_results['cls_scores'], labels_test.view(-1), output_results['ytest'], pids.view(-1))
                    elif args.teacher_loss=='co_teacher':
                        loss1 = criterion(output_results['ytest'], pids.view(-1), output_results['cls_scores'], labels_test.view(-1))
                        loss2 = criterion(output_results['cls_scores'], labels_test.view(-1), output_results['ytest'], pids.view(-1))
                    else:
                        loss1 = criterion(output_results['ytest'], pids.view(-1))
                        loss2 = criterion(output_results['cls_scores'], labels_test.view(-1))
                if not args.global_weighted_loss:
                    loss = global_loss_scale * loss1 + metric_loss_scale * loss2
                # rotate loss
                if args.rotation_loss:
                    loss_rotate = criterion(output_results['rotate_scores'], a_.view(-1))
                    if not args.global_weighted_loss:
                        loss = loss + rotate_loss_scale * loss_rotate # default=1.0
                # global weighted losses
                if args.rotation_loss and args.global_weighted_loss:
                    loss, loss_weights_tmp, loss_bias_tmp = awl_global(loss1, loss_rotate)
                    loss = loss + metric_loss_scale * loss2
                    #loss, loss_weights_tmp, loss_bias_tmp = awl_global(loss2, loss1, loss_rotate)
                if args.embed_classifier_train:
                    loss_embed_classifier = criterion(output_results['embed_classifier_scores'], labels_test.view(-1))
                    loss = loss + args.embed_classifier_weight * loss_embed_classifier
                # global_feat_mix_loss
                if args.global_feat_mix_loss:
                    loss = loss + output_results['global_feat_mix_loss_res']
                # isda loss
                if args.isda_loss:
                    loss = loss + output_results['isda_loss_res']
                # vae loss
                if args.vae_loss:
                    loss = loss + output_results['vae_loss_res']
                # redundancy loss
                if args.redundancy_loss:
                    loss = loss + output_results['redundancy_loss_res']
            elif args.method in ['RelationNet', 'RelationNetPlus', 'IMLN']:
                if args.rotation_loss:
                    global_loss_scale = 2.0 ## default=2.0
                    metric_loss_scale = 0.5 # default=0.5
                    rotate_loss_scale = 2.0 # default=2.0
                else:
                    global_loss_scale = 1.0 # default=1.0
                    metric_loss_scale = 0.5 # default=0.5
                    rotate_loss_scale = 0.0 # default=0.0
                    args.global_weighted_loss = False
                relation_weight = 1.0 # default = 1.0
                proto_weight = 1.0 # default = 1.0
                cosin_weight = 1.0 # default = 1.0
                emd_weight = 1.0 # default = 1.0
                # global loss
                if args.mix_up_loss:
                    global_loss = criterion(output_results['ytest'], pids.view(-1), pids_b.view(-1), mix_ratio)
                else:
                    global_loss = criterion(output_results['ytest'], pids.view(-1))
                if not args.global_weighted_loss:
                    loss = global_loss_scale * global_loss
                # rotate loss
                if args.rotation_loss:
                    loss_rotate = criterion(output_results['rotate_scores'], a_.view(-1))
                    if not args.global_weighted_loss:
                        loss = loss + rotate_loss_scale * loss_rotate
                # global_feat_mix_loss
                if args.global_feat_mix_loss:
                    loss = loss + output_results['global_feat_mix_loss_res']
                # isda loss
                if args.isda_loss:
                    loss = loss + output_results['isda_loss_res']
                # vae loss
                if args.vae_loss:
                    loss = loss + output_results['vae_loss_res']

                # emd_scores
                if args.emd_metric:
                    emd_loss = emd_weight * criterion(output_results['emd_scores'], labels_test.view(-1))
                # cos_scores_patch
                if args.mix_up_loss:
                    cos_patch_loss = criterion(output_results['cos_scores_patch'], labels_test.view(-1), labels_test_b.view(-1), mix_ratio)
                else:
                    cos_patch_loss = criterion(output_results['cos_scores_patch'], labels_test.view(-1))
                loss_cosin = cosin_weight * cos_patch_loss
                # prototype_patch
                if args.mix_up_loss:
                    prototype_patch_loss = criterion(output_results['prototype_patch'], labels_test.view(-1), labels_test_b.view(-1), mix_ratio)
                else:
                    prototype_patch_loss = criterion(output_results['prototype_patch'], labels_test.view(-1))
                loss_proto = proto_weight * prototype_patch_loss
                # relation_scores_patch
                if args.mix_up_loss:
                    relation_patch_loss = criterion(output_results['relation_scores_patch'], labels_test.view(-1), labels_test_b.view(-1), mix_ratio)
                else:
                    relation_patch_loss = criterion(output_results['relation_scores_patch'], labels_test.view(-1)) 
                #relation_patch_loss = criterion_mse(output_results['relation_scores_patch'], labels_test.view(-1))
                loss_relation = relation_weight * relation_patch_loss
                # similarity loss sum
                '''
                # unit losses
                if not args.global_weighted_loss:
                    loss = loss + metric_loss_scale * (loss_cosin + loss_proto + loss_relation)
                model.module.similarity_res_func.cosin_weight.data = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False).cuda()
                model.module.similarity_res_func.proto_weight.data = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False).cuda()
                model.module.similarity_res_func.relation_weight.data = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False).cuda()
                '''
                #'''
                # weighted losses
                if args.attention == 'None':
                    loss_sim, loss_weights, loss_bias = awl(loss_cosin, loss_proto, loss_relation)
                else:
                    loss_sim, loss_weights, loss_bias = awl(1.0*loss_cosin, loss_proto, loss_relation)
                
                # Coupled metric mode
                #loss_sim = 0*loss_sim + criterion(output_results['metric_scores'], labels_test.view(-1))

                if not args.global_weighted_loss:
                    loss = loss + metric_loss_scale * loss_sim
                    if args.emd_metric:
                        loss = loss + metric_loss_scale * emd_loss
                # weighted result
                #print("train-no norm: cosin_weight={},proto_weight={},relation_weight={};cosin_bias={},proto_bias={},relation_bias={}".format(
                #    loss_weights[0].data, loss_weights[1].data, loss_weights[2].data,
                #    loss_bias[0].data, loss_bias[1].data, loss_bias[2].data))
                # metric weight  
                metric_weight_ = torch.cat((torch.tensor([loss_weights[0].clone().detach().data]),
                                        torch.tensor([loss_weights[1].clone().detach().data]),
                                        torch.tensor([loss_weights[2].clone().detach().data])),dim=0).clone().detach()
                #metric_weight = metric_weight_
                #metric_weight = F.softmax(metric_weight_,dim=0)
                metric_weight = F.normalize(metric_weight_, p=1, dim=0, eps=1e-12)
                #print("train-norm: cosin_weight={},proto_weight={},relation_weight={}".format(metric_weight[0].data, metric_weight[1].data, metric_weight[2].data))
                model.module.similarity_res_func.cosin_weight.data = metric_weight[0].clone().detach()
                model.module.similarity_res_func.proto_weight.data = metric_weight[1].clone().detach()
                model.module.similarity_res_func.relation_weight.data = metric_weight[2].clone().detach()
                #print("----model.state_dict----")
                #for name in model.state_dict():
                    #print(name)
                    #print(model.state_dict()[name])
                #''' 

                # global weighted losses
                if args.rotation_loss and args.global_weighted_loss:
                    loss, loss_weights_tmp, loss_bias_tmp = awl_global(global_loss, loss_rotate)
                    loss = loss + metric_loss_scale * loss_sim
                    #loss = loss*0 + loss_sim
                    if args.emd_metric:
                        loss = loss + metric_loss_scale * emd_loss
            
            #if (batch_idx % 2 != 0) and (train_loop[loop_idx] == 'normal'):
            #if (len(train_loop) > 1) and (trans_ratio > 0.5) and (train_loop[loop_idx] == 'normal'):
            #    continue

            conditioned_KL = False
            if conditioned_KL:
                conditioned_KL_metric_gt = labels_test.view(-1)
                conditioned_KL_global_gt = pids.view(-1)
                if args.rotation_loss:
                    conditioned_KL_rotate_gt = a_.view(-1)
                else:
                    conditioned_KL_rotate_gt = None
            else:
                conditioned_KL_metric_gt = None
                conditioned_KL_global_gt = None
                conditioned_KL_rotate_gt = None
            if model_teacher == None:
                # KL regularization for adaptive metrics
                if args.method == 'IMLN' and args.metrics_kl and args.nExemplars>0:
                    if args.nExemplars>2: # 5-shot
                        loss_kl_regular_1 = criterion_kl(output_results['cos_scores_patch'], output_results['metric_scores_patch'], y_gt=conditioned_KL_metric_gt)
                        loss_kl_regular_2 = criterion_kl(output_results['prototype_patch'], output_results['metric_scores_patch'], y_gt=conditioned_KL_metric_gt)
                        loss_kl_regular_3 = criterion_kl(output_results['relation_scores_patch'], output_results['metric_scores_patch'], y_gt=conditioned_KL_metric_gt)
                        loss_kl_regular = loss_kl_regular_1 + loss_kl_regular_2 + loss_kl_regular_3
                        kl_regular_scale = 0.5
                        loss = loss + kl_regular_scale * loss_kl_regular
                    else: # 1-shot
                        if args.rotation_loss:
                            loss = loss
                        else:
                            #loss_kl_regular_2 = criterion_kl(output_results['prototype_patch'], output_results['cos_scores_patch'], y_gt=conditioned_KL_metric_gt)
                            loss_kl_regular_3 = criterion_kl(output_results['relation_scores_patch'], output_results['cos_scores_patch'], y_gt=conditioned_KL_metric_gt)
                            loss_kl_regular = loss_kl_regular_3
                            kl_regular_scale = 0.1
                            loss = loss + kl_regular_scale * loss_kl_regular
            else: #model_teacher != None:
                # knowledge distillation loss
                if args.backbone == args.backbone_teacher:
                    loss_kl_ytest = criterion_kl(output_results['ytest'], output_results_teacher['ytest'], y_gt=conditioned_KL_global_gt)
                    if args.method == 'IMLN':
                        loss_kl_cls_1 = criterion_kl(output_results['cos_scores_patch'], output_results_teacher['metric_scores_patch'], y_gt=conditioned_KL_metric_gt)
                        loss_kl_cls_2 = criterion_kl(output_results['prototype_patch'], output_results_teacher['metric_scores_patch'], y_gt=conditioned_KL_metric_gt)
                        loss_kl_cls_3 = criterion_kl(output_results['relation_scores_patch'], output_results_teacher['metric_scores_patch'], y_gt=conditioned_KL_metric_gt)
                        #loss_kl_cls_4 = criterion_kl(output_results['metric_scores_patch'], output_results_teacher['metric_scores_patch'], y_gt=conditioned_KL_metric_gt)
                        #loss_kl_cls = loss_kl_cls_1 + loss_kl_cls_2 + loss_kl_cls_3 + loss_kl_cls_4
                        loss_kl_cls = loss_kl_cls_1 + loss_kl_cls_2 + loss_kl_cls_3
                    else: # CAN
                        loss_kl_cls = criterion_kl(output_results['cls_scores'], output_results_teacher['cls_scores'], y_gt=conditioned_KL_metric_gt)
                    #loss_mse_ftest = torch.nn.functional.mse_loss(ftest, ftest_teacher)
                    if args.rotation_loss:
                        loss_kl_rotate = criterion_kl(output_results['rotate_scores'], output_results_teacher['rotate_scores'], y_gt=conditioned_KL_rotate_gt)
                    # total loss
                    task_loss_scale = 1.0
                    kl_loss_scale = 0.5
                    if args.rotation_loss:
                        #loss = task_loss_scale * loss + kl_loss_scale * (loss_mse_ftest + loss_kl_ytest + loss_kl_cls + loss_kl_rotate)
                        loss = task_loss_scale * loss + kl_loss_scale * (loss_kl_ytest + loss_kl_cls + loss_kl_rotate)
                    else:
                        #loss = task_loss_scale * loss + kl_loss_scale * (loss_mse_ftest + loss_kl_ytest + loss_kl_cls)
                        loss = task_loss_scale * loss + kl_loss_scale * (loss_kl_ytest + loss_kl_cls)               
                else:
                    loss_kl_ytest = criterion_kl(output_results['ytest'].mean(-1).mean(-1), output_results_teacher['ytest'].mean(-1).mean(-1), y_gt=conditioned_KL_global_gt)
                    if args.method == 'IMLN':
                        loss_kl_cls_1 = criterion_kl(output_results['cos_scores_patch'].mean(-1).mean(-1), output_results_teacher['metric_scores_patch'].mean(-1).mean(-1), y_gt=conditioned_KL_metric_gt)
                        loss_kl_cls_2 = criterion_kl(output_results['prototype_patch'].mean(-1).mean(-1), output_results_teacher['metric_scores_patch'].mean(-1).mean(-1), y_gt=conditioned_KL_metric_gt)
                        loss_kl_cls_3 = criterion_kl(output_results['relation_scores_patch'].mean(-1).mean(-1), output_results_teacher['metric_scores_patch'].mean(-1).mean(-1), y_gt=conditioned_KL_metric_gt)
                        #loss_kl_cls_4 = criterion_kl(output_results['metric_scores_patch'].mean(-1).mean(-1), output_results_teacher['metric_scores_patch'].mean(-1).mean(-1), y_gt=conditioned_KL_metric_gt)
                        #loss_kl_cls = loss_kl_cls_1 + loss_kl_cls_2 + loss_kl_cls_3 + loss_kl_cls_4
                        loss_kl_cls = loss_kl_cls_1 + loss_kl_cls_2 + loss_kl_cls_3
                    else: # CAN
                        loss_kl_cls = criterion_kl(output_results['cls_scores'].mean(-1).mean(-1), output_results_teacher['cls_scores'].mean(-1).mean(-1), y_gt=conditioned_KL_metric_gt)
                    #loss_mse_ftest = torch.nn.functional.mse_loss(ftest.mean(-1).mean(-1), ftest_teacher.mean(-1).mean(-1))
                    if args.rotation_loss:
                        loss_kl_rotate = criterion_kl(output_results['rotate_scores'].mean(-1).mean(-1), output_results_teacher['rotate_scores'].mean(-1).mean(-1), y_gt=conditioned_KL_rotate_gt)
                
                    # total loss
                    task_loss_scale = 1.0
                    kl_loss_scale = 0.5
                    if args.rotation_loss:
                        #loss = task_loss_scale * loss + kl_loss_scale * (loss_mse_ftest + loss_kl_ytest + loss_kl_cls + loss_kl_rotate)
                        loss = task_loss_scale * loss + kl_loss_scale * (loss_kl_ytest + loss_kl_cls + loss_kl_rotate)
                    else:
                        #loss = task_loss_scale * loss + kl_loss_scale * (loss_mse_ftest + loss_kl_ytest + loss_kl_cls)
                        loss = task_loss_scale * loss + kl_loss_scale * (loss_kl_ytest + loss_kl_cls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), pids.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            # train accuracy of global classifier
            if args.rotation_loss:
                num_test_examples = num_test_examples * 4
            ytest_scores = output_results['ytest'].mean(-1).mean(-1)
            ytest_scores = ytest_scores.view(batch_size * num_test_examples, -1)
            pids = pids.view(batch_size * num_test_examples)
            _, ytest_preds = torch.max(ytest_scores.detach().cpu(), 1) 
            ytest_acc = (torch.sum(ytest_preds == pids.detach().cpu()).float()) / pids.size(0)
            ytest_accs.update(ytest_acc.item(), pids.size(0))

            # adaptive metric loss
            if args.adaptive_metrics and (args.method in ['IMLN']):
                # fix model, and only trian adaptive_metrics weights
                for name, param in model.named_parameters():
                    #print(name)
                    param.requires_grad=False
                    if name == 'module.similarity_res_func.adaptive_relation_weight':
                        param.requires_grad=True
                    if name == 'module.similarity_res_func.adaptive_proto_weight':
                        param.requires_grad=True
                    if name == 'module.similarity_res_func.adaptive_cosin_weight':
                        param.requires_grad=True

                output_results, _, ftest, _ = model(images_train, images_test, labels_train_1hot, labels_test_1hot, pids=pids, isda_ratio=isda_ratio, adapt_metrics=True)
                loss = criterion(output_results['metric_scores'], labels_test.view(-1))
                optimizer.zero_grad()
                loss.backward()
                # train model
                for name, param in model.named_parameters():
                    param.requires_grad=True


            if (len(train_loop) > 1) and (trans_ratio <= 0.5):
                break

            #if ((batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(trainloader)):
            #    print("== step: batch[{:3}/{}], epoch[{}/{}]".format(
            #          batch_idx + 1,
            #          len(trainloader),
            #          epoch,
            #          args.max_epoch,
            #          )
            #    )
    
    print('Epoch{0} '
          'lr: {1} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Global_Acc:{ytest_acc.avg:.4f} '
          'Loss:{loss.avg:.4f} '.format(
           epoch+1, learning_rate, batch_time=batch_time, 
           data_time=data_time, ytest_acc=ytest_accs, loss=losses))


def test(args, model, testloader, use_gpu, device):
    accs = AverageMeter()
    test_accuracies = []

    model_dir = os.path.dirname(args.resume)
    #model_dir = '/apdcephfs/private_jinxianglai/shared_info'
    for batch_idx , (images_train, labels_train, images_test, labels_test) in enumerate(testloader):
        if args.fine_tune:
            # test data for fine tune
            images_test_fine = images_train
            labels_test_fine = labels_train
            b, n2, c, h, w = images_test_fine.size()
            #rotation_loss = args.rotation_loss
            rotation_loss = False
            if rotation_loss:
                # rotate4 loss
                total_test_num = b * n2
                x = images_test_fine.view(total_test_num, c, h, w)
                y = labels_test_fine.view(total_test_num)
                x_ = []
                y_ = []
                a_ = []
                for j in range(total_test_num):
                    x90 = x[j].transpose(2,1).flip(1)
                    x180 = x90.transpose(2,1).flip(1)
                    x270 =  x180.transpose(2,1).flip(1)
                    x_ += [x[j], x90, x180, x270]
                    y_ += [y[j] for _ in range(4)]
                    a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]
                # to dataset
                x_ = Variable(torch.stack(x_,0)).view(b, n2*4, c, h, w)
                y_ = Variable(torch.stack(y_,0)).view(b, n2*4)
                a_ = Variable(torch.stack(a_,0)).view(b, n2*4)
                images_test_fine = x_
                labels_test_fine = y_
                if use_gpu:
                    a_ = a_.cuda()

        if use_gpu:
            if args.norm_layer == 'torchsyncbn':
                images_train = images_train.to(device)
                images_test = images_test.to(device)
                if args.fine_tune:
                    images_test_fine = images_test_fine.to(device)
            else:
                images_train = images_train.cuda()
                images_test = images_test.cuda()
                if args.fine_tune:
                    images_test_fine = images_test_fine.cuda()

        end = time.time()

        batch_size, num_train_examples, channels, height, width = images_train.size()
        num_test_examples = images_test.size(1)
        labels_test_org = labels_test
        labels_train_org = labels_train

        if args.norm_layer == 'torchsyncbn':
            labels_train_1hot = one_hot(labels_train).to(device)
            labels_test_1hot = one_hot(labels_test).to(device)
            if args.fine_tune:
                labels_test_fine_1hot = one_hot(labels_test_fine).to(device)
        else:
            labels_train_1hot = one_hot(labels_train).cuda()
            labels_test_1hot = one_hot(labels_test).cuda()
            if args.fine_tune:
                labels_test_fine_1hot = one_hot(labels_test_fine).cuda()

        # fine_tuning
        if args.fine_tune:
            # get ftest
            model.eval()
            cls_scores_fine, ftrain, ftest, _ = model(images_train, images_test_fine, labels_train_1hot, labels_test_fine_1hot)
            b, n2, c, h, w = ftest.size()

            # fine-tune
            # define classifier head
            clasifier_fine = ClasifierFinetune(args, c, h, w, args.nKnovel, in_feat=ftest, in_feat_onehot_label=labels_test_fine_1hot)
            ## DataParallel
            if len(args.gpu_devices) > 1:
                #print("=> {} GPU parallel".format(len(args.gpu_devices)))
                if args.norm_layer == 'bn':
                    clasifier_fine = nn.DataParallel(clasifier_fine, device_ids=args.gpu_devices)
                elif args.norm_layer == 'syncbn':
                    from extensions.tools.parallel import DataParallelModel
                    clasifier_fine = DataParallelModel(clasifier_fine, gather_=True)
                elif args.norm_layer == 'torchsyncbn':
                    # DistributedDataParallel
                    clasifier_fine = torch.nn.SyncBatchNorm.convert_sync_batchnorm(clasifier_fine)
                    clasifier_fine = clasifier_fine.to(device)
                    clasifier_fine = DDP(clasifier_fine, device_ids=[device], output_device=device, find_unused_parameters=True)
            # fine tuning param
            criterion = CrossEntropyLoss(args.using_focal_loss)
            fine_tune_lr = 0.01
            #optimizer = init_optimizer(args.optim, clasifier_fine.parameters(), fine_tune_lr, args.weight_decay)
            optimizer = torch.optim.SGD(clasifier_fine.parameters(), lr = fine_tune_lr, momentum=0.9, dampening=0.9, weight_decay=0.001)
            clasifier_fine.train()
            # start fine tuning
            if args.novel_classifier == 'Linear':
                fine_tune_epoch = 200
            elif args.novel_classifier == 'distLinear':
                fine_tune_epoch = 1000
            elif args.novel_classifier == 'ProtoInitDistLinear':
                fine_tune_epoch = 1500
            elif args.novel_classifier == 'Conv1':
                fine_tune_epoch = 300
            elif args.novel_classifier == 'ProtoInitConv1':
                fine_tune_epoch = 200
            for epoch in range(fine_tune_epoch):
                finetune_res, f_novel, classifier_constrain_loss = clasifier_fine(ftest, ftrain=ftrain)
                loss = criterion(finetune_res, labels_test_fine.view(-1))
                if args.novel_classifier == 'ProtoInitDistLinear':
                    if args.novel_classifier_constrain == 'ProtoMean':
                        classifier_constrain_scale = 1.0
                        loss = loss + classifier_constrain_scale * classifier_constrain_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch == 0 or (epoch + 1) % 10 == 0:
                    print('Test_epoch{0} '
                    'Epoch{1} '
                    'lr: {2} '
                    'Loss:{3} '.format(
                    batch_idx+1, epoch+1, fine_tune_lr, loss))

        # testing
        if args.fine_tune:
            clasifier_fine.eval()
        model.eval()
        with torch.no_grad():
            output_results, ftrain, ftest, ftest_visual = model(images_train, images_test, labels_train_1hot, labels_test_1hot)
            cls_scores = output_results['metric_scores']
            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)

            if args.fine_tune:
                finetune_res, f_novel_test, classifier_constrain_loss = clasifier_fine(ftest, ftrain=ftrain)
                cls_scores = finetune_res # novel_classifier only
                ##cls_scores = cls_scores + finetune_res # best setting

                #cls_scores = cls_scores/2.0 + finetune_res
                #cls_scores = F.softmax(cls_scores,dim=1) + F.softmax(finetune_res,dim=1)
                #cls_scores = cls_scores + F.softmax(finetune_res,dim=1)
                #cls_scores = F.softmax(cls_scores,dim=1) + finetune_res
                if args.novel_metric_classifier:
                    # f_novel_train
                    _, _, ftrain, _ = model(images_train, images_test_fine, labels_train_1hot, labels_test_fine_1hot)
                    _, f_novel_train, classifier_constrain_loss = clasifier_fine(ftrain)
                    # novel_metric_classifier
                    output_results_novel, _, ftest_novel, ftest_visual = model(images_train, images_test, labels_train_1hot, labels_test_1hot, ftrain_in=f_novel_train, ftest_in=f_novel_test)
                    cls_scores_novel = output_results_novel['metric_scores']
                    cls_scores_novel = cls_scores_novel.view(batch_size * num_test_examples, -1)
                    # final result
                    cls_scores = cls_scores_novel # novel_metric_classifier only
                    #cls_scores = cls_scores + cls_scores_novel

            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)
            labels_test = labels_test.view(batch_size * num_test_examples)

            _, preds = torch.max(cls_scores.detach().cpu(), 1) 
            preds_org = preds
            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
            accs.update(acc.item(), labels_test.size(0))
            #print('accs.avg: {:.2%}'.format(accs.avg))            

            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(batch_size, num_test_examples).numpy() #[b, n]
            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (batch_size))
            test_accuracies.append(acc)

            # badcase visual
            badcase_visual = False
            if badcase_visual:
                preds_org = preds_org.view(batch_size,num_test_examples)
                if int(args.local_rank) == 0:
                    # save images
                    for batch_i in range(batch_size):
                        # save dir root
                        batch_num_dir = model_dir + '/visual_sim/' + str(batch_idx*batch_size + batch_i)
                        check_makedirs(batch_num_dir)
                        # train data
                        images_train_bi = images_train[batch_i].cpu()
                        labels_train_org_bi = labels_train_org[batch_i].cpu()
                        for img_i in range(num_train_examples):
                            images_train_i = images_train_bi[img_i]
                            labels_train_org_i = int(labels_train_org_bi[img_i])
                            images_train_i_path = batch_num_dir + '/support-' + str(labels_train_org_i) + '.jpg'
                            imageSavePIL(images_train_i, images_train_i_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        # test data
                        images_test_bi = images_test[batch_i].cpu()
                        labels_test_org_bi = labels_test_org[batch_i].cpu()
                        preds_org_bi = preds_org[batch_i].cpu()
                        ftest_bi = ftest_visual[batch_i].cpu()
                        for img_i in range(num_test_examples):
                            images_test_i = images_test_bi[img_i]
                            labels_test_org_i = int(labels_test_org_bi[img_i])
                            preds_org_i = int(preds_org_bi[img_i])
                            ftest_i = ftest_bi[img_i]
                            images_test_i_path = batch_num_dir + '/' + str(labels_test_org_i) + '-' + str(preds_org_i) + '.jpg'
                            # badcase
                            #if labels_test_org_i != preds_org_i:
                            #    imageSavePIL(images_test_i, images_test_i_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            if True: # all
                            #if labels_test_org_i == preds_org_i: # only positive
                                imageSavePIL(images_test_i, images_test_i_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                # class activation map
                                cam_res = getCAM(images_test_i_path, ftest_i)
                                cam_res_i_path = batch_num_dir + '/' + str(labels_test_org_i) + '-' + str(preds_org_i) + '_cam.jpg'
                                cv2.imwrite(cam_res_i_path, cam_res)

    accuracy = accs.avg
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    stds = np.std(test_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(args.epoch_size)
    print('Accuracy: {:.2%}, std: :{:.2%}'.format(accuracy, ci95))

    return accuracy


if __name__ == '__main__':
    main()
