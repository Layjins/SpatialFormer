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
import cv2

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

from torchFewShot.models.net import Model
from torchFewShot.models.gcn import SCGCN_self
from torchFewShot.models.classifier_finetune import ClasifierFinetune
from torchFewShot.data_manager import DataManager
from torchFewShot.losses import CrossEntropyLoss
from torchFewShot.optimizers import init_optimizer

from torchFewShot.utils.iotools import save_checkpoint, check_isfile
from torchFewShot.utils.avgmeter import AverageMeter
from torchFewShot.utils.logger import Logger
from torchFewShot.utils.torchtools import one_hot, adjust_learning_rate
from torchFewShot.utils.mkdir import check_mkdir, check_makedirs
from torchFewShot.utils.tensor_to_img import imageSavePIL
from tensorboardX import SummaryWriter

#from test_tiered_args import argument_parser
from test_mini_args import argument_parser

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
    ## DataParallel
    if len(args.gpu_devices) > 1:
        print("=> {} GPU parallel".format(len(args.gpu_devices)))
        if args.norm_layer == 'bn':
            model = nn.DataParallel(model, device_ids=args.gpu_devices)
        elif args.norm_layer == 'syncbn':
            from extensions.tools.parallel import DataParallelModel
            model = DataParallelModel(model, gather_=True)
        elif args.norm_layer == 'torchsyncbn':
            # DistributedDataParallel
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(device)
            #model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=True)

        # load the model
        checkpoint = torch.load(args.resume, map_location=device)
    else:
        # load the model
        checkpoint = torch.load(args.resume, map_location='cuda:0')
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
    if len(args.gpu_devices) > 1:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=False)
    print("Loaded checkpoint from '{}'".format(args.resume))

    if use_gpu:
        model = model.cuda()

    test(args, model, testloader, use_gpu, device)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

# generate class activation mapping
def getCAM(img_path, feature):
    # class activation mapping
    feature = feature.numpy()
    c, h, w = feature.shape
    cam = feature.mean(0)
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    # sum the img and cam
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    CAM = cv2.resize(cam_img, (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    return result

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
        using_novel_query = False
        finetune_novel_query = False
        if args.fine_tune:
            # get ftest
            model.eval()
            if args.num_tSF_novel_queries > 0:
                using_novel_query = True
                finetune_novel_query = True
            cls_scores_fine, ftrain, ftest, _ = model(images_train, images_test_fine, labels_train_1hot, labels_test_fine_1hot, using_novel_query=using_novel_query, finetune_novel_query=finetune_novel_query)
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
            if args.num_tSF_novel_queries > 0:
                # fix model, and only trian tSF_novel_queries
                for name, param in model.named_parameters():
                    #print(name)
                    param.requires_grad=False
                    if name == 'module.tSF_novel_encoder.tSF_novel_block.novel_query_embed':
                        param.requires_grad=True
                optimizer = torch.optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': fine_tune_lr, 'momentum': 0.9, 'weight_decay': 0.001, 'nesterov': True},
                    {'params': clasifier_fine.parameters(), 'lr': fine_tune_lr, 'momentum': 0.9, 'weight_decay': 0.001, 'nesterov': True}])
            else:
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
                # linear classifier loss
                finetune_res, f_novel, classifier_constrain_loss = clasifier_fine(ftest, ftrain=ftrain)
                loss = criterion(finetune_res, labels_test_fine.view(-1))
                if args.novel_classifier == 'ProtoInitDistLinear':
                    if args.novel_classifier_constrain == 'ProtoMean':
                        classifier_constrain_scale = 1.0
                        loss = loss + classifier_constrain_scale * classifier_constrain_loss
                # metric classifier loss
                if args.num_tSF_novel_queries > 0:
                    loss_metric = criterion(cls_scores_fine['cls_scores'], labels_test_fine.view(-1))
                    loss = loss + 0.5*loss_metric
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
            finetune_novel_query = False
        model.eval()
        with torch.no_grad():
            output_results, ftrain, ftest, ftest_visual = model(images_train, images_test, labels_train_1hot, labels_test_1hot, using_novel_query=using_novel_query, finetune_novel_query=False)
            cls_scores = output_results['metric_scores']
            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)

            if args.fine_tune:
                if args.novel2base_feat == 'ClassWeightAttentionGT':
                    finetune_res, f_novel_test, classifier_constrain_loss = clasifier_fine(ftest, ftrain=ftrain, ytest=labels_test.view(-1))
                else:
                    finetune_res, f_novel_test, classifier_constrain_loss = clasifier_fine(ftest, ftrain=ftrain)
                cls_scores = finetune_res # novel_classifier only
                ##cls_scores = cls_scores + finetune_res # best setting

                #cls_scores = cls_scores/2.0 + finetune_res
                #cls_scores = F.softmax(cls_scores,dim=1) + F.softmax(finetune_res,dim=1)
                #cls_scores = cls_scores + F.softmax(finetune_res,dim=1)
                #cls_scores = F.softmax(cls_scores,dim=1) + finetune_res
                if args.novel_metric_classifier:
                    # f_novel_train
                    _, _, ftrain, _ = model(images_train, images_test_fine, labels_train_1hot, labels_test_fine_1hot, using_novel_query=using_novel_query, finetune_novel_query=False)
                    if args.novel2base_feat == 'ClassWeightAttentionGT':
                        _, f_novel_train, classifier_constrain_loss = clasifier_fine(ftrain, ytest=labels_train.view(-1))
                    else:
                        _, f_novel_train, classifier_constrain_loss = clasifier_fine(ftrain)
                    # novel_metric_classifier
                    output_results_novel, _, ftest_novel, ftest_visual = model(images_train, images_test, labels_train_1hot, labels_test_1hot, ftrain_in=f_novel_train, ftest_in=f_novel_test, using_novel_query=using_novel_query, finetune_novel_query=False)
                    cls_scores_novel = output_results_novel['metric_scores']
                    cls_scores_novel = cls_scores_novel.view(batch_size * num_test_examples, -1)
                    # final result
                    #cls_scores = cls_scores_novel # novel_metric_classifier only
                    cls_scores = cls_scores + cls_scores_novel

            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)
            labels_test = labels_test.view(batch_size * num_test_examples)

            _, preds = torch.max(cls_scores.detach().cpu(), 1) 
            preds_org = preds
            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
            accs.update(acc.item(), labels_test.size(0))
            print('accs.avg: {:.2%}'.format(accs.avg))            

            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(batch_size, num_test_examples).numpy() #[b, n]
            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (batch_size))
            test_accuracies.append(acc)

            # t-SNE visual
            tsne_visual = False
            if tsne_visual and (int(args.local_rank) == 0):
                # ftest: (b, n2, c, h, w)
                #ftest_tsne = ftest.contiguous().view(-1, *ftest.size()[2:]) # (b*n2, c, h, w)
                ftest_tsne = ftest.contiguous().view(batch_size * num_test_examples, -1)
                with SummaryWriter(log_dir=model_dir + '/visual_tsne') as writer:
                    writer.add_embedding(
                        ftest_tsne.data,
                        metadata=labels_test.data,
                        #label_img=images_test.contiguous().view(-1, *images_test.size()[2:]).data,
                        global_step=batch_idx)

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
