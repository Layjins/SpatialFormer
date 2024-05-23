import time
import argparse
import torchFewShot

def argument_parser():

    parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('-d', '--dataset', type=str, default='tieredImageNet_load')
    parser.add_argument('--load', default=True)

    parser.add_argument('-j', '--workers', default=8, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--org_height', type=int, default=84, help="height of an image")
    parser.add_argument('--height', type=int, default=84,
                        help="height of an image (default: 84)")
    parser.add_argument('--width', type=int, default=84,
                        help="width of an image (default: 84)")

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='sgd',
                        help="optimization algorithm (see optimizers.py)")
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")

    parser.add_argument('--max-epoch', default=100, type=int,
                        help="maximum epochs to run, default=100")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--stepsize', default=[80], nargs='+', type=int,
                        help="stepsize to decay learning rate, default=[80]")
    parser.add_argument('--LUT_lr', default=[(40, 0.05), (60, 0.01), (80, 0.001), (100, 0.0001)],
                        help="multistep to decay learning rate, default=[(40, 0.05), (60, 0.01), (80, 0.001), (100, 0.0001)]")

    parser.add_argument('--train-batch', default=8//8, type=int,
                        help="train batch size")
    parser.add_argument('--test-batch', default=4//4, type=int,
                        help="test batch size")


    # ************************************************************
    # Architecture settings
    # ************************************************************
    parser.add_argument('--method', type=str, default='CAN', help="CAN, IMLN")
    parser.add_argument('--backbone', default='resnet12_gcn_640', 
                        help="conv4, conv4_512, conv4_512_s, resnet12, resnet12_avg, resnet12_gcn, resnet12_gcn_640, resnet12_gcn_avg, resnet12_cae, ResNet12_BDC, \
                        res2net50, rest_small, wrn28_10, wrn28_10_cam, wrn28_10_gcn, hrnet_w18_small_v1, hrnetv2_w18, densenet121")
    # neck
    parser.add_argument('--neck', default='None', help="None, Coarse_feat, Fine_feat, Task_feat, Coarse_Fine, Coarse_Task, Coarse_Fine_Task")
    parser.add_argument('--num_queries', default={'coarse': 5, 'fine': 64, 'dataset': 5}, help="num_queries for Coarse_feat, Fine_feat, Task_feat in neck")
    parser.add_argument('--num_heads', default=4, help="num_heads for Coarse_feat, Fine_feat, Task_feat in neck")
    parser.add_argument('--CECE_mode', default="Transformer", help="CECE_mode: MatMul, Cosine, GCN, Transformer")
    parser.add_argument('--class_center', default="Mean", help="class center module: Mean")
    # global classifier attention
    parser.add_argument('--global_classifier_attention', default="None", help="global classifier attention: None, ClassWeightAttention, ThreshClassAttention, TSCA, TCA")
    parser.add_argument('--palleral_attentions', default=True, help="global_classifier_attention and cross attention are palleral")
    # attention and distance metric
    parser.add_argument('--attention', default="TCAM", help="attention module: None, CAM, DGC_CAM, SuperGlue, LoFTR, NLBlockND_CAM, NLCAM, TCAM, CECM, TSIA, TIA")
    parser.add_argument('--CECM_mode', default="Transformer", help="CECM_mode: MatMul, Cosine, GCN, Transformer")
    parser.add_argument('--CECM_connect', default="one2one", help="CECM_connect mode: one2one, one2many")
    parser.add_argument('--distance_metric', default="Cosine", help="distance metric: Cosine, CECD")
    parser.add_argument('--CECD_mode', default="Cosine", help="CECD_mode: MatMul, Cosine, GCN, Transformer")
    parser.add_argument('--attention_pool', type=int, default=1, help="pool after the cross attention module")
    parser.add_argument('--auxiliary_attention', default=False, help="gcn attention for auxiliary tasks of global and rotation classifiers")
    parser.add_argument('--relation_dim', type=int, default=8, help="only for relation module")
    parser.add_argument('--adaptive_metrics', default=False, help="adaptive metrics for multiple metrics fusion")
    parser.add_argument('--metrics_kl', default=False, help="KL regularization for adaptive metrics")
    parser.add_argument('--emd_metric', default=False, help="adding EMD metric into AMM module")
    # auxiliary loss
    parser.add_argument('--using_power_trans', default=False, help="power transform for feature in testing stage")
    parser.add_argument('--using_focal_loss', default=True, help="focal loss")
    parser.add_argument('--manifold_mixup', default=False, help="manifold_mixup loss from S2M2")
    parser.add_argument('--mosaic', default=False, help="augmentation by stacking multiple images")
    parser.add_argument('--rotation_loss', default=True, help="rotation self-supervied learning")
    parser.add_argument('--global_weighted_loss', default=True, help="global automatic weighted loss")
    parser.add_argument('--teacher_loss', type=str, default='None', help="teacher_loss=['None', 'global_teacher', 'metric_teacher', 'co_teacher']")
    parser.add_argument('--mix_up_loss', default=False, help="mix up loss")
    parser.add_argument('--global_feat_mix_loss', default=False, help="global feature mix loss")
    parser.add_argument('--isda_loss', default=False, help="implicit semantic data augmentation loss")
    parser.add_argument('--vae_loss', default=False, help="vae reconstruction loss")
    parser.add_argument('--redundancy_loss', default=False, help="redundancy loss to make feature distingwish")
    # knowledge distillation loss
    parser.add_argument('--backbone_teacher', default='None', 
                        help="None, conv4, conv4_512, resnet12, resnet12_avg, \
                        resnet12_gcn, resnet12_gcn_640, resnet12_gcn_avg, wrn28_10")
    parser.add_argument('--backbone_t_path', type=str, default='None')
    # fine-tuning in test stage
    parser.add_argument('--fine_tune', default=False, help="fine tuning in the test stage")
    parser.add_argument('--novel_feature', default="None", help="module to obtain novel feature in test stage: None, Coarse_feat, Task_feat, SuperGlue, ResT, DGCN_self, NLBlockND, NLBlockSimple")
    parser.add_argument('--novel_feature_constrain', default="None", help="novel feature constrain for preventing overfitting in test stage: None, ProtoMean")
    parser.add_argument('--novel_classifier', default="distLinear", help="novel classifier by fineturning in test stage: Linear, distLinear, Conv1, ProtoInitDistLinear, ProtoInitConv1")
    parser.add_argument('--novel_classifier_constrain', default="None", help="novel classifier weights constrain for preventing overfitting in test stage: None, ProtoMean")
    parser.add_argument('--novel_metric_classifier', default=False, help="metric-based classifier to classify the novel feature")
    parser.add_argument('--novel2base_feat', default="None", help="update the base-pattern feature by using novel classifier in test stage: None, SimMapAttention, ClassWeightAttention")
    parser.add_argument('--task_specific_scaling', default=False, help="task-specific scaling in test stage")
    # other setting
    parser.add_argument('--num_classes', type=int, default=351, help="num_classes: train=351, val=97")
    parser.add_argument('--scale_cls', type=int, default=7)

    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--save-dir', type=str, default='/persons/jinxianglai/FewShotLearning/few-shot_classification/CAN_output/tieredImageNet/{}'.format(time.strftime('%Y.%m.%d.%H.%M.%S', time.localtime(time.time()))))
    parser.add_argument('--fix_backbone', default=False, help="fix backbone in training")
    parser.add_argument('--load_backbone', default=False, help="only load the backbone in training")
    parser.add_argument('--resume', type=str, default='None', metavar='PATH')
    #parser.add_argument('--resume', type=str, default='/persons/jinxianglai/FewShotLearning/few-shot_classification/CAN_output/tieredImageNet/2021.06.22.18.28.12/best_model.pth.tar', metavar='PATH')
    parser.add_argument('--gpu-devices', default=[0,1,2,3,4,5,6,7])
    parser.add_argument('--norm_layer', type=str, default='torchsyncbn', help="bn, in, syncbn, torchsyncbn")
    parser.add_argument('--local_rank')

    # ************************************************************
    # FewShot settting
    # ************************************************************
    parser.add_argument('--nKnovel', type=int, default=5,
                        help='number of novel categories')
    parser.add_argument('--nExemplars', type=int, default=1,
                        help='number of training examples per novel category.')

    parser.add_argument('--train_nTestNovel', type=int, default=6 * 5,
                        help='number of test examples for all the novel category when training')
    parser.add_argument('--train_epoch_size', type=int, default=13980,
                        help='number of batches per epoch when training')
    parser.add_argument('--nTestNovel', type=int, default=15 * 5,
                        help='number of test examples for all the novel category')
    parser.add_argument('--epoch_size', type=int, default=2000,
                        help='number of batches per epoch')

    parser.add_argument('--phase', default='test', type=str,
                        help='use test or val dataset to early stop') # default='test' of CAN origin
    parser.add_argument('--seed', type=int, default=1)

    return parser

