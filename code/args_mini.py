import time
import argparse
import torchFewShot

def argument_parser():

    parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('-d', '--dataset', type=str, default='miniImageNet_load',help="miniImageNet, miniImageNet_load")
    parser.add_argument('--load', default=True,help="miniImageNet=False, miniImageNet_load=True")

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
                        help="optimization algorithm (see optimizers.py), default=sgd")
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        help="initial learning rate, default=0.1")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")

    parser.add_argument('--max-epoch', default=120, type=int,
                        help="maximum epochs to run, default=100")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--stepsize', default=[90], nargs='+', type=int,
                        help="stepsize to decay learning rate, default=[70]")
    parser.add_argument('--LUT_lr', default=[(20, 0.05), (90, 0.1), (100, 0.006), (110, 0.0012), (120, 0.00024)],
                        help="multistep to decay learning rate, default=[(70, 0.1), (80, 0.006), (90, 0.0012), (100, 0.00024)]")

    parser.add_argument('--train-batch', default=4//4, type=int,
                        help="train batch size, default=4")
    parser.add_argument('--test-batch', default=4//4, type=int,
                        help="test batch size, default=4")

    # ************************************************************
    # Architecture settings
    # ************************************************************
    parser.add_argument('--method', type=str, default='CAN', help="CAN, IMLN")
    parser.add_argument('--backbone', default='resnet12_exp_c896_k3', 
                        help="conv4, conv4_512, conv4_512_s, resnet12, resnet12_avg, resnet12_gcn, resnet12_gcn_640, resnet12_gcn_avg, resnet12_cae, ResNet12_BDC, \
                        resnet12_gcn_640_tSF, res2net50, rest_small, wrn28_10, wrn28_10_cam, wrn28_10_gcn, hrnet_w18_small_v1, hrnetv2_w18, densenet121, \
                        resnet12_exp_c640_k3, resnet12_exp_c640_k5, resnet12_exp_c512_k3, resnet12_exp_c768_k3, resnet12_exp_c896_k3, resnet12_exp_c960_k3, resnet12_exp_c1024_k3, \
                        wrn28_10_16_16_32_64, wrn28_10_32_32_64_96, wrn28_10_64_32_64_96")
    # neck
    parser.add_argument('--neck', default='tSF_plus', help="None, Coarse_feat, Fine_feat, Task_feat, Coarse_Fine, Coarse_Task, Coarse_Fine_Task, \
                        tSF_novel, tSF_stacker1, tSF_stacker2, tSF_encoder2, SIQ_encoder2, SIQ_encoder3, tSF_T, tSF_T_tSF, tSF_T_tSF2, tSF_BDC, CECE, \
                        MSF, tSF_prototype, tPF, tSF_tPF, tSF_plus")
    parser.add_argument('--tSF_plus_mode', default="tSF_BEP", help="tSF_plus_mode: tSF_F, tSF_E, tSF_SP, tSF_BEP, tSF_BEP_SP, tSF_BEP_local, tSF_BEP_global, tSF_E_Metric, \
                        SAT_F, SAT_E, SAT_SP, SAT_BEP")
    parser.add_argument('--tSF_plus_num', default=1, help="stacking layer number of tSF_plus")
    parser.add_argument('--add_tSF', default=False, help="add tSF with the output of neck")
    parser.add_argument('--num_queries', default={'coarse': 5, 'fine': 64, 'dataset': 5}, help="num_queries for Coarse_feat, Fine_feat, Task_feat in neck")
    parser.add_argument('--num_heads', default=4, help="num_heads for Coarse_feat, Fine_feat, Task_feat in neck")
    parser.add_argument('--CECE_mode', default="Transformer", help="CECE_mode: MatMul, Cosine, GCN, Transformer")
    parser.add_argument('--class_center', default="Mean", help="class center module: Mean, EmbeddingAlignment")
    # global classifier attention
    parser.add_argument('--global_classifier_attention', default="None", help="global classifier attention: None, ClassWeightAttention, ThreshClassAttention, TSCA, TCA")
    parser.add_argument('--palleral_attentions', default=True, help="global_classifier_attention and cross attention are palleral")
    # attention and distance metric
    parser.add_argument('--attention', default="None", help="attention module: None, CAM, DGC_CAM, SuperGlue, LoFTR, NLBlockND_CAM, NLCAM, TCAM, CECM, TSIA, TIA")
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
    parser.add_argument('--embed_classifier_train', default=False, help="embed_classifier_train for auxilialy similarity metric")
    parser.add_argument('--embed_classifier_test', default=False, help="embed_classifier_test for auxilialy similarity metric")
    parser.add_argument('--cluster_embed_by_support', default=False, help="cluster embed by considering support as the cluster center")
    parser.add_argument('--embed_classifier_weight', default=0.1, help="embed_classifier_weight for auxilialy similarity metric")
    # auxiliary loss
    parser.add_argument('--using_power_trans', default=False, help="power transform for feature in testing stage")
    parser.add_argument('--using_focal_loss', default=True, help="focal loss")
    parser.add_argument('--manifold_mixup', default=False, help="manifold_mixup loss from S2M2")
    parser.add_argument('--mosaic', default=False, help="augmentation by stacking multiple images")
    parser.add_argument('--rotation_loss', default=False, help="rotation self-supervied learning")
    parser.add_argument('--global_weighted_loss', default=False, help="global automatic weighted loss")
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
    parser.add_argument('--num_tSF_novel_queries', default=5, help="fine tuning tSF_novel_queries: num_tSF_novel_queries for tSF_novel in neck")
    parser.add_argument('--novel_feature', default="None", help="module to obtain novel feature in test stage: None, Coarse_feat, Task_feat, SuperGlue, ResT, DGCN_self, NLBlockND, NLBlockSimple")
    parser.add_argument('--novel_feature_constrain', default="None", help="novel feature constrain for preventing overfitting in test stage: None, ProtoMean")
    parser.add_argument('--novel_classifier', default="distLinear", help="novel classifier by fineturning in test stage: Linear, distLinear, Conv1, ProtoInitDistLinear, ProtoInitConv1")
    parser.add_argument('--novel_classifier_constrain', default="None", help="novel classifier weights constrain for preventing overfitting in test stage: None, ProtoMean")
    parser.add_argument('--novel_metric_classifier', default=False, help="metric-based classifier to classify the novel feature")
    parser.add_argument('--novel2base_feat', default="None", help="update the base-pattern feature by using novel classifier in test stage: None, SimMapAttention, ClassWeightAttention")
    parser.add_argument('--task_specific_scaling', default=False, help="task-specific scaling in test stage")
    # other setting
    parser.add_argument('--num_classes', type=int, default=64, help="num_classes: train=64, val=16")
    parser.add_argument('--scale_cls', type=int, default=7)

    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--save-dir', type=str, default='/persons/jinxianglai/FewShotLearning/few-shot_classification/CAN_output/miniImageNet/{}'.format(time.strftime('%Y.%m.%d.%H.%M.%S', time.localtime(time.time()))))
    parser.add_argument('--fix_backbone', default=False, help="fix backbone in training")
    parser.add_argument('--load_backbone', default=False, help="only load the backbone in training")
    parser.add_argument('--resume', type=str, default='None', metavar='PATH')
    #parser.add_argument('--resume', type=str, default='/persons/jinxianglai/FewShotLearning/few-shot_classification/CAN_output/miniImageNet/2021.08.20.09.59.03/best_model.pth.tar', metavar='PATH')
    parser.add_argument('--gpu-devices', default=[0,1,2,3])
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
                        help='number of test examples for all the novel category when training, default=6 * 5')
    parser.add_argument('--train_epoch_size', type=int, default=1200,
                        help='number of batches per epoch when training, default=1200')
    parser.add_argument('--nTestNovel', type=int, default=15 * 5,
                        help='number of test examples for all the novel category')
    parser.add_argument('--epoch_size', type=int, default=2000,
                        help='number of batches per epoch, default=2000')

    parser.add_argument('--phase', default='test', type=str,
                        help='use test or val dataset to early stop') # default='test' of CAN origin 
    parser.add_argument('--seed', type=int, default=1)

    return parser

