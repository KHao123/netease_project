import torch
import argparse
import os
import sys
import os.path as osp
import torch.backends.cudnn as cudnn
from torchsummary import summary
from torch.nn.parallel.data_parallel import DataParallel
from config import cfg
from thop import profile
from thop import clever_format
from ptflops import get_model_complexity_info
from model import get_model,get_model_small
import time
from collections import OrderedDict
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

sys.path.insert(0, cfg.smpl_path)
from smplpytorch.pytorch.smpl_layer import SMPL_Layer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--stage', type=str, dest='stage')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    if not args.stage:
        assert 0, "Please set training stage among [lixel, param]"

    assert args.test_epoch, 'Test epoch is required.'
    return args

# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids, args.stage)
cudnn.benchmark = True

# SMPL joint set
joint_num = 29 # original: 24. manually add nose, L/R eye, L/R ear
joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear')
flip_pairs = ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23) , (25,26), (27,28) )
skeleton = ( (0,1), (1,4), (4,7), (7,10), (0,2), (2,5), (5,8), (8,11), (0,3), (3,6), (6,9), (9,14), (14,17), (17,19), (19, 21), (21,23), (9,13), (13,16), (16,18), (18,20), (20,22), (9,12), (12,24), (24,15), (24,25), (24,26), (25,27), (26,28) )

# SMPl mesh
vertex_num = 6890
smpl_layer = SMPL_Layer(gender='neutral', model_root=cfg.smpl_path + '/smplpytorch/native/models')
face = smpl_layer.th_faces.numpy() # (13776, 3)
joint_regressor = smpl_layer.th_J_regressor.numpy() # (24, 6890) 这个表示什么，就是论文里面的J，为什么每个数据集都有一个
root_joint_idx = 0

# snapshot load
# model_path = './snapshot_%d.pth.tar' % int(args.test_epoch)
# assert osp.exists(model_path), 'Cannot find model at ' + model_path
# print('Load checkpoint from {}'.format(model_path))

# model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % int(args.test_epoch))
# assert os.path.exists(model_path), 'Cannot find model at ' + model_path
# print('Load checkpoint from {}'.format(model_path))

model = get_model_small(vertex_num, joint_num, 'test')

# model = DataParallel(model).cuda()
model = DataParallel(model).cuda()
# ckpt = torch.load(model_path)
# model.load_state_dict(ckpt['network'])
model.eval()

# time_list = []
# for i in range(100):
#     # input = torch.randn(1, 3, 256, 256).cuda()
#     input = torch.randn(1, 3, 256, 256)
#     torch.cuda.synchronize()
#     time_start = time.time()
#     predict = model(input)
#     torch.cuda.synchronize()
#     time_end = time.time()
#     time_sum = time_end - time_start
#     print("GPU inference time:", time_sum)
#     time_list.append(time_sum)



# model = model.module

# # summary(model, (3, 256, 256))

input = torch.randn(1, 3, 256, 256).cuda()
input = {'img': input}
# macs, params = profile(model, inputs=(input,))
# macs, params = clever_format([macs, params], "%.3f")
# flops, params1 = get_model_complexity_info(model, (3, 256, 256),as_strings=True, print_per_layer_stat=False)
# print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params1))
flops = FlopCountAnalysis(model, input)
print(flop_count_table(flops))

# print("mean time comsume:", sum(time_list)/len(time_list))  
