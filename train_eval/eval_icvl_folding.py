'''
evaluation
'''
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import random
import progressbar
import time
import logging
import pdb
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from thop import profile, clever_format

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from utils import group_points, rotate_point_cloud_by_angle_flip

from dataset_icvl import HandPointDataset

from network_icvl_folding import PointNet_Plus

from utils import group_points

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python eval.py

parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
parser.add_argument('--bit_width', type=int, default=4, help='quantize for bit width')
parser.add_argument('--SAMPLE_NUM', type=int, default = 1024,  help='number of sample points')
parser.add_argument('--JOINT_NUM', type=int, default = 16,  help='number of joints')
parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 6,  help='number of input point features')
parser.add_argument('--PCA_SZ', type=int, default = 42,  help='number of PCA components')
parser.add_argument('--knn_K', type=int, default = 64,  help='K for knn search')
parser.add_argument('--sample_num_level1', type=int, default = 512,  help='number of first layer groups')
parser.add_argument('--sample_num_level2', type=int, default = 128,  help='number of second layer groups')
parser.add_argument('--ball_radius', type=float, default=0.015, help='square of radius for ball query in level 1')
parser.add_argument('--ball_radius2', type=float, default=0.04, help='square of radius for ball query in level 2')

parser.add_argument('--test_index', type=int, default = 0,  help='test index for cross validation, range: 0~8')
parser.add_argument('--save_root_dir', type=str, default='../results/icvlfolding',  help='output folder')
parser.add_argument('--model', type=str, default = 'netR_SOTA.pth',  help='model name for training resume')
parser.add_argument('--test_path', type=str, default = '../data/ICVL_center_pre0/Testing',  help='model name for training resume')

opt = parser.parse_args()
print (opt)

# torch.cuda.set_device(opt.main_gpu)

opt.manualSeed = 10
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

save_dir = os.path.join(opt.save_root_dir)

# 1. Load data                                         
# test_data = HandPointDataset(root_path='./data/ICVL_center_pre0/Testing', opt=opt, train = False)
test_data = HandPointDataset(root_path=opt.test_path, opt=opt, train = False)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers), pin_memory=False)
                                          
print('#Test data:', len(test_data))
print (opt)

# 2. Define model, loss
netR = PointNet_Plus(opt)
if opt.ngpu > 1:
    netR.netR_1 = torch.nn.DataParallel(netR.netR_1, range(opt.ngpu))
    netR.netR_2 = torch.nn.DataParallel(netR.netR_2, range(opt.ngpu))
    netR.netR_3 = torch.nn.DataParallel(netR.netR_3, range(opt.ngpu))
if opt.model != '':
    netR.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))
    
netR.cuda()
print(netR)

criterion = nn.MSELoss(size_average=True).cuda()

# 3. evaluation
torch.cuda.synchronize()

netR.eval()
test_mse = 0.0
test_wld_err = 0.0
test_wld_err_fold1 = 0.0
test_class_err = torch.zeros(opt.JOINT_NUM, 1).cuda()
t = 0
saved_points = []
saved_gt = []
saved_fold1 = []
saved_final = []
saved_error = []
saved_length = []

for i, data in enumerate(tqdm(test_dataloader, 0)):
	torch.cuda.synchronize()
	# 3.1 load inputs and targets
	with torch.no_grad():
		points, volume_length, gt_xyz = data
		points, volume_length, gt_xyz = points.cuda(), volume_length.cuda(), gt_xyz.cuda()
		
		permutation = torch.randperm(points.size(1))
		points = points[:,permutation,:]

		timer = time.time()
		# points: B * 1024 * 6
		inputs_level1, inputs_level1_center = group_points(points, opt)
		inputs_level1, inputs_level1_center = Variable(inputs_level1, volatile=True), Variable(inputs_level1_center, volatile=True)
		
		# 3.2 compute output

		fold1, fold2, estimation = netR(inputs_level1, inputs_level1_center)
		loss = (criterion(estimation, gt_xyz)*2+criterion(fold1, gt_xyz)+criterion(fold2, gt_xyz))*opt.JOINT_NUM*3

		t = t + time.time() - timer
		if not i:
			macs, params = profile(netR, inputs=(inputs_level1, inputs_level1_center))
			macs, params = clever_format([macs, params], "%.3f")
			print(macs, params)
		# break
		# netR.summary()


	torch.cuda.synchronize()
	test_mse = test_mse + loss.item()*len(points)

	# 3.3 compute error in world cs
	# wrist: [0], index_R: [1], index_T: [4], middle_R: [5], middle_T: [8], ring_R: [9], ring_T: [12], little_R: [13], little_T: [16], thumb_R: [17], thumb_T: [20] 
	outputs_xyz = estimation
	diff = torch.pow(outputs_xyz-gt_xyz, 2).view(-1,opt.JOINT_NUM,3)
	diff_sum = torch.sum(diff,2)
	diff_sum_sqrt = torch.sqrt(diff_sum)
	diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
	diff_mean_wld = torch.mul(diff_mean,volume_length)

	diff_mean_class = torch.mul(diff_sum_sqrt, volume_length)
	diff_mean_class = torch.sum(diff_mean_class,0).view(-1,1)

	test_wld_err = test_wld_err + diff_mean_wld.sum()
	test_class_err = test_class_err + diff_mean_class	

	# fold1
	outputs_xyz = fold1
	diff = torch.pow(outputs_xyz-gt_xyz, 2).view(-1,opt.JOINT_NUM,3)
	diff_sum = torch.sum(diff,2)
	diff_sum_sqrt = torch.sqrt(diff_sum)
	diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
	diff_mean_wld = torch.mul(diff_mean,volume_length)

	test_wld_err_fold1 = test_wld_err_fold1 + diff_mean_wld.sum()


	saved_points.append(points.cpu().numpy())
	saved_gt.append(gt_xyz.cpu().numpy())
	saved_fold1.append(fold1.cpu().numpy())
	saved_final.append(estimation.cpu().numpy())
	saved_error.append(diff_mean_wld.cpu().numpy())
	saved_length.append(volume_length.cpu().numpy())

print('saving data.......')
np.save(os.path.join(opt.save_root_dir, "saved_points"), np.array(saved_points))
np.save(os.path.join(opt.save_root_dir, "saved_gt"), np.array(saved_gt))
np.save(os.path.join(opt.save_root_dir, "saved_fold1"), np.array(saved_fold1))
np.save(os.path.join(opt.save_root_dir, "saved_final"), np.array(saved_final))
np.save(os.path.join(opt.save_root_dir, "saved_error"), np.array(saved_error))
np.save(os.path.join(opt.save_root_dir, "saved_length"), np.array(saved_length))

# time taken
torch.cuda.synchronize()
# timer = time.time() - timer
t = t / len(test_data)
print('==> time to learn 1 sample = %f (ms)' %(t*1000))

# print mse
test_wld_err = test_wld_err / len(test_data)
print('average estimation error in world coordinate system: %f (mm)' %(test_wld_err))
test_wld_err_fold1 = test_wld_err_fold1/ len(test_data)
print('average fold1 error in world coordinate system: %f (mm)' %(test_wld_err_fold1))

# test_class_err = test_class_err / len(test_data)
# print("wrist error: ", test_class_err[0])
# print("index_R: ", test_class_err[1])
# print("index_T: ", test_class_err[4])
# print("middle_R: ", test_class_err[5])
# print("middle_T: ", test_class_err[8])
# print("ring_R: ", test_class_err[9])
# print("ring_T: ", test_class_err[12])
# print("little_R: ", test_class_err[13])
# print("little_T: ", test_class_err[16])
# print("thumb_R: ", test_class_err[17])
# print("thumb_T: ",test_class_err[20])
