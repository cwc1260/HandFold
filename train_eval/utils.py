'''
utils
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import numpy as np

def offset_cal(points,opt):
	#make offset using knn and ball query
	#points: B*(21+1024)*3
        cur_train_size=len(points)
        inputs1_diff = points.transpose(1,2).unsqueeze(1).expand(cur_train_size,opt.JOINT_NUM,3,points.size(1)) \
        			 - points[:,0:opt.JOINT_NUM,:].unsqueeze(-1).expand(cur_train_size,opt.JOINT_NUM,3,points.size(1)) #B*21*3*1045
        inputs1_diff = inputs1_diff[:,:,:,opt.JOINT_NUM:points.size(1)] #B*21*3*1024
        inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)      # B * 21 * 3 * 1024
        inputs1_diff = inputs1_diff.sum(2)                      # B * 21* 1024
        dists, inputs1_idx = torch.topk(inputs1_diff, opt.knn_K, 2, largest=False, sorted=False)  # dists: B * 21 * 64; inputs1_idx: B * 21 * 64

        # ball query
        invalid_map = dists.gt(opt.ball_radius) # B * 21 * 64
        for jj in range(opt.JOINT_NUM):
                inputs1_idx[:,jj,:][invalid_map[:,jj,:]] = jj
        idx_group_l1_long = inputs1_idx.view(cur_train_size,opt.JOINT_NUM*opt.knn_K,1).expand(cur_train_size,opt.JOINT_NUM*opt.knn_K,3)  # B*(21*64)*3
        idx_group_l1_long_long = inputs1_idx.view(cur_train_size,opt.JOINT_NUM,opt.knn_K).unsqueeze(3).expand(cur_train_size,opt.JOINT_NUM,opt.knn_K,4)  # B*21*64*4
        
		
        
		
        #all_idx_long_1 = torch.arange(opt.SAMPLE_NUM, dtype=torch.long).repeat(opt.JOINT_NUM).unsqueeze(0).unsqueeze(2).expand(cur_train_size,opt.JOINT_NUM*opt.SAMPLE_NUM,3) # B*(21*1024)*3
        #all_idx_long_2 = torch.arange(opt.SAMPLE_NUM, dtype=torch.long).repeat(opt.JOINT_NUM) #(21*1024)
        point_cld = points[:,opt.JOINT_NUM:points.size(1),:] 	#points: B*(1024)*3
		
        #point_cld_mlt = point_cld[:,all_idx_long_2,:].view(cur_train_size,opt.JOINT_NUM,opt.SAMPLE_NUM,3)      #B*21*1024*3
        #zeros_1 = torch.zeros(cur_train_size,opt.JOINT_NUM*opt.SAMPLE_NUM,3)   #B*(21*1024)*3
        #zeros_1[idx_group_l1_long] = point_cld[:,:,:].gather(1,idx_group_l1_long)  # zeros_1=B*(21*1024)*3
        inputs_level1 = point_cld[:,:,:].gather(1,idx_group_l1_long).view(cur_train_size,opt.JOINT_NUM,opt.knn_K,3) # B*21*64*3
        inputs_level1_center = points[:,0:opt.JOINT_NUM,:].unsqueeze(2)       # B*21*1*3
        offset = inputs_level1[:,:,:,:] - inputs_level1_center.expand(cur_train_size,opt.JOINT_NUM,opt.knn_K,3) # B*21*64*3
        dist =  torch.mul(offset, offset) # B*21*64*3
        dist = offset.sum(3).unsqueeze(3) # B*21*64*1
        heatmap = 1-dist/opt.ball_radius   # B*21*64*1
        vector = offset/dist.expand(cur_train_size,opt.JOINT_NUM,opt.knn_K,3) # B*21*64*3
        heatmap = torch.cat((heatmap,vector),3)   # B*21*64*4
        heatmap_1 = torch.zeros(cur_train_size,opt.JOINT_NUM,points.size(1)-opt.JOINT_NUM,4).cuda() # B*21*1024*4
        heatmap_1 = heatmap_1.scatter_(2, idx_group_l1_long_long, heatmap) #heatmap_1 = B*21*1024*4
        #heatmap_1[:,idx_group_l1_long_long,idx_group_l1_long_long,:] = heatmap #heatmap_1 = B*21*1024*4
		
        heatmap = heatmap_1.permute(0,1,3,2) #B*21*4*1024
       #print(heatmap.shape)
        heatmap = heatmap.contiguous().view(cur_train_size,opt.JOINT_NUM*4,points.size(1)-opt.JOINT_NUM) #B*4J*1024
		#heatmap_1 = torch.zeros(cur_train_size,opt.JOINT_NUM*4,1024) #B*4J*(1024)
        #append = torch.zeros(cur_train_size,opt.JOINT_NUM*4,1024-opt.knn_K) #B*4J*(1024-64)
        #heatmap = torch.cat((heatmap,append),2) #B*4J*1024
        heatmap = heatmap.unsqueeze(3) #B*4J*1024*1
        return heatmap


def jitter(points, var=0.01, clip=0.05):
    device = points.device
    B, N, C = points.shape

    noise = torch.clamp(torch.normal(0.0, var, [B, N, 3]), -1 * clip, clip).to(device)

    points[:,:, 0:3] = points[:,:, 0:3] + noise
    return points

def rotate_point_cloud_by_random(batch_data, batch_gt, normal=False):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    device = batch_data.device
    B, N, C = batch_data.shape

    for k in range(B):
        # ICVL
        rotation_angle = np.random.rand() * 2 * np.pi
        # MSRA
        # rotation_angle = rotation_region / 48.0 * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cosval, -sinval],
                                    [0, sinval,  cosval]], dtype=np.float32)

        rotation_matrix = torch.from_numpy(rotation_matrix).to(device)
        batch_data[k, :, 0:3] = torch.mm(batch_data[k, :, 0:3], rotation_matrix)
        batch_gt[k, :, 0:3] = torch.mm(batch_gt[k, :, 0:3], rotation_matrix)
        if normal is True:
            batch_data[k, :, 3:6] = torch.mm(batch_data[k, :, 3:6], rotation_matrix)
    return batch_data, batch_gt


def rotate_point_cloud_by_angle(batch_data, batch_gt, normal=False):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    device = batch_data.device
    B, N, C = batch_data.shape

    for k in range(B):
        rotation_region = np.random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        # ICVL
        rotation_angle = rotation_region / 48.0 * 2 * np.pi
        # MSRA
        # rotation_angle = rotation_region / 48.0 * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cosval, -sinval],
                                    [0, sinval,  cosval]], dtype=np.float32)

        rotation_matrix = torch.from_numpy(rotation_matrix).to(device)
        batch_data[k, :, 0:3] = torch.mm(batch_data[k, :, 0:3], rotation_matrix)
        batch_gt[k, :, 0:3] = torch.mm(batch_gt[k, :, 0:3], rotation_matrix)
        if normal is True:
            batch_data[k, :, 3:6] = torch.mm(batch_data[k, :, 3:6], rotation_matrix)
    return batch_data, batch_gt

def rotate_point_cloud_by_angle_flip(batch_data, batch_gt, normal=False):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    device = batch_data.device
    B, N, C = batch_data.shape

    for k in range(B):
        rotation_region = np.random.choice([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # ICVL
        rotation_angle = rotation_region / 48.0 * 2 * np.pi
        # MSRA
        # rotation_angle = rotation_region / 48.0 * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0,  1]], dtype=np.float32)
        if np.random.rand() > 0.5:
            rotation_matrix = np.dot(rotation_matrix, np.array([[-1, 0, 0],
                                                                [0, -1, 0],
                                                                [0, 0,  1]], dtype=np.float32))
        rotation_matrix = torch.from_numpy(rotation_matrix).to(device)
        batch_data[k, :, 0:3] = torch.mm(batch_data[k, :, 0:3], rotation_matrix)
        batch_gt[k, :, 0:3] = torch.mm(batch_gt[k, :, 0:3], rotation_matrix)
        if normal is True:
            batch_data[k, :, 3:6] = torch.mm(batch_data[k, :, 3:6], rotation_matrix)
    
    return batch_data, batch_gt

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def group_points(points, opt, sampling_func='random'):
    # group points using knn and ball query
    # points: B * 1024 * 6
    cur_train_size = len(points)

    permutation = torch.randperm(points.size(1))
    points = points[:,permutation,:]

    centroid_pts = points[:,0:opt.sample_num_level1,0:3]

    inputs1_diff = points[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size,opt.sample_num_level1,3,opt.SAMPLE_NUM) \
                 - centroid_pts[:,:,0:3].unsqueeze(-1).expand(cur_train_size,opt.sample_num_level1,3,opt.SAMPLE_NUM)# B * 512 * 3 * 1024
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 512 * 3 * 1024
    inputs1_diff = inputs1_diff.sum(2)                      # B * 512 * 1024
    dists, inputs1_idx = torch.topk(inputs1_diff, opt.knn_K, 2, largest=False, sorted=False)  # dists: B * 512 * 64; inputs1_idx: B * 512 * 64
        
    # ball query
    invalid_map = dists.gt(opt.ball_radius) # B * 512 * 64
    for jj in range(opt.sample_num_level1):
        inputs1_idx[:,jj,:][invalid_map[:,jj,:]] = jj
        
    idx_group_l1_long = inputs1_idx.view(cur_train_size,opt.sample_num_level1*opt.knn_K,1).expand(cur_train_size,opt.sample_num_level1*opt.knn_K,opt.INPUT_FEATURE_NUM)
    inputs_level1 = points.gather(1,idx_group_l1_long).view(cur_train_size,opt.sample_num_level1,opt.knn_K,opt.INPUT_FEATURE_NUM) # B*512*64*6

    inputs_level1_center = centroid_pts[:,:,0:3].unsqueeze(-2)       # B*512*1*3

    inputs_level1[:,:,:,0:3] = inputs_level1[:,:,:,0:3] - inputs_level1_center.expand(cur_train_size,opt.sample_num_level1,opt.knn_K,3)
    inputs_level1 = inputs_level1.unsqueeze(1).transpose(1,4).squeeze(4)  # B*6*512*64
    inputs_level1_center = inputs_level1_center.contiguous().view(-1,1,opt.sample_num_level1,3).transpose(1,3)  # B*3*512*1
    return inputs_level1, inputs_level1_center
    #inputs_level1: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, inputs_level1_center: B*3*sample_num_level1*1
    
def group_points_2(points, sample_num_level1, sample_num_level2, knn_K, ball_radius, sampling_func='random'):
    # group points using knn and ball query
    # points: B*(3+128)*512
    cur_train_size = points.size(0)

    # permutation = torch.randperm(points.size(2))
    permutation = torch.randperm(points.size(2))
    points = points[:,:,permutation]
    centroid_pts = points[:,:,0:sample_num_level2]


    inputs1_diff = points[:,0:3,:].unsqueeze(1).expand(cur_train_size,sample_num_level2,3,sample_num_level1) \
                 - centroid_pts[:,0:3,:].transpose(1,2).unsqueeze(-1).expand(cur_train_size,sample_num_level2,3,sample_num_level1)# B * 128 * 3 * 512
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 128 * 3 * 512
    inputs1_diff = inputs1_diff.sum(2)                      # B * 128 * 512
    dists, inputs1_idx = torch.topk(inputs1_diff, knn_K, 2, largest=False, sorted=False)  # dists: B * 128 * 64; inputs1_idx: B * 128 * 64
        
    # ball query
    invalid_map = dists.gt(ball_radius) # B * 128 * 64, invalid_map.float().sum()
    #pdb.set_trace()
    for jj in range(sample_num_level2):
        inputs1_idx.data[:,jj,:][invalid_map.data[:,jj,:]] = jj

    idx_group_l1_long = inputs1_idx.view(cur_train_size,1,sample_num_level2*knn_K).expand(cur_train_size,points.size(1),sample_num_level2*knn_K)
    inputs_level2 = points.gather(2,idx_group_l1_long).view(cur_train_size,points.size(1),sample_num_level2,knn_K) # B*131*128*64

    inputs_level2_center = centroid_pts[:,0:3,:].unsqueeze(3)       # B*3*128*1
    
    inputs_level2[:,0:3,:,:] = inputs_level2[:,0:3,:,:] - inputs_level2_center.expand(cur_train_size,3,sample_num_level2,knn_K) # B*3*128*64
    return inputs_level2, inputs_level2_center
    # inputs_level2: B*131*sample_num_level2*knn_K, inputs_level2_center: B*3*sample_num_level2*1

def final_group(joint_coarse, original_points, num_neighbors, ball_radius):
    # group points using knn and ball query
    # points: B * 1024 * 6
    cur_train_size = len(original_points)
    original_length = original_points.size(1)
    joint_length = joint_coarse.size(1)
    feature_width = original_points.size(2)

    diff = original_points[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size, joint_length, 3, original_length) \
                 - joint_coarse[:,:,0:3].unsqueeze(-1).expand(cur_train_size, joint_length, 3, original_length)# B * n * 3 * 21
    diff = torch.mul(diff, diff)    # B * n * 3 * 21
    diff = diff.sum(2)              # B * n * 1* 21
    # diff = diff.transpose(1,2)                   # B * 21 * n
    dists, idx = torch.topk(diff, num_neighbors, 2, largest=False, sorted=False)  # dists: B * 21 * n; inputs1_idx: B * 21 * n
        
    # ball query
    invalid_map = dists.gt(ball_radius) # B * 21 * n
    for jj in range(joint_length):
        idx[:,jj,:][invalid_map[:,jj,:]] = jj
        
    idx_group_l1_long = idx.view(cur_train_size,joint_length*num_neighbors, 1).expand(cur_train_size, joint_length*num_neighbors, feature_width)
    grouped_points = original_points.gather(1,idx_group_l1_long).view(cur_train_size, joint_length, num_neighbors, feature_width) # B*21*64*f

    grouped_points[:,:,:,0:3] = grouped_points[:,:,:,0:3] - joint_coarse[:,:,0:3].unsqueeze(-2).expand(cur_train_size, joint_length, num_neighbors, 3)
    
    return grouped_points
    #inputs_level1: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, inputs_level1_center: B*3*sample_num_level1*1

def knn(point_center, points_samples, num_neighbors):
    cur_train_size = len(point_center)
    center_length = point_center.size(1)
    sample_length = points_samples.size(1)
    feature_width = point_center.size(2)

    diff = point_center[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size, sample_length, 3, center_length) \
                 - points_samples[:,:,0:3].unsqueeze(-1).expand(cur_train_size, sample_length, 3, center_length)# B * n * 3 * 21
    diff = torch.mul(diff, diff)    # B * n * 3 * 21
    diff = diff.sum(2)              # B * n * 1* 21
    # diff = diff.transpose(1,2)                   # B * 21 * n
    dists, idx = torch.topk(diff, num_neighbors, 2, largest=False, sorted=False)  # dists: B * 21 * n; inputs1_idx: B * 21 * n

    return (idx, dists)

def group_points_coarse(points, opt, sampling_func='random'):
    # group points using knn and ball query
    # points: B * 1024 * 6
    cur_train_size = len(points)

    permutation = torch.randperm(points.size(1))
    points = points[:,permutation,:]

    centroid_pts = points[:,0:opt.sample_num_level1,0:3]

    inputs1_diff = points[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size,opt.sample_num_level1,3,opt.SAMPLE_NUM) \
                 - centroid_pts[:,:,0:3].unsqueeze(-1).expand(cur_train_size,opt.sample_num_level1,3,opt.SAMPLE_NUM)# B * 512 * 3 * 1024
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 512 * 3 * 1024
    inputs1_diff = inputs1_diff.sum(2)                      # B * 512 * 1024
    dists, inputs1_idx = torch.topk(inputs1_diff, 64, 2, largest=False, sorted=False)  # dists: B * 512 * 64; inputs1_idx: B * 512 * 64

       
    # ball query
    invalid_map = dists.gt(opt.ball_radius) # B * 512 * 64
    for jj in range(opt.sample_num_level1):
        inputs1_idx[:,jj,:][invalid_map[:,jj,:]] = jj

    permutation = torch.randperm(inputs1_idx.size(2))
    inputs1_idx = inputs1_idx[:,:,permutation[0:opt.knn_K]]
    dists = dists[:,:,permutation[0:opt.knn_K]]


    idx_group_l1_long = inputs1_idx.view(cur_train_size,opt.sample_num_level1*opt.knn_K,1).expand(cur_train_size,opt.sample_num_level1*opt.knn_K,opt.INPUT_FEATURE_NUM)
    inputs_level1 = points.gather(1,idx_group_l1_long).view(cur_train_size,opt.sample_num_level1,opt.knn_K,opt.INPUT_FEATURE_NUM) # B*512*64*6

    inputs_level1_center = centroid_pts[:,:,0:3].unsqueeze(-2)       # B*512*1*3

    inputs_level1[:,:,:,0:3] = inputs_level1[:,:,:,0:3] - inputs_level1_center.expand(cur_train_size,opt.sample_num_level1,opt.knn_K,3)
    inputs_level1 = inputs_level1.unsqueeze(1).transpose(1,4).squeeze(4)  # B*6*512*64
    inputs_level1_center = inputs_level1_center.contiguous().view(-1,1,opt.sample_num_level1,3).transpose(1,3)  # B*3*512*1
    return inputs_level1, inputs_level1_center
    #inputs_level1: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, inputs_level1_center: B*3*sample_num_level1*1
    
def group_points_2_coarse(points, sample_num_level1, sample_num_level2, knn_K, ball_radius, sampling_func='random'):
    # group points using knn and ball query
    # points: B*(3+128)*512
    cur_train_size = points.size(0)

    # permutation = torch.randperm(points.size(2))
    permutation = torch.randperm(points.size(2))
    points = points[:,:,permutation]
    centroid_pts = points[:,:,0:sample_num_level2]


    inputs1_diff = points[:,0:3,:].unsqueeze(1).expand(cur_train_size,sample_num_level2,3,sample_num_level1) \
                 - centroid_pts[:,0:3,:].transpose(1,2).unsqueeze(-1).expand(cur_train_size,sample_num_level2,3,sample_num_level1)# B * 128 * 3 * 512
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 128 * 3 * 512
    inputs1_diff = inputs1_diff.sum(2)                      # B * 128 * 512
    dists, inputs1_idx = torch.topk(inputs1_diff, 64, 2, largest=False, sorted=False)  # dists: B * 128 * 64; inputs1_idx: B * 128 * 64

    # ball query
    invalid_map = dists.gt(ball_radius) # B * 128 * 64, invalid_map.float().sum()
    #pdb.set_trace()
    for jj in range(sample_num_level2):
        inputs1_idx.data[:,jj,:][invalid_map.data[:,jj,:]] = jj

    permutation = torch.randperm(inputs1_idx.size(2))
    inputs1_idx = inputs1_idx[:,:,permutation[0:knn_K]]
    dists = dists[:,:,permutation[0:knn_K]]


    idx_group_l1_long = inputs1_idx.view(cur_train_size,1,sample_num_level2*knn_K).expand(cur_train_size,points.size(1),sample_num_level2*knn_K)
    inputs_level2 = points.gather(2,idx_group_l1_long).view(cur_train_size,points.size(1),sample_num_level2,knn_K) # B*131*128*64

    inputs_level2_center = centroid_pts[:,0:3,:].unsqueeze(3)       # B*3*128*1
    
    inputs_level2[:,0:3,:,:] = inputs_level2[:,0:3,:,:] - inputs_level2_center.expand(cur_train_size,3,sample_num_level2,knn_K) # B*3*128*64
    return inputs_level2, inputs_level2_center
    # inputs_level2: B*131*sample_num_level2*knn_K, inputs_level2_center: B*3*sample_num_level2*1

def grouping(points, k, radius, sample_size=None):
    # group points using knn and ball query
    # points: B * 6 * 1024

    batch_size = points.size(0)
    feature_size = points.size(1)
    point_size = points.size(2)

    if sample_size == None:
        sample_size = point_size

    centroid_pts = points[:,0:3,0:sample_size]

    diff = points[:,0:3,:].unsqueeze(1).expand(batch_size,sample_size,3,point_size) \
    - centroid_pts[:,0:3,:].transpose(1,2).unsqueeze(-1).expand(batch_size,sample_size, 3, point_size)# B * 512 * 3 * 1024
    
    diff = torch.mul(diff, diff)    # B * sample_size * 3 * 1024
    diff = diff.sum(2)              # B * sample_size * input_size
    dists, idx = torch.topk(diff, k, 2, largest=False, sorted=False)  # dists: B * 512 * 64; inputs1_idx: B * 512 * 64
        
    # ball query
    invalid_map = dists.gt(radius) # B * 512 * 64
    for jj in range(sample_size):
        idx[:,jj,:][invalid_map[:,jj,:]] = jj
        
    idx_group_long = idx.view(batch_size,1,sample_size*k).expand(batch_size,3,sample_size*k)
    gathered = points[:,0:3,:].gather(2,idx_group_long).view(batch_size,3,sample_size,k) # B*3*512*64

    center = centroid_pts[:,0:3,:].unsqueeze(-1)       # B*3*512*1

    gathered[:,0:3,:,:] = gathered[:,0:3,:,:] - center.expand(batch_size,3,sample_size,k) # B*3*512*64
    center = center.contiguous().view(-1,3,sample_size,1) # B*3*512*1
    return idx, gathered, center
    #inputs_level1: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, inputs_level1_center: B*3*sample_num_level1*1
    
def random_cut_arm(points):

    B = len(points)
    P = len(points[0])

    for k in range(B):
        arm_len = np.random.rand() * 0.3

        mask = (points[k,:,0] - arm_len) < -0.5
        mask_size = 0
        for i in mask:
            if i == True:
                mask_size = mask_size + 1
        points[k, mask] = points[k, mask==False][np.random.choice(range(P - mask_size), mask_size)]

    return points

