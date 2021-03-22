import torch
import torch.nn as nn
import math
from utils import group_points_2, final_group
import numpy as np

encoder_1 = [32,32,128]
encoder_2 = [64,64,256]
encoder_3 = [128,128,512]


folding_1 = [256, 256, 128]
folding_2 = [256, 256, 256, 256, 256]
folding_3 = [256, 256, 256, 256, 256]



relative_idx1 = np.array([1, 
				0, 1, 2, 
				0, 4, 5,  
				0, 7, 8,  
				0, 10, 11,  
				0, 13, 14])

relative_idx2 = np.array([13, 
				2, 3, 3, 
				5, 6, 6,  
				8, 9, 9,  
				11, 12, 12,  
				14, 15, 15])


class PointNet_Plus(nn.Module):
    def __init__(self, opt):
        super(PointNet_Plus, self).__init__()
        self.num_outputs = opt.PCA_SZ
        self.knn_K = opt.knn_K
        self.ball_radius2 = opt.ball_radius2
        self.sample_num_level1 = opt.sample_num_level1
        self.sample_num_level2 = opt.sample_num_level2
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM
        
        self.netR_1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM, encoder_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(encoder_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(encoder_1[0], encoder_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(encoder_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(encoder_1[1], encoder_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(encoder_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            # B*128*sample_num_level1*1
        )
        
        self.netR_2 = nn.Sequential(
            # B*131*sample_num_level2*knn_K
            nn.Conv2d(3+encoder_1[2], encoder_2[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(encoder_2[0]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(encoder_2[0], encoder_2[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(encoder_2[1]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(encoder_2[1], encoder_2[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(encoder_2[2]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            # B*256*sample_num_level2*1
        )
        
        self.netR_3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(3+encoder_2[2], encoder_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(encoder_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(encoder_3[0], encoder_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(encoder_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(encoder_3[1], encoder_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(encoder_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            nn.MaxPool2d((self.sample_num_level2,1),stride=1),
            # B*1024*1*1
        )

        # self.folding = torch.Tensor([
        #                              [0, -0.15, -0.15, -0.15, -0.15,  0,     0,     0,     0,    0.175, 0.175, 0.175, 0.175, 0.3,  0.3,  0.3,  0.3,  -0.12,  -0.12,  -0.12,  -0.12,  ],
        #                              [0,  0.26,  0.4,   0.5,   0.6,   0.33,  0.49,  0.59,  0.69, 0.3,   0.45,  0.55,  0.65,  0.175,0.275, 0.35, 0.425,  0.07,  0.2,  0.3,  0.4]], requires_grad=False).cuda()


        self.netFolding1_1 = nn.Sequential(
            # B*1024
            nn.Conv2d(2+encoder_3[2], folding_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(folding_1[0]),
            nn.ReLU(inplace=True),
            # B*1024
            nn.Conv2d(folding_1[0], folding_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(folding_1[1]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Conv2d(folding_1[1], folding_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(folding_1[2]),
            nn.ReLU(inplace=True),
            # nn.Linear(nstates_plus_3[4], self.num_outputs),
            # B*num_outputs
        )

        self.netFolding1_2 = nn.Sequential(
            nn.Conv2d(folding_1[2], 3, kernel_size=(1, 1)),
        )

        self.netFolding2_1 = nn.Sequential(
            # B*131*sample_num_level2*knn_K
            nn.Conv2d(3+folding_1[2]*3+3+encoder_1[2], folding_2[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(folding_2[0]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(folding_2[0], folding_2[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(folding_2[1]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(folding_2[1], folding_2[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(folding_2[2]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*knn_K
            nn.MaxPool2d((64,1),stride=1)
        )


        self.netFolding2_2 = nn.Sequential(
            # B*1024
            nn.Conv2d(folding_2[2], folding_2[3], kernel_size=(1, 1)),
            nn.BatchNorm2d( folding_2[3]),
            # nn.Linear(nstates_plus_3[2], nstates_plus_3[3]),
            # nn.BatchNorm1d(nstates_plus_3[3]),
            nn.ReLU(inplace=True),
            # B*1024
            nn.Conv2d(folding_2[3],  folding_2[4], kernel_size=(1, 1)),
            nn.BatchNorm2d(folding_2[4]),
            # nn.Linear(nstates_plus_3[3], nstates_plus_3[4]),
            # nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Conv2d(folding_2[4], 3, kernel_size=(1, 1)),
            # nn.Linear(nstates_plus_3[4], self.num_outputs),
            # B*num_outputs

            # B*256*sample_num_level2*1
        )
        self.netFolding3_1 = nn.Sequential(
            # B*131*sample_num_level2*knn_K
            nn.Conv2d(3+folding_2[2]*3+3+encoder_1[2], folding_2[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(folding_3[0]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(folding_3[0], folding_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(folding_3[1]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(folding_3[1], folding_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(folding_3[2]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*knn_K
            nn.MaxPool2d((64,1),stride=1)

        )
        self.netFolding3_2 = nn.Sequential(
            # B*1024
            nn.Conv2d(folding_3[2], folding_3[3], kernel_size=(1, 1)),
            nn.BatchNorm2d( folding_3[3]),
            # nn.Linear(nstates_plus_3[2], nstates_plus_3[3]),
            # nn.BatchNorm1d(nstates_plus_3[3]),
            nn.ReLU(inplace=True),
            # B*1024
            nn.Conv2d(folding_3[3],  folding_3[4], kernel_size=(1, 1)),
            nn.BatchNorm2d(folding_3[4]),
            # nn.Linear(nstates_plus_3[3], nstates_plus_3[4]),
            # nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Conv2d(folding_3[4], 3, kernel_size=(1, 1)),
            # nn.Linear(nstates_plus_3[4], self.num_outputs),
            # B*num_outputs
            # B*256*sample_num_level2*1
        )

    def forward(self, x, y):
        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*3*sample_num_level1*1

        x = self.netR_1(x)
        # B*128*sample_num_level1*1
        level1 = torch.cat((y, x),1).squeeze(-1)
        # B*(3+128)*sample_num_level1
        
        inputs_level2, inputs_level2_center = group_points_2(level1, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        # B*131*sample_num_level2*knn_K, B*3*sample_num_level2*1
        
        # B*131*sample_num_level2*knn_K
        x = self.netR_2(inputs_level2)
        # B*256*sample_num_level2*1
        x = torch.cat((inputs_level2_center, x),1)
        # B*259*sample_num_level2*1
        
        del inputs_level2, inputs_level2_center

        code = self.netR_3(x)

        # del x
        # B*512*1*1

        skeleton = torch.from_numpy(np.array([
                                     [0,   
                                     0.15,   0.25,    0.35,  
                                     0.21,   0.21,    0.21,  
                                     0,      0,       0,    
                                     -0.125, -0.125,  -0.125, 
                                     -0.17,  -0.17,   -0.17,],

                                     [0,  
                                     0 ,    0.1,   0.2,  
                                     0.21,  0.36,  0.46,   
                                     0.3,   0.45,  0.55, 
                                     0.22,  0.37,  0.47,  
                                     0.1,   0.21,  0.3]], dtype=np.float32)).cuda()

        skeleton = skeleton.unsqueeze(0).unsqueeze(-2).expand(x.size(0),2, 1, 16)
        
        code = code.expand(x.size(0),encoder_3[2], 1, 16)
        x = torch.cat((skeleton, code),1)
        # B*(2+512)*1*21
        fold1_1 = self.netFolding1_1(x)
        fold1 = self.netFolding1_2(fold1_1)
        # B*3*1*21

        relative_idx1_cuda = torch.from_numpy(relative_idx1).cuda().unsqueeze(0).unsqueeze(0).expand(fold1_1.size(0), fold1_1.size(1), 16)
        relative_idx2_cuda = torch.from_numpy(relative_idx2).cuda().unsqueeze(0).unsqueeze(0).expand(fold1_1.size(0), fold1_1.size(1), 16)

        rel1 = torch.gather(fold1_1.squeeze(2), 2, relative_idx1_cuda).unsqueeze(-2).expand(fold1_1.size(0), fold1_1.size(1), 1, 16)
        rel2 = torch.gather(fold1_1.squeeze(2), 2, relative_idx2_cuda).unsqueeze(-2).expand(fold1_1.size(0), fold1_1.size(1), 1, 16)

        link1 = final_group(fold1.squeeze(2).transpose(1,2), level1.squeeze(-1).transpose(1,2), 64, 0.16).transpose(1, 3)

        x = torch.cat((fold1.expand(x.size(0), 3, 64, 16), 
                        fold1_1.expand(x.size(0), folding_1[2], 64, 16),
                        rel1.expand(x.size(0), folding_1[2], 64, 16), 
                        rel2.expand(x.size(0), folding_1[2], 64, 16), 
                        link1),1)

        # del fold1_1, rel1, rel2, level1

        # B*(3+730+128)*64*21
        fold2_1 = self.netFolding2_1(x)
        # B*256*1*21
       
        x = self.netFolding2_2(fold2_1)
        # B*3*1*21

        fold2 = x + fold1


        relative_idx1_cuda = torch.from_numpy(relative_idx1).cuda().unsqueeze(0).unsqueeze(0).expand(fold2_1.size(0), fold2_1.size(1), 16)
        relative_idx2_cuda = torch.from_numpy(relative_idx2).cuda().unsqueeze(0).unsqueeze(0).expand(fold2_1.size(0), fold2_1.size(1), 16)

        rel2_1 = torch.gather(fold2_1.squeeze(2), 2, relative_idx1_cuda).unsqueeze(-2).expand(fold2_1.size(0), fold2_1.size(1), 1, 16)
        rel2_2 = torch.gather(fold2_1.squeeze(2), 2, relative_idx2_cuda).unsqueeze(-2).expand(fold2_1.size(0), fold2_1.size(1), 1, 16)

        link2 = final_group(fold2.squeeze(2).transpose(1,2), level1.squeeze(-1).transpose(1,2), 64, 0.16).transpose(1, 3)

        x = torch.cat((fold2.expand(x.size(0), 3, 64, 16), 
                        fold2_1.expand(x.size(0), folding_2[2], 64, 16),
                        rel2_1.expand(x.size(0), folding_2[2], 64, 16), 
                        rel2_2.expand(x.size(0), folding_2[2], 64, 16), 
                        link2),1)

        fold3_1 = self.netFolding3_1(x)
        # B*256*1*21
       
        x = self.netFolding3_2(fold3_1)        

        fold3 = x + fold2

        fold3 = fold3.transpose(1,3).contiguous().view(-1,48)
        fold2 = fold2.transpose(1,3).contiguous().view(-1,48)
        fold1 = fold1.transpose(1,3).contiguous().view(-1,48)


        # B*(21*3)
        # B*(21*3)
        
        return fold1, fold2, fold3