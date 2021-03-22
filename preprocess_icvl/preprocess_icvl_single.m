save_dir='../data/ICVL_center_single1/';

if ~ exist(save_dir,'dir')
    mkdir(save_dir);
end


flag = 'Train'; % 'Train' or 'Test'

fFocal_MSRA_ = 241.42;	% mm

if strcmp(flag,'Train')
    dataset_dir='/workspace/workspace/Hand-Pointnet/preprocess/icvl/Training/';
    labelsID = fopen([dataset_dir '/labels.txt']);
    mkdir([save_dir 'Training/']);
    recordID = fopen([save_dir 'Training/record.txt'], 'a');
elseif strcmp(flag,'Test')
    dataset_dir='/workspace/workspace/Hand-Pointnet/preprocess/icvl/Testing/';
    labelsID = fopen([dataset_dir '/test_merge.txt']);
    mkdir([save_dir 'Testing/']);
    recordID = fopen([save_dir 'Testing/record.txt'], 'a');
end
labels = textscan(labelsID,'%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f');

predID = fopen('./saved_fold1.txt');
preds = textscan(predID,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f');

JOINT_NUM = 16;
SAMPLE_NUM = 1024;
sample_num_level1 = 512;
sample_num_level2 =  128;                                                                                                                                                                                                                        
for file_idx = 1:length(labels{1})
    
    if (file_idx == 205510 || file_idx == 304574|| file_idx == 304667)
        continue;
    end


    file_names = strsplit(labels{1}{file_idx}, {'/','.'});

    if (length(file_names{1}) < 12)
        continue;
    end

    display(file_idx);
    display(labels{1}{file_idx});
    
    if strcmp(flag,'Train')
        save_pos = [save_dir 'Training/' file_names{1} '/' ]
        mkdir(save_pos);
    elseif strcmp(flag,'Test')
        save_pos = [save_dir 'Testing/' file_names{1} '/' ]
        mkdir(save_pos);
    end   

    if ~ exist([dataset_dir 'Depth/' labels{1}{file_idx}],'file')
        continue;
    end
    
    depth = imread([dataset_dir 'Depth/' labels{1}{file_idx}]);  
    
    joints = [ labels{2}(file_idx), labels{3}(file_idx),labels{4}(file_idx);
        labels{5}(file_idx),labels{6}(file_idx),labels{7}(file_idx);
        labels{8}(file_idx),labels{9}(file_idx),labels{10}(file_idx);
        labels{11}(file_idx),labels{12}(file_idx),labels{13}(file_idx);
        labels{14}(file_idx),labels{15}(file_idx),labels{16}(file_idx);
        labels{17}(file_idx),labels{18}(file_idx),labels{19}(file_idx);
        labels{20}(file_idx),labels{21}(file_idx),labels{22}(file_idx);
        labels{23}(file_idx),labels{24}(file_idx),labels{25}(file_idx);
        labels{26}(file_idx),labels{27}(file_idx),labels{28}(file_idx);
        labels{29}(file_idx),labels{30}(file_idx),labels{31}(file_idx);
        labels{32}(file_idx),labels{33}(file_idx),labels{34}(file_idx);
        labels{35}(file_idx),labels{36}(file_idx),labels{37}(file_idx);
        labels{38}(file_idx),labels{39}(file_idx),labels{40}(file_idx);
        labels{41}(file_idx),labels{42}(file_idx),labels{43}(file_idx);
        labels{44}(file_idx),labels{45}(file_idx),labels{46}(file_idx);
        labels{47}(file_idx),labels{48}(file_idx),labels{49}(file_idx)];
    
    palm = [labels{2}(file_idx), labels{3}(file_idx),labels{4}(file_idx)];

    palm_left = floor(palm(:,1) - 30);
    palm_right = ceil(palm(:,1) + 30);
    palm_top = floor(palm(:,2) - 30);
    palm_bottom = ceil(palm(:,2) + 30);
    
    palm_front = floor(palm(:,3) - 30);
    palm_back = ceil(palm(:,3) + 30);
    
    
    left = max(min(floor(min(joints(:,1)) - 20), palm_left), 1);
    right = min(max(ceil(max(joints(:,1)) + 20), palm_right), 320);
    top =  max(min(floor(min(joints(:,2)) - 20), palm_top), 1);
    bottom = min(max(ceil(max(joints(:,2)) + 20), palm_bottom), 240);
    back = max(ceil(max(joints(:,3)) + 20), palm_back);
    front =  min(floor(min(joints(:,3)) - 20), palm_front);

    depth = depth(top:bottom, left:right);



    % 2. get point cloud and surface normal

    Point_Cloud_FPS = zeros(SAMPLE_NUM,6);
    Volume_rotate = zeros(3,3);
    Volume_length = zeros(1);
    Volume_offset = zeros(3);
    Volume_GT_XYZ = zeros(JOINT_NUM,3);



    %% 2.1 read binary file
    img_width = 320;
    img_height = 240;

    bb_width = right - left;
    bb_height = bottom - top;

    valid_pixel_num = bb_width*bb_height;

%     depth = depth';

    %% 2.2 convert depth to xyz
    fFocal_MSRA_ = 241.42;	% mm
    hand_3d = zeros(valid_pixel_num,3);
    for ii=1:bb_height
        for jj=1:bb_width
            if  (depth(ii,jj) < front ||  depth(ii,jj) > back)
                continue;
            end
            idx = (jj-1)*bb_height+ii;
            hand_3d(idx, 1) = -(img_width/2.0 - (jj+left-1))*double(depth(ii,jj))/fFocal_MSRA_;
            hand_3d(idx, 2) = (img_height/2.0 - (ii+top-1))*double(depth(ii,jj))/fFocal_MSRA_;
            hand_3d(idx, 3) = depth(ii,jj);
        end
    end

    valid_idx = 1:valid_pixel_num;
    valid_idx = valid_idx(hand_3d(:,1)~=0 | hand_3d(:,2)~=0 | hand_3d(:,3)~=0);
    hand_points = hand_3d(valid_idx,:);
    
    if(length(hand_points) <= 0)
        continue;
    end
    
    jnt_xyz = zeros(length(joints),3);
    for j=1:length(joints)
            jnt_xyz(j, 1) = -(img_width/2 - joints(j,1))*joints(j,3)/fFocal_MSRA_;
            jnt_xyz(j, 2) = (img_height/2 - joints(j,2))*joints(j,3)/fFocal_MSRA_;
            jnt_xyz(j, 3) = joints(j,3);
    end
%     jnt_xyz = squeeze(gt_wld(frm_idx,:,:));

    %% 2.3 create OBB
    [coeff,score,latent] = pca(hand_points);
    if coeff(2,1)<0
        coeff(:,1) = -coeff(:,1);
    end
    if coeff(3,3)<0
        coeff(:,3) = -coeff(:,3);
    end
    coeff(:,2)=cross(coeff(:,3),coeff(:,1));

    ptCloud = pointCloud(hand_points);

    hand_points_rotate = hand_points*coeff;

    %% 2.4 sampling
    if size(hand_points,1)<SAMPLE_NUM
        tmp = floor(SAMPLE_NUM/size(hand_points,1));
        rand_ind = [];
        for tmp_i = 1:tmp
            rand_ind = [rand_ind 1:size(hand_points,1)];
        end
        rand_ind = [rand_ind randperm(size(hand_points,1), mod(SAMPLE_NUM, size(hand_points,1)))];
    else
        rand_ind = randperm(size(hand_points,1),SAMPLE_NUM);
    end
    hand_points_sampled = hand_points(rand_ind,:);
    hand_points_rotate_sampled = hand_points_rotate(rand_ind,:);

    %% 2.5 compute surface normal
    normal_k = 30;
    normals = pcnormals(ptCloud, normal_k);
    normals_sampled = normals(rand_ind,:);

    sensorCenter = [0 0 0];
    for k = 1 : SAMPLE_NUM
       p1 = sensorCenter - hand_points_sampled(k,:);
       % Flip the normal vector if it is not pointing towards the sensor.
       angle = atan2(norm(cross(p1,normals_sampled(k,:))),p1*normals_sampled(k,:)');
       if angle > pi/2 || angle < -pi/2
           normals_sampled(k,:) = -normals_sampled(k,:);
       end
    end
    normals_sampled_rotate = normals_sampled*coeff;

    %% 2.6 Normalize Point Cloud
    x_min_max = [min(hand_points_rotate(:,1)), max(hand_points_rotate(:,1))];
    y_min_max = [min(hand_points_rotate(:,2)), max(hand_points_rotate(:,2))];
    z_min_max = [min(hand_points_rotate(:,3)), max(hand_points_rotate(:,3))];

    scale = 1.2;
    bb3d_x_len = scale*(x_min_max(2)-x_min_max(1));
    bb3d_y_len = scale*(y_min_max(2)-y_min_max(1));
    bb3d_z_len = scale*(z_min_max(2)-z_min_max(1));
    max_bb3d_len = bb3d_x_len;

    hand_points_normalized_sampled = hand_points_rotate_sampled/max_bb3d_len;
    if size(hand_points,1)<SAMPLE_NUM
        offset = mean(hand_points_rotate)/max_bb3d_len;
    else
        offset = mean(hand_points_normalized_sampled);
    end
    hand_points_normalized_sampled = hand_points_normalized_sampled - repmat(offset,SAMPLE_NUM,1);

    %% 2.7 FPS Sampling
    pc = [hand_points_normalized_sampled normals_sampled_rotate];
    % 1st level
    sampled_idx_l1 = farthest_point_sampling_fast(hand_points_normalized_sampled, sample_num_level1)';
    other_idx = setdiff(1:SAMPLE_NUM, sampled_idx_l1);
    new_idx = [sampled_idx_l1 other_idx];
    pc = pc(new_idx,:);
    % 2nd level
    sampled_idx_l2 = farthest_point_sampling_fast(pc(1:sample_num_level1,1:3), sample_num_level2)';
    other_idx = setdiff(1:sample_num_level1, sampled_idx_l2);
    new_idx = [sampled_idx_l2 other_idx];
    pc(1:sample_num_level1,:) = pc(new_idx,:);

    %% 2.8 ground truth
    jnt_xyz_normalized = (jnt_xyz*coeff)/max_bb3d_len;
    jnt_xyz_normalized = jnt_xyz_normalized - repmat(offset,JOINT_NUM,1);

    Point_Cloud_FPS(:,:) = pc;
    Volume_rotate(:,:) = coeff;
    Volume_length = max_bb3d_len;
    Volume_offset = offset;
    Volume_GT_XYZ(:,:) = jnt_xyz_normalized;

    % 3. save files
    if strcmp(flag,'Train')
        save([save_pos file_names{2} '_Point_Cloud_FPS.mat'],'Point_Cloud_FPS');
        fprintf(recordID, '%s ', [file_names{1} '/' file_names{2}  ]);
    elseif strcmp(flag,'Test')
        save([save_pos file_names{2} '_Point_Cloud_FPS.mat'],'Point_Cloud_FPS');
        fprintf(recordID, '%s ', [file_names{1} '/' file_names{2}]);    
    end
    
    fprintf(recordID, '%.6f ', Volume_length);
    
    for ri=1:16
        for rj=1:3
             fprintf(recordID, '%.6f ', Volume_GT_XYZ(ri,rj));
        end
    end
    
    fprintf(recordID, '%.6f ', Volume_offset);
    
    for ri=1:3
        for rj =1:3
             fprintf(recordID, '%.6f ', Volume_rotate(ri,rj));
        end
    end
    fprintf(recordID, '\r\n');
    
end
fclose(recordID);
