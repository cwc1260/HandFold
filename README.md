
# HandFoldingNet✌️: A 3D Hand Pose Estimation Network Using Multiscale-Feature Guided Folding of a 2D Hand Skeleton
Wencan Cheng, Jae Hyun Park, Jong Hwan Ko

IEEE International Conference on Computer Vision (ICCV), 2021\

arXiv preprint:https://arxiv.org/abs/2108.05545

1. Prepare dataset 

    please download the ICVL and MSRA dataset, and put them under path '[./data/ICVL/](https://github.com/forever1260/HandFold/tree/master/data/ICVL)' and '[./data/MSRA/](https://github.com/forever1260/HandFold/tree/master/data/MSRA)', respectively.

    execute instructions in the '[./preprocess_icvl/](https://github.com/forever1260/HandFold/tree/master/preprocess_icvl)' and '[./preprocess_msra/](https://github.com/forever1260/HandFold/tree/master/preprocess_msra)' for datasets preprocessing 

2. Evaluate

    navigate to "[./train_eval](https://github.com/forever1260/HandFold/tree/master/train_eval)" directory

    execute ``` python3 eval_[dataset name]_folding.py --model [saved model name] --test_path [testing set path]```

    for example on ICVL
    
    ```python3 eval_icvl_folding.py --model netR_SOTA.pth --test_path ../data/ICVL_center_pre0/Testing/```

    or on MSRA
    
    ```python3 eval_msra_folding.py --model netR.pth --test_path ../data/msra_preprocess/```

    we provided the pre-trained models ('[./results/icvlfolding](https://github.com/forever1260/HandFold/tree/master/results/icvlfolding)/netR_SOTA.pth' and '[./results/msrafolding](https://github.com/forever1260/HandFold/tree/master/results/msrafolding])/P0/netR.pth') for testing ICVL and MSRA

    we also provided the predicted labels located at '[./labels](https://github.com/forever1260/HandFold/tree/master/)' directory for visulizing the performance through [awesome-hand-pose-estimation](https://github.com/xinghaochen/awesome-hand-pose-estimation)  

3. If a new training process is needed, please execute the following instructions after step1 is completed

   navigate to "[./train_eval](https://github.com/forever1260/HandFold/tree/master/train_eval)" directory

   . for training MSRA
   
    execute ``` python3 train_msra_folding.py --dataset_path [MSAR dataset path]```
    
    example ``` python3 train_msra_folding.py --dataset_path ../data/msra_preprocess/```


   . for training ICVL
   
   execute ``` python3 train_icvl_folding.py --train_path [ICVL training dataset path] --test_path [ICVL testing dataset path]```
   
If you find our code useful for your research, please cite our paper
```
@inproceedings{cheng2021handfoldingnet,
  title={HandFoldingNet: A 3D Hand Pose Estimation Network Using Multiscale-Feature Guided Folding of a 2D Hand Skeleton},
  author={Cheng, Wencan and Park, Jae Hyun and Ko, Jong Hwan},
  booktitle={IEEE International Conference on Computer Vision ({ICCV})},
  year={2021}
}
```
   
