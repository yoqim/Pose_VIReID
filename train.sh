
#!/bin/bash


###### sysu ###### 
# python train31.py --dataset sysu --gpu 0 --lr 1e-2 --suffix '_pwei5_refine2'
# python train31.py --dataset sysu --gpu 1 --resume 'model31_re45_IDfeat_best.t' --lr 1e-2 --suffix "_re45_IDfeat_3"

# python train32.py --dataset sysu --gpu 0 --lr 1e-2 --suffix '_pwei5_refine1_x3add' 
# python train32.py --dataset sysu --gpu 0 --lr 1e-2 --suffix '_pwei1e-2_refine1_x3add_l2' --resume 'pose_pwei5_refine1_x3add_pre_best.t'
# python train33.py --dataset sysu --gpu 1 --lr 1e-2 --suffix '_pwei10_refine1_x3add_3RS' 
# python train34.py --dataset sysu --gpu 1 --lr 1e-2 --suffix '_pwei10_noLp' --resume 'pose_model34_pwei10_kdwei0.5_refine1_x3add_3RS_swap_best.t'
# python train35.py --dataset sysu --gpu 0 --lr 1e-2 --suffix '_pwei3_wLpose'
# python train_ml.py --dataset sysu --gpu 1 --lr 1e-2 --suffix '_mask_erase_after_wo3'
# python train_ml2.py --dataset sysu --gpu 0 --lr 1e-2 --suffix '_debug'


###### regdb ###### 
# python train31.py --dataset regdb --gpu 1 --trial 1 --lr 1e-2 
# python train32.py --dataset regdb --gpu 1 --trial 1 --lr 1e-2 --suffix '_pwei7_refine1_x3add' 
# python train33.py --dataset regdb --gpu 0 --trial 3 --lr 1e-2 --suffix '_pwei7_refine1_x3add_3RS_noflip' 
# python train34.py --dataset regdb --gpu 0 --trial 8 --lr 1e-2 --suffix '_pwei7_kdwei1_hcwei0.1_swap' 

# --resume 'pose_model34_regdb_trial5_pwei10_kdwei1_refine1_x3add_3RS_swap_best'

# python train35.py --dataset regdb --gpu 1 --trial 6 --lr 1e-2 --suffix '_debug'
# python train36.py --dataset regdb --gpu 0 --trial 3 --lr 1e-4 --suffix '_debug'
# python train_ml.py --dataset regdb --gpu 1 --trial 1 --lr 1e-2 --suffix '_KL1.25'
python train_ml.py --dataset regdb --gpu 0 --trial 5 --lr 1e-2 --suffix '_mask_erase_after_w3'
# python train_ml2.py --dataset regdb --gpu 1 --trial 1 --lr 1e-2 --suffix '_merge*_sigmoid_ML'
