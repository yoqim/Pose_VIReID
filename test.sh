#!/bin/bash

###### regdb ###### 
# python test34.py --dataset regdb --gpu 0 --trial 1 --resume 'pose_model34_regdb_trial1_pwei10_kdwei1_refine1_x3add_3RS_swap_best.t'
# python test36.py --dataset regdb --gpu 0 --trial 3 --resume 'model36_regdb_trial3_deit_small_patch16_stride16_triplet0.1_best.t'
# python test_ml.py --dataset regdb --gpu 1 --trial 2 --resume 'ML_regdb_trial2_mask_erase_after_ts_best.t'



###### sysu ###### 
# python test34.py --dataset sysu --gpu 0 --resume 'pose_model33_pwei7_refine1_x3add_3RS_best.t' --mode 'indoor' 
# python test36.py --dataset sysu --gpu 0 --resume 'model36_deit_small_patch16_stride16_triplet0.1_best.t' --vis
python test_ml.py --dataset sysu --gpu 0 --resume 'ML_mask_erase_after_best.t' --mode 'all'
