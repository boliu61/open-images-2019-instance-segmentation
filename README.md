7th place solution to the Open Images 2019 Instance Segmentation competition

Solution summary: https://www.kaggle.com/c/open-images-2019-instance-segmentation/discussion/110983

The code is based on [this commit](https://github.com/open-mmlab/mmdetection/commit/084a3890db88043634ccbb3974f1ef584e19dfd5) of mmdetection library


## (optional) pre-processing

The resulting pkl files from this step is available [here](https://drive.google.com/drive/folders/1aHA7osGpgO-MgvVY7z5WPfSXA-87GQdl), so this step is optional.

#### make train (275 leaf class and 23 parent class) and test annotation pkl
```
python util/make_train_leaf_ann_pkl.py
python util/make_train_parent_ann_pkl.py
python util/make_test_ann_pkl.py
```
#### make re-balanced train annotation pkl
```
python util/make_rebalanced_train_ann.py
python util/make_rebalanced_train_ann_oversample_test.py
python util/make_rebalanced_train_ann_parent.py
```

## (optional) training
The weights files of the 4 single models are [here](https://drive.google.com/drive/folders/1hI_3_h4pyvmBEn-MjrjD1ErOILw_9rqt), so this step is optional too.

Mapping between model and weight file name:
- L1 (backbone x101): `aug_large_ms_lr_test_epoch_1_iter_76000.pth`
- L2 (backbone r101 + deformable module): `aug_large_lr15_dcn_mscale_balanced_rnd2_epoch_1_iter_12000.pth`
- L3 (backbone x101 + deformable module): `x101_dcn_gcp_lr50_ep1_it26000.pth`
- P1 (backbone x101): `parent_A_lr50_ep1_it8000.pth`

#### training commands
Our training and inference was done on Google Cloud with images on Google storage buckets. 

model L1
```
GPU_NUM=4
CONFIG_FILE=configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_colab.py
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
# stop after 760k iterations

CONFIG_FILE=configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_colab_lr5.py
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
# stop after 60k iterations

CONFIG_FILE=configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_colab_lr50.py
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
# stop after 152k iterations
```
model L2
```
GPU_NUM=4
CONFIG_FILE=configs/cascade_mask_rcnn_dconv_c3-c5_r101_fpn_1x_colab.py
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
# stop after 675k iterations

CONFIG_FILE=configs/cascade_mask_rcnn_dconv_c3-c5_r101_fpn_1x_colab_lr3.py
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
# stop after 64k iterations

CONFIG_FILE=configs/cascade_mask_rcnn_dconv_c3-c5_r101_fpn_1x_colab_lr15.py
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
# stop after 48k iterations
```
model L3
```
GPU_NUM=4
CONFIG_FILE=configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_new_fp16_dcn.py
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
# stop after 226k iterations

CONFIG_FILE=configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_new_fp16_dcn_lr5.py
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
# stop after 132k iterations

CONFIG_FILE=configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_new_fp16_dcn_lr50.py
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
# stop after 26k iterations
```
model P1
```
GPU_NUM=4
CONFIG_FILE=configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_colab_parent.py
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 
# stop after 148k iterations

CONFIG_FILE=configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_colab_parent_lr5.py
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
# stop after 62k iterations

CONFIG_FILE=configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_colab_parent_lr50.py
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
# stop after 8k iterations
```

## inference (TTA)
The TTA code is adopted from https://github.com/amirassov/kaggle-imaterialist
#### leaf models: 3 models, 2 scales, and flip
```
epoch_str=aug_large_ms_lr_test_epoch_1_iter_76000
epoch_str2=aug_large_lr15_dcn_mscale_balanced_rnd2_epoch_1_iter_12000
epoch_str3=x101_dcn_gcp_lr50_ep1_it26000
img_scale="1333,800 1600,960"
out_str=avg3_2scale_flip

thres=0
max_per_img=120

for i in {0..24}
do 
    python tools/ensemble_test.py configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_colab.py \
    --cfg_list configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_colab.py \
    configs/cascade_mask_rcnn_dconv_c3-c5_r101_fpn_1x_colab.py \
    configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_new_fp16_dcn.py \
    --checkpoint /home/bo_liu/${epoch_str}.pth \
    /home/bo_liu/${epoch_str2}.pth \
    /home/bo_liu/${epoch_str3}.pth \
    --ann_file test_ann_${i}_of_25.pkl \
    --flip \
    --img_scale ${img_scale} \
    --thres ${thres} --max_per_img ${max_per_img} \
    --out /home/jupyter/LB_${out_str}_thr${thres}_${max_per_img}_${i}of25.pkl
done
```
#### parent model: single model, 2 scales, and flip
```
epoch_str=parent_A_lr50_ep1_it8000
img_scale="1333,800 1600,960"
out_str=parent_lr50_8k_2scale_flip

thres=0
max_per_img=120

for i in {0..24}
do 
    python tools/ensemble_test.py configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_colab_parent.py \
    --cfg_list configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_colab_parent.py \
    --checkpoint /home/bo_liu/${epoch_str}.pth \
    --ann_file test_ann_${i}_of_25.pkl \
    --flip \
    --img_scale ${img_scale} \
    --thres ${thres} --max_per_img ${max_per_img} \
    --out /home/jupyter/LB_${out_str}_thr${thres}_${max_per_img}_${i}of25.pkl
done
```

## post-processing
#### convert output pkl files to submission csv formats
```
# leaf models
file_prefix=LB_avg3_2scale_flip_thr0_120
parent=0
for i in {0..24}
do 
  python util/convert_seg_results_to_sub_25.py --pkl_path LB_pkl/${file_prefix}_${i}of25.pkl \
  --parent=${parent}
done
```

```
# parent model
file_prefix=LB_parent_lr50_8k_2scale_flip_thr0_120
parent=1
for i in {0..24}
do 
  python util/convert_seg_results_to_sub_25.py --pkl_path LB_pkl/${file_prefix}_${i}of25.pkl \
  --parent=${parent}
done
```
#### expand to parent classes
```
# expand the 275 classes from leaf model to 25 parent/grandparent classes
file_prefix=LB_avg3_2scale_flip_thr0_120
python util/seg_expand_and_adjust_thres_25.py --sub_csv_pattern "LB_csv/${file_prefix}_*of25.csv" --thres 0 --parents_only=1
```
```
# expand the 23 parent classes from parent model to 25 parent/grandparent classes
file_prefix=LB_parent_lr50_8k_2scale_flip_thr0_120
python util/seg_expand_and_adjust_thres_25.py --sub_csv_pattern "LB_csv/${file_prefix}_*of25.csv" --thres 0
```
#### nms for parent/grandparent classescascade_mask_rcnn_x101_64x4d_fpn_1x_colab
```
thres=0
iou_thr=0.5
model_dir=LB_csv/
for i in {0..24}
do
  csv_1=LB_avg3_2scale_flip_thr0_120_${i}of25_expand_thr0_25cls.csv
  csv_2=LB_parent_lr50_8k_2scale_flip_thr0_120_${i}of25.csv
  out_csv=LB_avg3_2scale_flip_NMS_G8k_2scale_flip_thr${thres}_${iou_thr}_${i}of25.csv
  python util/nms_on_csvs.py --single_or_two=two --thres=${thres} --iou_thr=${iou_thr} \
        --csv_path_1=${model_dir}${csv_1} --csv_path_2=${model_dir}${csv_2} \
        --out_path=${model_dir}${out_csv}
done
```

#### finally, combine the leaf classes and parent/grandparent classes
```
python combine_leaf_and_parent.py
```



